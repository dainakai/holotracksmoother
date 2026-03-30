using Statistics
using LinearAlgebra
using Distributions: MvNormal, Normal, pdf, logpdf, loglikelihood # 確率分布関連
using DataFrames           # データフレーム操作
using CSV                  # CSVファイル読み込み
using Glob                 # ファイルパス検索
using Plots                # グラフ描画
using ProgressMeter        # 進捗表示
using Parameters           # パラメータ管理用 @with_kw
using LogExpFunctions      # logsumexp など数値安定化用
using Printf             # フォーマット出力
using Formatting
using Printf: @sprintf
using UUIDs
using YAML
using SavitzkyGolay
using ImageMagick

# ---------------------------------------------------------------------------
# 0. データ構造と型定義
# ---------------------------------------------------------------------------

struct Trajectory
    filename::String   # 元データファイル名
    id::Int          # 軌跡ID
    times::Vector{Float64} # 時間 (ここではフレーム番号をFloat64で扱う)
    x::Vector{Float64}   # x座標
    y::Vector{Float64}   # y座標
    z::Vector{Float64}   # z座標
    diameter::Vector{Float64} # 直径
    uuids::Vector{UUID}  # 各点のUUID
    n_points::Int      # データ点数
end

# 事後分布 (ガウス分布) のパラメータ
struct GaussianPosterior
    mean::Vector{Float64}
    cov::Matrix{Float64}
    precision::Matrix{Float64} # 精度行列も保持しておくと便利な場合がある
    chol_precision::Cholesky{Float64, Matrix{Float64}} # 精度行列のコレスキー分解
end

# EMアルゴリズムの収束履歴用
mutable struct EMHistory
    log_likelihood::Vector{Float64} # (近似的な) 対数尤度 or Q関数値
    params::Dict{Symbol, Vector{Any}} # 各パラメータの履歴
    diff::Vector{Float64}           # パラメータ変化量の履歴
end

# 推定結果全体をまとめる構造体
struct EstimationResult
    d::Int
    alpha_xz::Float64
    alpha_y::Float64
    mu_xz::Vector{Float64}
    Sigma_xz::Matrix{Float64}
    sigma2x::Float64
    beta_x_posteriors::Vector{GaussianPosterior}
    mu_y::Vector{Float64}
    Sigma_y::Matrix{Float64}
    sigma2y::Float64
    beta_y_posteriors::Vector{GaussianPosterior}
    pi_z::Float64
    sigma2z0::Float64
    sigma2z1::Float64
    beta_z_map::Vector{Vector{Float64}} # MAP推定値
    responsibilities::Vector{Vector{Float64}} # 各zデータ点の負担率
    history::Dict{Symbol, EMHistory} # 各ステップの収束履歴
    model_params_selection_results_x::Vector{Tuple{Int, Float64, Float64}} # モデル選択結果
    model_params_selection_results_y::Vector{Tuple{Int, Float64, Float64}} # モデル選択結果
    trajs::Vector{Trajectory} # 元の軌跡データ  
end

# ---------------------------------------------------------------------------
# 1. 補助関数
# ---------------------------------------------------------------------------

"""
    polynomial_basis(t::Float64, d::Int) -> Vector{Float64}

時刻 t における d 次多項式基底ベクトル [1, t, t^2, ..., t^d] を計算する。
"""
function polynomial_basis(t::Float64, d::Int)::Vector{Float64}
    return [t^k for k in 0:d]
end

"""
    design_matrix(times::Vector{Float64}, d::Int) -> Matrix{Float64}

時刻ベクトル `times` に対する計画行列 Φ (n x (d+1)) を計算する。
Φ[j, k+1] = times[j]^k
"""
function design_matrix(times::Vector{Float64}, d::Int)::Matrix{Float64}
    n = length(times)
    Φ = Matrix{Float64}(undef, n, d + 1)
    for k in 0:d
        Φ[:, k+1] = times .^ k
    end
    return Φ
end

# --- 追加（修正の要）: 各軌跡の先頭を t=0 に揃えるためのユーティリティ ---
@inline relative_times(times::AbstractVector{<:Real}) = Float64.(times .- first(times))
@inline relative_times(traj::Trajectory) = relative_times(traj.times)
@inline function design_matrix_zeroed(traj::Trajectory, d::Int)::Matrix{Float64}
    return design_matrix(relative_times(traj), d)
end
# --------------------------------------------------------------------------

"""
    regularization_matrix(d::Int; type::Symbol=:L2_sq) -> Matrix{Float64}

正則化構造を定義する対角行列 D ((d+1) x (d+1)) を計算する。
D[1,1] = 0 (定数項は正則化しない)
type = :L2_sq -> D[k+1, k+1] = k^2
type = :L2    -> D[k+1, k+1] = k
"""
function regularization_matrix(d::Int; type::Symbol=:L2_sq)::Matrix{Float64}
    D = zeros(Float64, d + 1, d + 1)
    if type == :L2_sq
        for k in 1:d
            D[k+1, k+1] = Float64(k^2)
        end
    elseif type == :L2
        for k in 1:d
            D[k+1, k+1] = Float64(k)
        end
    else # :Ridge or unknown
         for k in 1:d
            D[k+1, k+1] = 1.0
        end
    end
    return D
end

"""
    precision_matrix(d::Int, alpha::Float64, D::Matrix{Float64};
                     lambda0::Float64=1e-6, lambda_other::Float64=1.0) -> Matrix{Float64}

事前分布の精度行列 Σ⁻¹ = Λ₀ + αD を計算する。
"""
function precision_matrix(d::Int, alpha::Float64, D::Matrix{Float64};
                          lambda0::Float64=1e-6, lambda_other::Float64=1.0)::Matrix{Float64}
    Lambda0 = Diagonal([lambda0; fill(lambda_other, d)])
    return Lambda0 + alpha * D
end

"""
    safe_inv(M::AbstractMatrix) -> Matrix{Float64}

数値的に安定な逆行列計算（正定値対称行列を想定）。
コレスキー分解が失敗した場合や逆行列が計算できない場合にエラーを出す。
"""
function safe_inv(M::AbstractMatrix{<:Real})::Matrix{Float64}
    if !issymmetric(M)
        @warn "Input matrix is not symmetric. Attempting cholesky anyway."
    end
    try
        C = cholesky(Hermitian(M)) # Hermitianで対称性を保証
        return C \ I
    catch e
        if isa(e, PosDefException)
            @error "Matrix is not positive definite. Cannot compute inverse via Cholesky."
            throw(e)
        else
            rethrow(e)
        end
    end
end

"""
    log_gaussian_pdf(x, μ, σ²) -> Float64

ガウス分布の対数確率密度関数 log(N(x | μ, σ²))。
"""
function log_gaussian_pdf(x::Real, μ::Real, σ²::Real)::Float64
    if σ² <= 0
        return -Inf
    end
    return -0.5 * log(2π * σ²) - 0.5 * (x - μ)^2 / σ²
end

"""
    log_mv_gaussian_pdf(x::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix) -> Float64

多変量ガウス分布の対数確率密度関数 log(N(x | μ, Σ))。
数値安定性のために Cholesky 分解を使用。
"""
function log_mv_gaussian_pdf(x::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix{<:Real})::Float64
    k = length(x)
    try
        C = cholesky(Hermitian(Σ))
        logdetΣ = logdet(C)
        z = C.L \ (x - μ)
        quad_form = dot(z, z)
        return -0.5 * (k * log(2π) + logdetΣ + quad_form)
    catch e
        if isa(e, PosDefException)
            @warn "Covariance matrix is not positive definite. Returning -Inf."
            return -Inf
        else
            rethrow(e)
        end
    end
end

"""
    calculate_posterior(Φ::Matrix{Float64}, data::Vector{Float64},
                        prior_precision::Matrix{Float64}, prior_mean::Vector{Float64},
                        sigma2::Float64) -> GaussianPosterior

データと事前分布から事後分布 N(β | m, Λ) を計算する。
"""
function calculate_posterior(Φ::Matrix{Float64}, data::Vector{Float64},
                             prior_precision::Matrix{Float64}, prior_mean::Vector{Float64},
                             sigma2::Float64)::GaussianPosterior
    if sigma2 <= 0
        error("sigma2 must be positive")
    end
    post_precision = prior_precision + Φ' * Φ / sigma2
    try
        chol_post_precision = cholesky(Hermitian(post_precision))
        rhs = prior_precision * prior_mean + Φ' * data / sigma2
        post_mean = chol_post_precision \ rhs
        post_cov = chol_post_precision \ I
        return GaussianPosterior(post_mean, post_cov, post_precision, chol_post_precision)
    catch e
         if isa(e, PosDefException)
            @error "Posterior precision matrix is not positive definite."
            throw(e)
        else
            rethrow(e)
        end
    end
end

"""
    expected_log_likelihood_term(Φ::Matrix{Float64}, data::Vector{Float64},
                                 posterior::GaussianPosterior, sigma2::Float64) -> Float64

E[log p(data | β, σ²)] の期待値を計算 (Mステップの σ² 更新用)。
"""
function expected_log_likelihood_term(Φ::Matrix{Float64}, data::Vector{Float64},
                                      posterior::GaussianPosterior, sigma2::Float64)::Float64
    m = posterior.mean
    Λ = posterior.cov
    n = size(Φ, 1)
    residual = data - Φ * m
    expected_sq_error = dot(residual, residual) + tr(Φ' * Φ * Λ)
    return -0.5 * n * log(2π * sigma2) - 0.5 * expected_sq_error / sigma2
end

# ---------------------------------------------------------------------------
# 2. モデル選択 (ステップ0)
# ---------------------------------------------------------------------------

"""
    log_marginal_likelihood(Φ::Matrix{Float64}, data::Vector{Float64},
                            μ::Vector{Float64}, Σ::Matrix{Float64}, σ²::Float64) -> Float64

対数周辺尤度 log p(data | μ, Σ, σ²) = log N(data | Φμ, σ²I + ΦΣΦᵀ) を計算。
"""
function log_marginal_likelihood(Φ::Matrix{Float64}, data::Vector{Float64},
                                 μ::Vector{Float64}, Σ::Matrix{Float64}, σ²::Float64)::Float64
    n = size(Φ, 1)
    if n == 0 return 0.0 end
    if σ² <= 0 return -Inf end

    mean_pred = Φ * μ
    cov_pred = σ² * I(n) + Φ * Σ * Φ'

    try
        C = cholesky(Hermitian(cov_pred))
        logdet_cov = logdet(C)
        z = C.L \ (data - mean_pred)
        quad_form = dot(z, z)
        return -0.5 * (n * log(2π) + logdet_cov + quad_form)
    catch e
        if isa(e, PosDefException)
            return -Inf
        else
            rethrow(e)
        end
    end
end

"""
    select_model_parameters(trajs::Vector{Trajectory}, axis::Symbol,
                            d_candidates::Vector{Int}, alpha_candidates::Vector{Float64};
                            lambda0=1e-6, lambda_other=1.0, reg_type=:L2_sq,
                            em_tol=1e-4, em_max_iter=100) -> Tuple{Int, Float64}

指定された軸のデータを用いて、対数周辺尤度に基づき最適な次数 d と正則化 α を選択する。
"""
function select_model_parameters(trajs::Vector{Trajectory}, axis::Symbol,
                                 d_candidates::Vector{Int}, alpha_candidates::Vector{Float64};
                                 lambda0=1e-6, lambda_other=1.0, reg_type=:L2_sq,
                                 em_tol=1e-4, em_max_iter=50)::Tuple{Int, Float64, Vector{Tuple{Int, Float64, Float64}}}
    best_d = -1
    best_alpha = -1.0
    max_lml = -Inf
    total_combinations = length(d_candidates) * length(alpha_candidates)
    p = Progress(total_combinations, 1, "Selecting model for axis $(axis)... ")

    # 軸データの抽出
    get_data(traj, ax) = ax == :x ? traj.x : (ax == :y ? traj.y : traj.z)

    results = Vector{Tuple{Int, Float64, Float64}}(undef, total_combinations)
    idx = 1
    for d in d_candidates
        D = regularization_matrix(d, type=reg_type)
        for alpha in alpha_candidates
            # 1. 事前分布 Σ を固定
            prior_prec = precision_matrix(d, alpha, D, lambda0=lambda0, lambda_other=lambda_other)
            prior_cov = safe_inv(prior_prec)

            # 2. この (d, α) で μ と σ² を一時的にEM推定
            temp_mu, temp_sigma2, _post_tmp, _hist_tmp = run_em_step12(trajs, axis, d, prior_prec, prior_cov,
                                                  tol=em_tol, max_iter=em_max_iter, verbose=false)

            # 3. 対数周辺尤度を計算（各軌跡は t0 揃え）
            current_lml = 0.0
            for traj in trajs
                if traj.n_points > d
                    Φ = design_matrix_zeroed(traj, d)           # ← 修正：相対時刻
                    data = get_data(traj, axis)
                    current_lml += log_marginal_likelihood(Φ, data, temp_mu, prior_cov, temp_sigma2)
                end
            end

            if current_lml > max_lml
                max_lml = current_lml
                best_d = d
                best_alpha = alpha
            end
            results[idx] = (d, alpha, current_lml)
            idx += 1
            ProgressMeter.next!(p; showvalues = [(:d, d), (:alpha, alpha), (:LML, current_lml)])
        end
    end
    ProgressMeter.finish!(p)

    if best_d == -1
        error("Failed to select model parameters. LML remained -Inf.")
    end
    println("Selected for axis $(axis): d = $(best_d), alpha = $(best_alpha) (Max LML = $(max_lml))")
    return best_d, best_alpha, results
end

# ---------------------------------------------------------------------------
# 3. EMアルゴリズム (ステップ1, 2) - X軸とY軸
# ---------------------------------------------------------------------------

"""
    run_em_step12(trajs::Vector{Trajectory}, axis::Symbol, d::Int,
                  prior_precision::Matrix{Float64}, prior_cov::Matrix{Float64};
                  mu_init=nothing, sigma2_init=nothing,
                  tol=1e-6, max_iter=100, verbose=true)
                  -> Tuple{Vector{Float64}, Float64, Vector{GaussianPosterior}, EMHistory}

X軸またはY軸のパラメータ (μ, σ²) と各軌跡の係数事後分布をEMアルゴリズムで推定する。
"""
function run_em_step12(trajs::Vector{Trajectory}, axis::Symbol, d::Int,
                       prior_precision::Matrix{Float64}, prior_cov::Matrix{Float64};
                       mu_init=nothing, sigma2_init=nothing,
                       tol=1e-6, max_iter=100, verbose=true)::Tuple{Vector{Float64}, Float64, Vector{GaussianPosterior}, EMHistory}

    N = length(trajs)
    dim_beta = d + 1
    get_data(traj, ax) = ax == :x ? traj.x : (ax == :y ? traj.y : traj.z)

    # --- 初期化 ---
    mu = mu_init !== nothing ? copy(mu_init) : zeros(Float64, dim_beta)
    if axis == :y && d >= 2 && mu_init === nothing
        mu[3] = -0.01
    end

    # sigma2 の初期値: 全データの単純な線形回帰の残差分散など（t0揃え）
    if sigma2_init !== nothing
        sigma2 = sigma2_init
    else
        all_residuals_sq = Float64[]
        for traj in trajs
            if traj.n_points > dim_beta
                Φ = design_matrix_zeroed(traj, d)              # ← 修正
                data = get_data(traj, axis)
                try
                    beta_ls = Φ \ data
                    append!(all_residuals_sq, (data - Φ * beta_ls).^2)
                catch e
                    @warn "LS failed for traj $(traj.id) during sigma2 init. Skipping."
                end
            end
        end
        sigma2 = isempty(all_residuals_sq) ? 1.0 : max(1e-6, mean(all_residuals_sq))
    end

    if verbose
        println("Initial params for axis $(axis): mu ≈ $(round.(mu, digits=3)), sigma2 ≈ $(round(sigma2, digits=6))")
    end

    # 履歴用
    history = EMHistory([], Dict(:mu => [], :sigma2 => []), [])
    push!(history.params[:mu], copy(mu))
    push!(history.params[:sigma2], sigma2)

    # 事後分布格納用
    posteriors = Vector{GaussianPosterior}(undef, N)

    # --- EMループ ---
    iter = 0
    diff = Inf
    prog = Progress(max_iter, (verbose ? 1 : Inf), "Running EM for axis $(axis)... ")

    while iter < max_iter && diff > tol
        iter += 1
        mu_old = copy(mu)
        sigma2_old = sigma2

        sum_E_beta = zeros(Float64, dim_beta)
        sum_E_loglik_term = 0.0
        total_points = 0

        # --- Eステップ ---
        for i in 1:N
            traj = trajs[i]
            if traj.n_points == 0 continue end
            Φ = design_matrix_zeroed(traj, d)                  # ← 修正
            data = get_data(traj, axis)

            try
                post = calculate_posterior(Φ, data, prior_precision, mu_old, sigma2_old)
                posteriors[i] = post
                sum_E_beta += post.mean
                sum_E_loglik_term += expected_log_likelihood_term(Φ, data, post, sigma2_old)
                total_points += traj.n_points
            catch e
                 @warn "Posterior calculation failed for traj $(traj.id) in EM step $(iter). Skipping. Error: $e"
                 continue
            end
        end

        if total_points == 0
            error("No data points available for EM update.")
        end

        # --- Mステップ ---
        mu = sum_E_beta / N

        sum_expected_sq_error = -2.0 * sum_E_loglik_term - total_points * log(2π * sigma2_old)
        sigma2 = max(1e-9, sum_expected_sq_error / total_points)

        # --- 収束判定 ---
        diff_mu = norm(mu - mu_old) / (norm(mu_old) + 1e-8)
        diff_sigma2 = abs(sigma2 - sigma2_old) / (sigma2_old + 1e-8)
        diff = max(diff_mu, diff_sigma2)

        push!(history.params[:mu], copy(mu))
        push!(history.params[:sigma2], sigma2)
        push!(history.diff, diff)

        ProgressMeter.next!(prog; showvalues = [(:iter, iter), (:diff, diff), (:sigma2, sigma2)])
    end
    ProgressMeter.finish!(prog)

    if iter == max_iter && diff > tol
        @warn "EM algorithm for axis $(axis) did not converge within $(max_iter) iterations. Final diff: $(diff)"
    elseif verbose
        println("EM algorithm for axis $(axis) converged after $(iter) iterations.")
        println("Final params: mu ≈ $(round.(mu, digits=3)), sigma2 ≈ $(round(sigma2, digits=6))")
    end

    # 最終的な事後分布を計算（最後のパラメータで）
    for i in 1:N
         traj = trajs[i]
         if traj.n_points == 0 continue end
         Φ = design_matrix_zeroed(traj, d)                      # ← 修正
         data = get_data(traj, axis)
         try
             posteriors[i] = calculate_posterior(Φ, data, prior_precision, mu, sigma2)
         catch e
             @warn "Final posterior calculation failed for traj $(traj.id). Error: $e"
         end
    end

    return mu, sigma2, posteriors, history
end

# ---------------------------------------------------------------------------
# 4. EMアルゴリズム (ステップ3) - Z軸 (混合ガウスノイズ)
# ---------------------------------------------------------------------------

"""
    run_em_step3_z(trajs::Vector{Trajectory}, d::Int,
                   prior_mean_xz::Vector{Float64}, prior_precision_xz::Matrix{Float64},
                   pi_init=0.05, sigma2z0_init=nothing, sigma2z1_init=nothing,
                   tol=1e-6, max_iter=100, verbose=true)
                   -> Tuple{Float64, Float64, Float64, Vector{Vector{Float64}}, Vector{Vector{Float64}}, EMHistory}

Z軸のパラメータ (π, σ²₀, σ²₁) と各軌跡の係数βzᵢ (MAP推定) をEMアルゴリズムで推定する。
"""
function run_em_step3_z(trajs::Vector{Trajectory}, d::Int,
                        prior_mean_xz::Vector{Float64}, prior_precision_xz::Matrix{Float64},
                        sigma2x_est::Float64; # sigma2z0, sigma2z1 の初期値用
                        pi_init=0.05, sigma2z0_init=nothing, sigma2z1_init=nothing,
                        tol=1e-6, max_iter=100, verbose=true)::Tuple{Float64, Float64, Float64, Vector{Vector{Float64}}, Vector{Vector{Float64}}, EMHistory}

    N = length(trajs)
    dim_beta = d + 1

    # --- 初期化 ---
    pi_z = pi_init
    sigma2z0 = sigma2z0_init !== nothing ? sigma2z0_init : max(1e-9, sigma2x_est * 1.5)
    sigma2z1 = sigma2z1_init !== nothing ? sigma2z1_init : max(1e-9, sigma2z0 * 50.0)

    beta_z = [copy(prior_mean_xz) for _ in 1:N]

    if verbose
        println("Initial params for Z-axis: pi=$(pi_z), sigma2z0=$(sigma2z0), sigma2z1=$(sigma2z1)")
    end

    history = EMHistory([], Dict(:pi => [], :sigma2z0 => [], :sigma2z1 => [], :beta_z_norm => []), [])
    push!(history.params[:pi], pi_z)
    push!(history.params[:sigma2z0], sigma2z0)
    push!(history.params[:sigma2z1], sigma2z1)
    push!(history.params[:beta_z_norm], mean(norm.(beta_z)))

    responsibilities = [zeros(Float64, traj.n_points) for traj in trajs]
    weights_W = [zeros(Float64, traj.n_points) for traj in trajs] # 未使用だが残す

    # --- EMループ ---
    iter = 0
    diff = Inf
    prog = Progress(max_iter, (verbose ? 1 : Inf), "Running EM for Z-axis... ")

    while iter < max_iter && diff > tol
        iter += 1
        pi_old = pi_z
        sigma2z0_old = sigma2z0
        sigma2z1_old = sigma2z1
        beta_z_old = deepcopy(beta_z)

        total_points = 0
        sum_resp = 0.0
        sum_1_minus_resp = 0.0
        sum_resp_sq_err1 = 0.0
        sum_1_minus_resp_sq_err0 = 0.0

        # --- Eステップ ---
        for i in 1:N
            traj = trajs[i]
            if traj.n_points == 0 continue end
            Φ = design_matrix_zeroed(traj, d)                  # ← 修正
            z_data = traj.z
            beta_zi = beta_z_old[i]
            z_pred = Φ * beta_zi

            log_pdf0 = log_gaussian_pdf.(z_data, z_pred, sigma2z0_old)
            log_pdf1 = log_gaussian_pdf.(z_data, z_pred, sigma2z1_old)

            log_p0 = log(1.0 - pi_old + 1e-12) .+ log_pdf0
            log_p1 = log(pi_old + 1e-12) .+ log_pdf1

            log_probs_matrix = hcat(log_p0, log_p1)
            log_denom = [logsumexp(row) for row in eachrow(log_probs_matrix)]  # 行ごと

            resp_ij = exp.(log_p1 .- log_denom)
            resp_ij[isnan.(resp_ij)] .= 0.0
            resp_ij = clamp.(resp_ij, 0.0, 1.0)

            responsibilities[i] = resp_ij
            sum_resp += sum(resp_ij)
            total_points += traj.n_points
        end

        if total_points == 0
            error("No data points available for Z-axis EM update.")
        end
        sum_1_minus_resp = total_points - sum_resp

        # --- Mステップ ---
        # π の更新
        pi_z = sum_resp / total_points

        # βzᵢ の更新 (MAP)
        for i in 1:N
            traj = trajs[i]
            if traj.n_points == 0 continue end
            Φ = design_matrix_zeroed(traj, d)                  # ← 修正
            z_data = traj.z
            resp_i = responsibilities[i]

            W_ij = (1.0 .- resp_i) / sigma2z0_old + resp_i / sigma2z1_old
            W_diag = Diagonal(W_ij)

            try
                A = Φ' * W_diag * Φ + prior_precision_xz
                b = Φ' * W_diag * z_data + prior_precision_xz * prior_mean_xz
                C_A = cholesky(Hermitian(A))
                beta_z[i] = C_A \ b
            catch e
                @warn "Failed to update beta_z for traj $(traj.id). Keeping previous value. Error: $e"
                beta_z[i] = beta_z_old[i]
            end
        end

        # σ²₀, σ²₁ の更新
        for i in 1:N
            traj = trajs[i]
            if traj.n_points == 0 continue end
            Φ = design_matrix_zeroed(traj, d)                  # ← 修正
            z_data = traj.z
            resp_i = responsibilities[i]
            z_pred_new = Φ * beta_z[i]
            sq_error = (z_data - z_pred_new).^2

            sum_1_minus_resp_sq_err0 += sum((1.0 .- resp_i) .* sq_error)
            sum_resp_sq_err1 += sum(resp_i .* sq_error)
        end

        sigma2z0 = max(1e-9, sum_1_minus_resp_sq_err0 / max(1e-9, sum_1_minus_resp))
        sigma2z1 = max(1e-9, sum_resp_sq_err1 / max(1e-9, sum_resp))

        if sigma2z0 > sigma2z1
            sigma2z0, sigma2z1 = sigma2z1, sigma2z0
            @warn "Swapped sigma2z0 and sigma2z1 at iteration $(iter)"
        end

        # --- 収束判定 ---
        diff_pi = abs(pi_z - pi_old) / (pi_old + 1e-8)
        diff_s0 = abs(sigma2z0 - sigma2z0_old) / (sigma2z0_old + 1e-8)
        diff_s1 = abs(sigma2z1 - sigma2z1_old) / (sigma2z1_old + 1e-8)
        diff_beta = mean([norm(beta_z[i] - beta_z_old[i]) / (norm(beta_z_old[i]) + 1e-8) for i in 1:N if trajs[i].n_points > 0])
        diff = max(diff_pi, diff_s0, diff_s1, diff_beta)

        push!(history.params[:pi], pi_z)
        push!(history.params[:sigma2z0], sigma2z0)
        push!(history.params[:sigma2z1], sigma2z1)
        push!(history.params[:beta_z_norm], mean(norm.(beta_z)))
        push!(history.diff, diff)

        ProgressMeter.next!(prog; showvalues = [(:iter, iter), (:diff, diff), (:pi, pi_z), (:s0, sigma2z0), (:s1, sigma2z1)])
    end
    ProgressMeter.finish!(prog)

    if iter == max_iter && diff > tol
        @warn "EM algorithm for Z-axis did not converge within $(max_iter) iterations. Final diff: $(diff)"
    elseif verbose
        println("EM algorithm for Z-axis converged after $(iter) iterations.")
        println("Final params: pi=$(pi_z), sigma2z0=$(sigma2z0), sigma2z1=$(sigma2z1)")
    end

    # 最終的な負担率を計算
    for i in 1:N
        traj = trajs[i]
        if traj.n_points == 0 continue end
        Φ = design_matrix_zeroed(traj, d)                      # ← 修正
        z_data = traj.z
        beta_zi = beta_z[i]
        z_pred = Φ * beta_zi
        log_pdf0 = log_gaussian_pdf.(z_data, z_pred, sigma2z0)
        log_pdf1 = log_gaussian_pdf.(z_data, z_pred, sigma2z1)
        log_p0 = log(1.0 - pi_z + 1e-12) .+ log_pdf0
        log_p1 = log(pi_z + 1e-12) .+ log_pdf1

        log_probs_matrix = hcat(log_p0, log_p1)
        log_denom = [logsumexp(row_vector) for row_vector in eachrow(log_probs_matrix)]

        resp_ij = exp.(log_p1 .- log_denom)
        resp_ij[isnan.(resp_ij)] .= 0.0
        responsibilities[i] = clamp.(resp_ij, 0.0, 1.0)
    end

    return pi_z, sigma2z0, sigma2z1, beta_z, responsibilities, history
end


# ---------------------------------------------------------------------------
# 5. 結果の可視化と軌跡推定
# ---------------------------------------------------------------------------

"""
    plot_em_history(history::EMHistory; title_suffix="")

EMアルゴリズムの収束履歴をプロットする。
"""
function plot_em_history(history::EMHistory; title_suffix="")
    plots_list = []

    if !isempty(history.log_likelihood)
        p_ll = plot(history.log_likelihood, xlabel="Iteration", ylabel="Log Likelihood", title="Log Likelihood$(title_suffix)", legend=false)
        push!(plots_list, p_ll)
    end

    p_diff = plot(history.diff, xlabel="Iteration", ylabel="Max Rel. Change", title="Convergence$(title_suffix)", yscale=:log10, legend=false)
    push!(plots_list, p_diff)

    for (param_name, values) in history.params
        if isempty(values) continue end
        if eltype(values) <: AbstractVector
            y_data = [norm(v) for v in values]
            ylabel_str = "Norm($(param_name))"
        else
            y_data = values
            ylabel_str = string(param_name)
        end
        p = plot(y_data, xlabel="Iteration", ylabel=ylabel_str, title="$(param_name) History$(title_suffix)", legend=false)
        push!(plots_list, p)
    end

    n_plots = length(plots_list)
    layout_cols = ceil(Int, sqrt(n_plots))
    layout_rows = ceil(Int, n_plots / layout_cols)
    plot(plots_list..., layout=(layout_rows, layout_cols), size=(layout_cols*300, layout_rows*250), dpi=600)
end


"""
    estimate_trajectory(beta_coeffs::Vector{Float64}, times::Vector{Float64}) -> Vector{Float64}

推定された係数ベクトルと時間ベクトルから、近似軌跡上の点を計算する。
"""
function estimate_trajectory(beta_coeffs::Vector{Float64}, times::Vector{Float64})::Vector{Float64}
    d = length(beta_coeffs) - 1
    Φ = design_matrix(times, d)
    return Φ * beta_coeffs
end

"""
    get_estimated_trajectory_3d(result::EstimationResult, traj_id::Int, times::Vector{Float64})
        -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}

指定された軌跡IDと時間ベクトルに対する3次元の近似軌跡 (x, y, z) を返す。
評価時も学習時と同様に、その軌跡の先頭を t=0 に揃える。
"""
function get_estimated_trajectory_3d(result::EstimationResult, traj_id::Int, times::Vector{Float64})
    beta_x = result.beta_x_posteriors[traj_id].mean
    beta_y = result.beta_y_posteriors[traj_id].mean
    beta_z = result.beta_z_map[traj_id]

    # 学習時と同じ原点（当該軌跡の先頭時刻）で評価
    t0 = result.trajs[traj_id].times[1]
    times_rel = times .- t0

    x_hat = estimate_trajectory(beta_x, times_rel)
    y_hat = estimate_trajectory(beta_y, times_rel)
    z_hat = estimate_trajectory(beta_z, times_rel)

    return x_hat, y_hat, z_hat
end

"""
    differentiate_coeffs(beta_coeffs::Vector{Float64}) -> Vector{Float64}

p(t)=β₀+β₁t+…+β_d t^d の係数から、p'(t) の係数
[β₁, 2β₂, …, dβ_d] を返す。
"""
function differentiate_coeffs(beta_coeffs::Vector{Float64})::Vector{Float64}
    d = length(beta_coeffs) - 1
    if d <= 0
        return Float64[]  # 定数多項式の微分は0
    end
    return [i * beta_coeffs[i+1] for i in 1:d]
end

"""
    estimate_velocity(beta_coeffs::Vector{Float64}, times::Vector{Float64}) -> Vector{Float64}

速度 v(t)=dp/dt を、既存の design_matrix を再利用して計算する。
"""
function estimate_velocity(beta_coeffs::Vector{Float64}, times::Vector{Float64})::Vector{Float64}
    d = length(beta_coeffs) - 1
    if d == 0
        return zeros(Float64, length(times))
    end
    beta_dot = differentiate_coeffs(beta_coeffs)  # 長さ d
    Φdot = design_matrix(times, d - 1)           # 次数を1つ下げた設計行列
    return Φdot * beta_dot
end

"""
    get_estimated_velocity_3d(result::EstimationResult, traj_id::Int, times::Vector{Float64})
        -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}

指定された軌跡IDと時間ベクトルに対する (vx, vy, vz) を返す。
評価時も学習時と同様に、その軌跡の先頭を t=0 に揃える。
"""
function get_estimated_velocity_3d(result::EstimationResult, traj_id::Int, times::Vector{Float64})
    beta_x = result.beta_x_posteriors[traj_id].mean
    beta_y = result.beta_y_posteriors[traj_id].mean
    beta_z = result.beta_z_map[traj_id]

    # 学習時と同じ原点（当該軌跡の先頭時刻）で評価
    t0 = result.trajs[traj_id].times[1]
    times_rel = times .- t0

    vx_hat = estimate_velocity(beta_x, times_rel)
    vy_hat = estimate_velocity(beta_y, times_rel)
    vz_hat = estimate_velocity(beta_z, times_rel)

    return vx_hat, vy_hat, vz_hat
end


# ---------------------------------------------------------------------------
# 6. メイン処理関数
# ---------------------------------------------------------------------------

"""
    load_trajectory_data(data_dir::String, file_pattern::String="*.csv") -> Vector{Trajectory}

指定されたディレクトリから軌跡データを読み込む。
CSVファイル形式はサンプルコードに合わせる (time, x, y, z が 1, 2, 3, 4 列目)。
"""
function load_trajectory_data(data_dir::String; file_pattern::String="*.csv")::Vector{Trajectory}
    trajs = Vector{Trajectory}()
    files = glob(file_pattern, data_dir)
    if isempty(files)
        @warn "No trajectory files found in $(data_dir) with pattern $(file_pattern)"
        return trajs
    end

    for (i, file) in enumerate(files)
        try
            df = CSV.read(file, DataFrame)
            if size(df, 2) < 4
                @warn "Skipping file $(basename(file)): Not enough columns (needs at least 4 for time, x, y, z)."
                continue
            end
            times = Float64.(df[!, 1]) # 時間フレーム
            x = Float64.(df[!, 2])
            y = Float64.(df[!, 3])
            z = Float64.(df[!, 4])
            diam = Float64.((df[!,8]))
            uuids = UUID.(df[!,9])
            n_points = length(times)
            if n_points > 0
                push!(trajs, Trajectory(basename(file), i, times, x, y, z, diam, uuids, n_points))
            end
        catch e
            @warn "Failed to read or parse file $(basename(file)). Error: $e"
        end
    end
    println("Loaded $(length(trajs)) trajectories from $(data_dir)")
    return trajs
end

"""
    run_full_analysis(data_dir::String; ...) -> EstimationResult

データ読み込みからモデル選択、EM推定までの一連の処理を実行する。
"""
function run_full_analysis(data_dir::String;
                           d_candidates::Vector{Int}=[2, 3, 4, 5],
                           alpha_candidates::Vector{Float64}=[0.0, 0.01, 0.1, 1.0, 10.0],
                           lambda0::Float64=1e-6,
                           lambda_other::Float64=1.0,
                           reg_type::Symbol=:L2_sq,
                           em_tol::Float64=1e-5,
                           em_max_iter::Int=100,
                           plot_convergence::Bool=true,
                           verbose::Bool=true
                           )::Union{EstimationResult, Nothing}

    # 1. データ読み込み
    trajs = load_trajectory_data(data_dir)
    if isempty(trajs)
        @error "No valid trajectories loaded. Aborting analysis."
        return nothing
    end

    # 2. モデル選択 (ステップ0)
    println("\n--- Step 0: Model Selection ---")
    best_d_x, best_alpha_xz, resultdata_x = select_model_parameters(trajs, :x, d_candidates, alpha_candidates,
                                                     lambda0=lambda0, lambda_other=lambda_other, reg_type=reg_type,
                                                     em_tol=em_tol, em_max_iter=100000)
    best_d_y, best_alpha_y, resultdata_y = select_model_parameters(trajs, :y, d_candidates, alpha_candidates,
                                                    lambda0=lambda0, lambda_other=lambda_other, reg_type=reg_type,
                                                    em_tol=em_tol, em_max_iter=100000)

    # 次数は共通にするのが一般的 (ここではX軸の結果を採用)
    final_d = best_d_x
    final_alpha_xz = best_alpha_xz
    final_alpha_y = best_alpha_y
    println("\nUsing final parameters: d = $(final_d), alpha_xz = $(final_alpha_xz), alpha_y = $(final_alpha_y)")

    # 事前分布の準備
    D = regularization_matrix(final_d, type=reg_type)
    prior_precision_xz = precision_matrix(final_d, final_alpha_xz, D, lambda0=lambda0, lambda_other=lambda_other)
    prior_cov_xz = safe_inv(prior_precision_xz)
    prior_precision_y = precision_matrix(final_d, final_alpha_y, D, lambda0=lambda0, lambda_other=lambda_other)
    prior_cov_y = safe_inv(prior_precision_y)

    # 3. X軸解析 (ステップ1)
    println("\n--- Step 1: X-axis Analysis ---")
    mu_xz, sigma2x, beta_x_posteriors, history_x = run_em_step12(
        trajs, :x, final_d, prior_precision_xz, prior_cov_xz,
        tol=em_tol, max_iter=em_max_iter, verbose=verbose
    )
    if plot_convergence
        display(plot_em_history(history_x, title_suffix=" (X-axis)"))
        savefig("em_history_x.png")
    end

    # 4. Y軸解析 (ステップ2)
    println("\n--- Step 2: Y-axis Analysis ---")
    mu_y, sigma2y, beta_y_posteriors, history_y = run_em_step12(
        trajs, :y, final_d, prior_precision_y, prior_cov_y,
        tol=em_tol, max_iter=em_max_iter, verbose=verbose
    )
    if plot_convergence
        display(plot_em_history(history_y, title_suffix=" (Y-axis)"))
        savefig("em_history_y.png")
    end

    # 5. Z軸解析 (ステップ3)
    println("\n--- Step 3: Z-axis Analysis ---")
    pi_z, sigma2z0, sigma2z1, beta_z_map, responsibilities, history_z = run_em_step3_z(
        trajs, final_d, mu_xz, prior_precision_xz, sigma2x,
        tol=em_tol, max_iter=em_max_iter, verbose=verbose
    )
     if plot_convergence
        display(plot_em_history(history_z, title_suffix=" (Z-axis)"))
        savefig("em_history_z.png")
    end

    # 6. 結果をまとめる
    estimation_result = EstimationResult(
        final_d, final_alpha_xz, final_alpha_y,
        mu_xz, prior_cov_xz, sigma2x, beta_x_posteriors,
        mu_y, prior_cov_y, sigma2y, beta_y_posteriors,
        pi_z, sigma2z0, sigma2z1, beta_z_map, responsibilities,
        Dict(:x => history_x, :y => history_y, :z => history_z),
        resultdata_x, resultdata_y, trajs
    )

    println("\n--- Analysis Finished ---")
    @printf "Final X-axis: sigma2x=%.4e\n" sigma2x
    @printf "Final Y-axis: sigma2y=%.4e\n" sigma2y
    @printf "Final Z-axis: pi=%.3f, sigma2z0=%.4e, sigma2z1=%.4e\n" pi_z sigma2z0 sigma2z1

    return estimation_result
end


# ---------------------------------------------------------------------------
# 7. 実行例・可視化
# ---------------------------------------------------------------------------
function acquire_result(date, scenenum)
# function acquire_result(date, scenenum; phase)
    # --- パラメータ設定 ---
    # date = "20250121"
    # scenenum = 10
    data_dir = joinpath(date, "07_trajectories", "C000H001S"*lpad(scenenum, 4, '0'))
    # data_dir = joinpath(date, "07_trajectories", "C000H001S"*lpad(scenenum, 4, '0'), lpad(phase, 2, '0'))

    # --- 解析実行 ---
    result = run_full_analysis(data_dir,
                               d_candidates=[1, 2],
                            #    alpha_candidates=[1.0],
                               alpha_candidates=[0.1, 1.0, 1.5],
                               em_max_iter=1000000,
                               plot_convergence=true)

    return result
end

function estimate_plot(result, traj_id)
    original_traj = result.trajs[traj_id]
    times_to_plot = original_traj.times # 表示軸は元の時刻（相対化しない）

    # 推定された軌跡を取得（内部で相対時刻に変換して評価）
    x_hat, y_hat, z_hat = get_estimated_trajectory_3d(result, traj_id, times_to_plot)

    println("\nPlotting trajectory $(traj_id)...")
    p1 = plot(original_traj.times, original_traj.x, label="Obs X", ls=:dash, lc=:gray, marker=:circle, mc=:gray, ms=3, alpha=0.7, legend=false, ylabel="X", framestyle=:box, bottommargin=-5Plots.mm)
    plot!(times_to_plot, x_hat, label="Est X", lw=1.5, xformatter = _ -> "", lc=palette(:heat)[2])
    p2 = plot(original_traj.times, original_traj.y, label="Obs Y", ls=:dash, marker=:circle, mc=:gray, ms=3, alpha=0.7, legend=false, ylabel="Y", framestyle=:box)
    plot!(times_to_plot, y_hat, label="Est Y", lw=1.5, xformatter = _ -> "", lc=palette(:heat)[2], bottommargin=-5Plots.mm)
    p3 = plot(original_traj.times, original_traj.z, label="Obs Z", ls=:dash, lc=:gray, marker=:circle, ms=0, alpha=0.7, ylabel="Z", xlabel="Time", framestyle=:box)
    scatter!(original_traj.times, original_traj.z, marker_z=result.responsibilities[traj_id], color=:coolwarm, clims=(0,1), ms=3, label="Obs Z (Resp)", legend=:right, cb=false, lc=:gray)
    plot!(times_to_plot, z_hat, label="Est Z", lw=2, legend=false, lc=palette(:heat)[2], bottommargin=5Plots.mm)
    p_combined = plot(p1, p2, p3, layout=(3,1), dpi=600, size=(800, 400), leftmargin=3Plots.mm)
    display(p_combined)
    return p_combined
end

function pathplot(result, traj_id; cam=(30,30))
    original_traj = result.trajs[traj_id]
    x_hat, y_hat, z_hat = get_estimated_trajectory_3d(result, traj_id, original_traj.times)
    path3d(original_traj.x, original_traj.z, original_traj.y, color=:black, lw=1, label="Observed", legend=false, dpi=600, size=(400,400))
    p3d = scatter!(original_traj.x, original_traj.z, original_traj.y,
    marker_z=result.responsibilities[traj_id], color=:coolwarm, clims=(0,1),
    ms=3, markerstrokewidth=0, label="Observed (color=Resp)", xlabel="X", ylabel="Z", zlabel="Y", camera=cam, cb=false)
    path3d!(x_hat, z_hat, y_hat, color=:green, lw=2, label="Estimated")
    display(p3d)
    return p3d
end

function all_estim_path_plot(result)
    p = plot()
    for traj_id in 1:length(result.trajs)
        original_traj = result.trajs[traj_id]
        x_hat, y_hat, z_hat = get_estimated_trajectory_3d(result, traj_id, original_traj.times)
        color = palette(:tab10)[traj_id % 10 + 1]
        path3d!(x_hat, z_hat, y_hat, color=color, lw=2, legend=false, dpi=600)
    end
    display(p)
end

function boxpathplot(result, starttime, endtime; cam=(30,30))
    p= plot()
    p1 = plot()
    # cam = (0,0)
    color(x) = palette(:tab10)[Int(round(x*pi*10000)) % 10 + 1]
    for traj_id in 1:length(result.trajs)
        original_traj = result.trajs[traj_id]
        # 指定された時間範囲でフィルタリング（表示軸は元の時刻）
        mask = (original_traj.times .>= starttime) .& (original_traj.times .<= endtime)
        if !any(mask) continue end

        x_hat, y_hat, z_hat = get_estimated_trajectory_3d(result, traj_id, original_traj.times[mask])
        path3d!(p1, x_hat .*10e-3, z_hat .*100e-3, y_hat .*10e-3, zflip=true, color=color(traj_id), lw=2, label="Trajectory $(traj_id)", legend=false, dpi=600, ylims=(0, 100), xlims=(0,10.24), zlims=(0,10.24), cam=cam, yflip=true, xlabel="X [mm]", ylabel="Z [mm]", zlabel="Y [mm]", xticks=0:5:10, zticks=0:5:10)
        path3d!(p, original_traj.x[mask] .*10e-3, original_traj.z[mask] .*100e-3, original_traj.y[mask] .*10e-3, zflip=true, color=color(traj_id), lw=2, label="Trajectory $(traj_id)", legend=false, dpi=600, ylims=(0, 100), xlims=(0,10.24), zlims=(0,10.24), cam=cam, yflip=true, xlabel="X [mm]", ylabel="Z [mm]", zlabel="Y [mm]", xticks=0:5:10, zticks=0:5:10)
    end
    return p, p1
end

function boxpointplot(result, time)
    p1 = plot()
    p2 = plot()
    cam = (30,30)
    color(x) = palette(:tab10)[Int(round(x*pi*10000)) % 10 + 1]
    for traj_id in 1:length(result.trajs)
        original_traj = result.trajs[traj_id]
        # 指定された時間に最も近い点を見つける
        idx = findfirst(x -> isapprox(x, time; atol=1e-6), original_traj.times)
        if isnothing(idx) continue end

        x_hat, y_hat, z_hat = get_estimated_trajectory_3d(result, traj_id, [original_traj.times[idx]])
        scatter!(p1, x_hat .*10e-3, z_hat .*100e-3, y_hat .*10e-3, zflip=true, color=color(traj_id), ms=2, markerstrokewidth=0, label="Trajectory $(traj_id)", legend=false, dpi=600, ylims=(0, 100), xlims=(0,10.24), zlims=(0,10.24), cam=cam, yflip=true, xlabel="X [mm]", ylabel="Z [mm]", zlabel="Y [mm]", xticks=0:5:10, zticks=0:5:10)
        scatter!(p2, [original_traj.x[idx]] .*10e-3, [original_traj.z[idx]] .*100e-3, [original_traj.y[idx]] .*10e-3, zflip=true,  color=color(traj_id), ms=2, markerstrokewidth=0, label="Trajectory $(traj_id)", legend=false, dpi=600, ylims=(0, 100), xlims=(0,10.24), zlims=(0,10.24), cam=cam, yflip=true, xlabel="X [mm]", ylabel="Z [mm]", zlabel="Y [mm]", xticks=0:5:10, zticks=0:5:10)
    end

    return p1, p2
end

function animpointplot(result, starttime, endtime)
    anim1 = Animation()
    anim2 = Animation()
    for t in starttime:endtime
        p1, p2 = boxpointplot(result, t)
        frame(anim1, p1)
        frame(anim2, p2)
    end
    gif(anim1, "anim_estimated.gif", fps=10)
    gif(anim2, "anim_observed.gif", fps=10)
end

function saveresultsascsv(result, savedir="20250121/09_smootheddata")
    for traj_id in 1:length(result.trajs)
        original_traj = result.trajs[traj_id]
        x_hat, y_hat, z_hat = get_estimated_trajectory_3d(result, traj_id, original_traj.times)
        vx, vy, vz = get_estimated_velocity_3d(result, traj_id, original_traj.times)

        # X,YはSavitsky−Golayで上書き
        ###
        if length(x_hat) >= 10
            x_hat = savitzky_golay(original_traj.x, 9, 3).y
            y_hat = savitzky_golay(original_traj.y, 9, 3).y
            vx = savitzky_golay(original_traj.x, 9, 3, deriv=1).y
            vy = savitzky_golay(original_traj.y, 9, 3, deriv=1).y
        end
        ###

        diam = original_traj.diameter
        uuids = original_traj.uuids
        frames = original_traj.times
        responsibility = result.responsibilities[traj_id]
        df = DataFrame([frames, x_hat, y_hat, z_hat, vx, vy, vz, diam, uuids, responsibility], [:frame, :x, :y, :z, :vx, :vy, :vz, :diameter, :uuid, :responsibility])
        savepath = joinpath(savedir, "smoothed_"*original_traj.filename)
        CSV.write(savepath, df)
    end
    return nothing
end

p(x) = plot(pathplot(result,x), estimate_plot(result,x), size=(1200,400), layout=Plots.grid(1,2, widths=(1/3,2/3)))


# 実行
# date = "251226"
# scenenum = 12
date = ARGS[1]
scenenum = parse(Int, ARGS[2])
println("GMM smoothing for date=$(date), scenenum=$(scenenum)")
# phase = parse(Int, ARGS[1])
result = acquire_result(date, scenenum)
# result = acquire_result(date, scenenum, phase)
# datasavedir = joinpath(date, "09_smootheddata", "C000H001S"*lpad(scenenum, 4, '0'), lpad(phase, 2, '0'))
datasavedir = joinpath(date, "09_smootheddata", "C000H001S"*lpad(scenenum, 4, '0'))
mkpath(datasavedir)
saveresultsascsv(result, datasavedir)

# mkpath(joinpath(date, "08_smoothedplot", "C000H001S"*lpad(scenenum, 4, '0')))
# mkpath(joinpath(date, "08_smoothedplot", "C000H001S"*lpad(scenenum, 4, '0'), lpad(phase, 2, '0')))
# for traj_id in 1:min(20, length(result.trajs))
#     if result.trajs[traj_id].n_points < 50
#         continue
#     end
#     savefig(p(traj_id), joinpath(date, "08_smoothedplot", "C000H001S"*lpad(scenenum, 4, '0'), "traj_"*lpad(traj_id, 4, '0')*".png"))
#     # savefig(p(traj_id), joinpath(date, "08_smoothedplot", "C000H001S"*lpad(scenenum, 4, '0'), lpad(phase, 2, '0'), "traj_"*lpad(traj_id, 4, '0')*".png"))
# end

# psum = 0
# traj_id = 1
# while psum < 20
#     if result.trajs[traj_id].n_points < 50
#         traj_id += 1
#         continue
#     end
#     savefig(p(traj_id), joinpath(date, "08_smoothedplot", "C000H001S"*lpad(scenenum, 4, '0'), "traj_"*lpad(traj_id, 4, '0')*".png"))
#     traj_id += 1
#     psum += 1
# end

outdict = Dict()
outdict["variancex"] = result.sigma2x
outdict["variancey"] = result.sigma2y
outdict["variancez0"] = result.sigma2z0
outdict["variancez1"] = result.sigma2z1
outdict["outlierfraction"] = result.pi_z
outdict["polynomialdegree"] = result.d
outdict["regularizationxz"] = result.alpha_xz
outdict["regularizationy"] = result.alpha_y

ourdir = "./"*date*"/06_analysis/"*"/C000H001S"*lpad(scenenum, 4, '0')*"/"
!isdir(ourdir) && mkpath(ourdir)
# YAML.write_file(ourdir*lpad(phase, 2, '0')*"estimated_parameters.yaml", outdict)
# YAML.write_file("./"*date*"/06_analysis/"*"/C000H001S"*lpad(scenenum, 4, '0')*"/"*lpad(phase, 2, '0')*"estimated_parameters.yaml", outdict)
YAML.write_file("./"*date*"/06_analysis/"*"/C000H001S"*lpad(scenenum, 4, '0')*"/estimated_parameters.yaml", outdict)


# animpointplot(result, 0, 100)