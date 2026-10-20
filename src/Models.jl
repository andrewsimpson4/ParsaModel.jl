using StatsBase

const LOG_2PI = log(2.0 * pi)

@inline _as_fvec(x::Vector{Float64}) = x
@inline _as_fvec(x::AbstractVector{<:Real}) = collect(Float64, x)

@inline function _invert_and_det(cov_new::Matrix{Float64})
    p = size(cov_new, 1)
    @inbounds for j in 1:p
        for i in (j+1):p
            avg = 0.5 * (cov_new[i, j] + cov_new[j, i])
            cov_new[i, j] = avg
            cov_new[j, i] = avg
        end
    end
    F = eigen!(Symmetric(cov_new))
    vals = F.values
    vecs = F.vectors
    d = 1.0
    inv_mat = zeros(Float64, p, p)
    @inbounds for k in 1:p
        lam = max(vals[k], 1e-6)
        d *= lam
        inv_lam = 1.0 / lam
        for j in 1:p
            vj_scaled = vecs[j, k] * inv_lam
            for i in 1:p
                inv_mat[i, j] += vecs[i, k] * vj_scaled
            end
        end
    end
    return (inv=inv_mat, det=d)
end

@inline function _quad_form(X::Vector{Float64}, mu::Vector{Float64}, cov_inv::Matrix{Float64})
    p = length(X)
    quad = 0.0
    @inbounds for j in 1:p
        yj = X[j] - mu[j]
        s = 0.0
        for i in 1:p
            s += (X[i] - mu[i]) * cov_inv[i, j]
        end
        quad += s * yj
    end
    return quad
end

@inline function _quad_form(X::AbstractVector{<:Real}, mu::AbstractVector{<:Real}, cov_inv::Matrix{Float64})
    return _quad_form(_as_fvec(X), _as_fvec(mu), cov_inv)
end

##### Normal Model #######

function normal_mean_update(value::Any, index_package::SummationPackage, log_pdf::Function)
    p = length(value)
    mu_new = zeros(Float64, p)
    cov = zeros(Float64, p, p)
    total_pr = 0.0
    for (x_obs, pr_raw, params) in index_package
        pr = Float64(pr_raw)
        pr == 0.0 && continue
        total_pr += pr
        cov_inv = (val(params[:cov]).inv)::Matrix{Float64}
        x = _as_fvec(val(x_obs))::Vector{Float64}
        @inbounds for j in 1:p
            xj = x[j]
            for i in 1:p
                cij = pr * cov_inv[i, j]
                cov[i, j] += cij
                mu_new[i] += cij * xj
            end
        end
    end
    total_pr <= 1e-12 && return copy(value)
    return ldiv!(lu!(cov), mu_new)
end

function normal_covariance_update(value::Any, index_package::SummationPackage, log_pdf::Function)
    p = size(value.inv, 1)
    cov_new = zeros(Float64, p, p)
    sum_taus = 0.0
    for (x_obs, pr_raw, params) in index_package
        pr = Float64(pr_raw)
        pr == 0.0 && continue
        sum_taus += pr
        x = _as_fvec(val(x_obs))::Vector{Float64}
        mu = _as_fvec(val(params[:mu]))::Vector{Float64}
        @inbounds for j in 1:p
            yj = x[j] - mu[j]
            pr_yj = pr * yj
            for i in 1:p
                yi = x[i] - mu[i]
                cov_new[i, j] += pr_yj * yi
            end
        end
    end
    sum_taus <= 1e-12 && return value
    cov_new ./= sum_taus
    return _invert_and_det(cov_new)
end

function normal_pdf(X::Any, params)
    x_vec = _as_fvec(X)::Vector{Float64}
    p = length(x_vec)
    mu = _as_fvec(val(params[:mu]))::Vector{Float64}
    cov = val(params[:cov])
    cov_inv = (cov.inv)::Matrix{Float64}
    cov_det = Float64(cov.det)
    quad = _quad_form(x_vec, mu, cov_inv)
    return -0.5 * (p * LOG_2PI + log(max(cov_det, eps(Float64))) + quad)
end

function normal_pdf_log(X::Any, params)
    N = MvNormal(params[:mu], params[:cov])
    return logpdf(N, X)
end

normal_cov_post(L) = L

normal_input(x, p) = length(x) == p && all(v -> v isa Real, x)

MtvNormal(p) = ParsaDensity(
    normal_pdf,
    normal_pdf_log,
    (x) -> normal_input(x, p),
    :mu => Parameter(zeros(Float64, p), normal_mean_update),
    :cov => Parameter((inv=Matrix{Float64}(I, p, p), det=1.0), div(p * (p + 1), 2), normal_covariance_update),
)

#### Safe Normal Model ####

function normal_mean_update_safe(value::Any, index_package::SummationPackage, log_pdf::Function)
    p = length(value)
    mu_new = zeros(Float64, p)
    cov = zeros(Float64, p, p)
    eff_n = 0.0
    for (x_obs, pr_raw, params) in index_package
        pr = Float64(pr_raw)
        pr == 0.0 && continue
        eff_n += pr
        cov_inv = (val(params[:cov]).inv)::Matrix{Float64}
        x = _as_fvec(val(x_obs))::Vector{Float64}
        @inbounds for j in 1:p
            xj = x[j]
            for i in 1:p
                cij = pr * cov_inv[i, j]
                cov[i, j] += cij
                mu_new[i] += cij * xj
            end
        end
    end
    if eff_n > (p + 1)
        return ldiv!(lu!(cov), mu_new)
    else
        return value
    end
end

function normal_covariance_update_safe(value::Any, index_package::SummationPackage, log_pdf::Function)
    p = size(value.inv, 1)
    cov_new = zeros(Float64, p, p)
    eff_n = 0.0
    for (x_obs, pr_raw, params) in index_package
        pr = Float64(pr_raw)
        pr == 0.0 && continue
        eff_n += pr
        x = _as_fvec(val(x_obs))::Vector{Float64}
        mu = _as_fvec(val(params[:mu]))::Vector{Float64}
        @inbounds for j in 1:p
            yj = x[j] - mu[j]
            pr_yj = pr * yj
            for i in 1:p
                yi = x[i] - mu[i]
                cov_new[i, j] += pr_yj * yi
            end
        end
    end
    if eff_n > (p + 1)
        cov_new ./= eff_n
        return _invert_and_det(cov_new)
    else
        return (inv=value.inv, det=-1.0)
    end
end

function normal_pdf_safe(X::Any, params)
    cov = val(params[:cov])
    if cov.det != -1
        x_vec = _as_fvec(X)::Vector{Float64}
        p = length(x_vec)
        mu = _as_fvec(val(params[:mu]))::Vector{Float64}
        quad = _quad_form(x_vec, mu, (cov.inv)::Matrix{Float64})
        return (2pi)^(-p / 2) * cov.det^(-0.5) * exp(-0.5 * quad)
    else
        return 0.0
    end
end

MtvNormalSafe(p) = ParsaDensity(
    normal_pdf_safe,
    normal_pdf_log,
    (x) -> normal_input(x, p),
    :mu => Parameter(zeros(Float64, p), normal_mean_update_safe),
    :cov => Parameter((inv=Matrix{Float64}(I, p, p), det=1.0), div(p * (p + 1), 2), normal_covariance_update_safe),
)

####### NormalDouble #########

function _normal_mean_update_double(value::Any, index_package::SummationPackage, other_mu_key::Symbol)
    p = length(value)
    mu_new = zeros(Float64, p)
    cov = zeros(Float64, p, p)
    total_pr = 0.0
    for (x_obs, pr_raw, params) in index_package
        pr = Float64(pr_raw)
        pr == 0.0 && continue
        total_pr += pr
        cov_inv = (val(params[:cov]).inv)::Matrix{Float64}
        x = _as_fvec(val(x_obs))::Vector{Float64}
        other_mu = _as_fvec(val(params[other_mu_key]))::Vector{Float64}
        @inbounds for j in 1:p
            diff_j = x[j] - other_mu[j]
            for i in 1:p
                cij = pr * cov_inv[i, j]
                cov[i, j] += cij
                mu_new[i] += cij * diff_j
            end
        end
    end
    total_pr <= 1e-12 && return copy(value)
    return ldiv!(lu!(cov), mu_new)
end

normal_mean_update_double_1(value::Any, index_package::SummationPackage, log_pdf::Function) =
    _normal_mean_update_double(value, index_package, :mu2)

normal_mean_update_double_2(value::Any, index_package::SummationPackage, log_pdf::Function) =
    _normal_mean_update_double(value, index_package, :mu1)

function normal_covariance_update_double(value::Any, index_package::SummationPackage, log_pdf::Function)
    p = size(value.inv, 1)
    cov_new = zeros(Float64, p, p)
    sum_taus = 0.0
    for (x_obs, pr_raw, params) in index_package
        pr = Float64(pr_raw)
        pr == 0.0 && continue
        sum_taus += pr
        x = _as_fvec(val(x_obs))::Vector{Float64}
        mu1 = _as_fvec(val(params[:mu1]))::Vector{Float64}
        mu2 = _as_fvec(val(params[:mu2]))::Vector{Float64}
        @inbounds for j in 1:p
            yj = x[j] - (mu1[j] + mu2[j])
            pr_yj = pr * yj
            for i in 1:p
                yi = x[i] - (mu1[i] + mu2[i])
                cov_new[i, j] += pr_yj * yi
            end
        end
    end
    sum_taus <= 1e-12 && return value
    cov_new ./= sum_taus
    return _invert_and_det(cov_new)
end

function normal_pdf_double(X::Any, params)
    x_vec = _as_fvec(X)::Vector{Float64}
    p = length(x_vec)
    mu1 = _as_fvec(val(params[:mu1]))::Vector{Float64}
    mu2 = _as_fvec(val(params[:mu2]))::Vector{Float64}
    cov = val(params[:cov])
    cov_inv = (cov.inv)::Matrix{Float64}
    quad = 0.0
    @inbounds for j in 1:p
        yj = x_vec[j] - (mu1[j] + mu2[j])
        s = 0.0
        for i in 1:p
            s += (x_vec[i] - (mu1[i] + mu2[i])) * cov_inv[i, j]
        end
        quad += s * yj
    end
    return (2pi)^(-p / 2) * cov.det^(-0.5) * exp(-0.5 * quad)
end

MtvNormalDouble(p) = ParsaDensity(
    normal_pdf_double,
    normal_pdf_log,
    (x) -> normal_input(x, p),
    :mu1 => Parameter(zeros(Float64, p), normal_mean_update_double_1),
    :mu2 => Parameter(zeros(Float64, p), normal_mean_update_double_2),
    :cov => Parameter((inv=Matrix{Float64}(I, p, p), det=1.0), div(p * (p + 1), 2), normal_covariance_update_double),
)

####### Normal Parsa (Parsimonious Gaussian) #########

function normal_parsa_a_update(a, package_index, log_pdf)
    a_new = 0.0
    sum_taus = 0.0
    p = 0
    for (x_obs, pr_raw, params) in package_index
        x = _as_fvec(val(x_obs))::Vector{Float64}
        p = length(x)
        pr = Float64(pr_raw)
        pr == 0.0 && continue
        sum_taus += pr
        mu = _as_fvec(val(params[:mu]))::Vector{Float64}
        V = val(params[:V])::Matrix{Float64}
        L = val(params[:L])::Vector{Float64}
        quad = 0.0
        @inbounds for j in 1:p
            zj = 0.0
            for i in 1:p
                zj += V[i, j] * (x[i] - mu[i])
            end
            quad += (zj * zj) / L[j]
        end
        a_new += pr * quad
    end
    (sum_taus <= 1e-12 || p == 0) && return Float64(a)
    return max(a_new / (sum_taus * p), 1e-12)
end

function normal_parsa_L_update(L, package_index, log_pdf)
    p = length(L)
    L_diag = zeros(Float64, p)
    sum_taus = 0.0
    for (x_obs, pr_raw, params) in package_index
        pr = Float64(pr_raw)
        pr == 0.0 && continue
        sum_taus += pr
        x = _as_fvec(val(x_obs))::Vector{Float64}
        mu = _as_fvec(val(params[:mu]))::Vector{Float64}
        V = val(params[:V])::Matrix{Float64}
        a = Float64(val(params[:a]))
        weight = pr / a
        @inbounds for j in 1:p
            zj = 0.0
            for i in 1:p
                zj += V[i, j] * (x[i] - mu[i])
            end
            L_diag[j] += weight * (zj * zj)
        end
    end
    sum_taus <= 1e-12 && return copy(L)
    prod_val = 1.0
    @inbounds for j in 1:p
        if L_diag[j] <= 1e-12
            L_diag[j] = 1e-12
        end
        prod_val *= L_diag[j]
    end
    inv_geom_mean = 1.0 / (prod_val^(1.0 / p))
    @inbounds for j in 1:p
        L_diag[j] *= inv_geom_mean
    end
    return L_diag
end

@inline function _orthogonality_sum_diff(V::Matrix{Float64})
    p = size(V, 1)
    total = 0.0
    @inbounds for j in 1:p
        for i in 1:p
            dot_ij = 0.0
            for k in 1:p
                dot_ij += V[k, i] * V[k, j]
            end
            total += i == j ? (dot_ij - 1.0) : dot_ij
        end
    end
    return abs(total)
end

@inline function _sym2x2_eigvecs(d11::Float64, d12::Float64, d22::Float64)
    if abs(d12) <= 1e-15
        if d11 <= d22
            return (1.0, 0.0, 0.0, 1.0)
        else
            return (0.0, 1.0, -1.0, 0.0)
        end
    end
    diff = d11 - d22
    r = hypot(diff, 2.0 * d12)
    if diff >= 0.0
        v1 = 2.0 * d12
        v2 = -diff - r
    else
        v1 = diff - r
        v2 = 2.0 * d12
    end
    nrm = hypot(v1, v2)
    e11 = v1 / nrm
    e21 = v2 / nrm
    return (e11, e21, -e21, e11)
end

function _optimize_V_from_groups!(V::Matrix{Float64}, az_list::Vector{Tuple{Vector{Float64},Matrix{Float64}}})
    p = size(V, 1)
    step = 0
    while (_orthogonality_sum_diff(V) > 1e-5 || step <= 3) && step < 10000
        step += 1
        if step == 10000
            error("max iter on V reached")
        end
        for _ in 1:3
            i = rand(1:p)
            j = rand(1:(p-1))
            if j >= i
                j += 1
            end
            d11 = 0.0
            d12_val = 0.0
            d22 = 0.0
            for (A, W) in az_list
                coeff = (1.0 / A[i]) - (1.0 / A[j])
                z11 = 0.0
                z12 = 0.0
                z22 = 0.0
                @inbounds for c in 1:p
                    vic = V[c, i]
                    vjc = V[c, j]
                    w_vi = 0.0
                    w_vj = 0.0
                    for r in 1:p
                        wrc = W[r, c]
                        w_vi += wrc * V[r, i]
                        w_vj += wrc * V[r, j]
                    end
                    z11 += w_vi * vic
                    z12 += w_vi * vjc
                    z22 += w_vj * vjc
                end
                d11 += coeff * z11
                d12_val += coeff * z12
                d22 += coeff * z22
            end
            (e11, e21, e12, e22) = _sym2x2_eigvecs(d11, d12_val, d22)
            @inbounds for r in 1:p
                vi = V[r, i]
                vj = V[r, j]
                V[r, i] = vi * e11 + vj * e21
                V[r, j] = vi * e12 + vj * e22
            end
        end
    end
    return V
end

function normal_parsa_V_update(V_, package_index, log_pdf)
    V = copy(V_)
    p = size(V, 1)
    groups = Dict{Tuple{UInt64,UInt64},Tuple{Vector{Float64},Matrix{Float64}}}()
    total_pr = 0.0
    for (x_obs, pr_raw, params) in package_index
        pr = Float64(pr_raw)
        pr == 0.0 && continue
        total_pr += pr
        L_param = params[:L]
        a_param = params[:a]
        key = (objectid(L_param), objectid(a_param))
        entry = get(groups, key, nothing)
        if isnothing(entry)
            L_val = val(L_param)::Vector{Float64}
            a_val = Float64(val(a_param))
            entry = (L_val .* a_val, zeros(Float64, p, p))
            groups[key] = entry
        end
        W = entry[2]
        x = _as_fvec(val(x_obs))::Vector{Float64}
        mu = _as_fvec(val(params[:mu]))::Vector{Float64}
        @inbounds for col in 1:p
            yc = x[col] - mu[col]
            pr_yc = pr * yc
            for row in 1:p
                yr = x[row] - mu[row]
                W[row, col] += pr_yc * yr
            end
        end
    end
    total_pr <= 1e-12 && return V
    az_list = collect(values(groups))
    return _optimize_V_from_groups!(V, az_list)
end

function _compute_parsa_cov_inv!(cov_inv::Matrix{Float64}, a::Float64, L::Vector{Float64}, V::Matrix{Float64})
    p = length(L)
    fill!(cov_inv, 0.0)
    inv_a = 1.0 / a
    @inbounds for k in 1:p
        scale = inv_a / L[k]
        for j in 1:p
            s_vj = scale * V[j, k]
            for i in 1:p
                cov_inv[i, j] += V[i, k] * s_vj
            end
        end
    end
    return cov_inv
end

function parsa_mean_update(value, index_package, log_pdf)
    p = length(value)
    mu_new = zeros(Float64, p)
    cov = zeros(Float64, p, p)
    cov_inv = zeros(Float64, p, p)
    last_key = (UInt64(0), UInt64(0), UInt64(0))
    total_pr = 0.0
    for (x_obs, pr_raw, params) in index_package
        pr = Float64(pr_raw)
        pr == 0.0 && continue
        total_pr += pr
        a_par = params[:a]
        L_par = params[:L]
        V_par = params[:V]
        key = (objectid(a_par), objectid(L_par), objectid(V_par))
        if key != last_key
            _compute_parsa_cov_inv!(
                cov_inv,
                Float64(p_v(a_par)),
                p_v(L_par)::Vector{Float64},
                p_v(V_par)::Matrix{Float64},
            )
            last_key = key
        end
        x = _as_fvec(x_v(x_obs))::Vector{Float64}
        @inbounds for j in 1:p
            xj = x[j]
            for i in 1:p
                cij = pr * cov_inv[i, j]
                cov[i, j] += cij
                mu_new[i] += cij * xj
            end
        end
    end
    total_pr <= 1e-12 && return copy(value)
    return ldiv!(lu!(cov), mu_new)
end

function normal_parsa_pdf_2(X, params)
    x_vec = _as_fvec(X)::Vector{Float64}
    p = length(x_vec)
    mu = _as_fvec(val(params[:mu]))::Vector{Float64}
    a = Float64(p_v(params[:a]))
    L = p_v(params[:L])::Vector{Float64}
    V = p_v(params[:V])::Matrix{Float64}
    quad = 0.0
    @inbounds for j in 1:p
        zj = 0.0
        for i in 1:p
            zj += V[i, j] * (x_vec[i] - mu[i])
        end
        quad += (zj * zj) / L[j]
    end
    return -0.5 * (p * LOG_2PI + p * log(max(a, eps(Float64))) + quad / a)
end

function normal_parsa_pdf_log_2(X, params)
    x_vec = _as_fvec(X)::Vector{Float64}
    mu = _as_fvec(params[:mu])::Vector{Float64}
    a = Float64(params[:a])
    eigval = params[:L]
    V = params[:V]
    p = length(mu)
    quad = 0.0
    @inbounds for j in 1:p
        zj = 0.0
        for i in 1:p
            zj += V[i, j] * (x_vec[i] - mu[i])
        end
        quad += (zj * zj) / eigval[j]
    end
    return -p * log(max(a, eps(Float64))) - quad / a
end

parsa_V(p) = Parameter(Matrix{Float64}(I, p, p), div(p * (p - 1), 2), normal_parsa_V_update)
parsa_L(p) = Parameter(ones(Float64, p), p - 1, normal_parsa_L_update)
parsa_a() = Parameter(1.0, 1, normal_parsa_a_update)

ParsimoniousNormal(p) = ParsaDensity(
    normal_parsa_pdf_2,
    normal_parsa_pdf_log_2,
    (x) -> normal_input(x, p),
    :mu => Parameter(zeros(Float64, p), parsa_mean_update),
    :a => parsa_a(),
    :L => parsa_L(p),
    :V => parsa_V(p),
)
