using StatsBase

##### Normal Model #######

function normal_mean_update(value::Any, index_package::SummationPackage, log_pdf::Function)
    mu_new = zeros(length(value))
    cov = zeros(length(value), length(value))
    for (x, pr, params) in index_package
        cov_inv = val(params[:cov]::Parameter).inv
        cov .+= pr .* cov_inv
        mu_new .+= pr .* (cov_inv * val(x))
    end
    mu_new = cov \ mu_new
    return mu_new
end

function normal_covariance_update(value::Any, index_package::SummationPackage, log_pdf::Function)
    cov_new = zeros(size(value.inv))
    tau_sum = 0.0
    for (x, pr, params) in index_package
        y = val(x) - val(params[:mu]::Parameter)
        cov_new .+= pr .* (y * y')
        tau_sum += pr
    end
    if tau_sum > 0.0
        cov_new ./= tau_sum
    end
    d = det(cov_new)
    if !(d > eps(Float64))
        @warn "Singular covariance detected; adding regularization"
        reg = eps(Float64)
        while !(d > eps(Float64))
            cov_new .+= reg .* I(size(cov_new, 1))
            reg *= 10.0
            d = det(cov_new)
        end
    end
    return (inv=inv(cov_new), det=d)
    # return (inv = inv(cov_new), chol = inv(cholesky(cov_new).L), det = logdet(cov_new))
end

@inline function _quadform_diff(X::AbstractVector, mu::AbstractVector, cov_inv::AbstractMatrix)
    q = 0.0
    @inbounds for j in eachindex(X, mu)
        yj = X[j] - mu[j]
        s = 0.0
        for i in eachindex(X, mu)
            s += (X[i] - mu[i]) * cov_inv[i, j]
        end
        q += s * yj
    end
    return q
end

@inline function normal_pdf(X::Any, params::Dict)
    p = length(X)
    mu = val(params[:mu]::Parameter)::Vector{Float64}
    cov = val(params[:cov]::Parameter)::NamedTuple{(:inv, :det), Tuple{Matrix{Float64}, Float64}}
    # (-p / 2) * log(2pi) + (-1 / 2) * log(val(params[:cov]).det) + ((-1 / 2 * y' * val(params[:cov]).inv * y))
    return (-0.5 * p) * log(2pi) - 0.5 * log(cov.det) - 0.5 * _quadform_diff(X, mu, cov.inv)
    # y = val(params[:cov]).chol * (X - val(params[:mu]))
    # (-p/2) * log(2pi) + (-1/2) * (val(params[:cov]).det) + ((-1/2 * y' * y))
end

function normal_pdf_log(X::Any, params::Dict)
    N = MvNormal(params[:mu], params[:cov])
    logpdf(N, X)
end

function normal_cov_post(L)
    # inv(L.inv)
    L
end

normal_input(x, p) = length(x) == p && all(isa.(x, Real))

MtvNormal(p) = ParsaDensity(normal_pdf, normal_pdf_log, (x) -> normal_input(x, p),
    :mu => Parameter(zeros(p), normal_mean_update),
    :cov => Parameter((inv=diagm(ones(p)), det=1.0), p * (p + 1) / 2, normal_covariance_update))


#### safe normal

function normal_mean_update_safe(value::Any, index_package::SummationPackage, log_pdf::Function)
    mu_new = zeros(length(value))
    cov = zeros(length(value), length(value))
    eff_n = 0
    p = length(value)
    for (_, pr, params) in index_package
        eff_n += pr
        cov += pr * val(params[:cov]).inv
    end
    for (x, pr, params) in index_package
        mu_new += pr * val(params[:cov]).inv * val(x)
    end
    if eff_n > (p + 1)
        mu_new = cov \ mu_new
        return mu_new
    else
        return value #zeros(length(value))
    end
end

function normal_covariance_update_safe(value::Any, index_package::SummationPackage, log_pdf::Function)
    cov_new = zeros(size(value.inv))
    eff_n = 0
    p = size(value.inv)[1]
    for (x, pr, params) in index_package
        y = val(x) - val(params[:mu])
        cov_new += pr * (y * y')
        eff_n += pr
    end
    if eff_n > (p + 1)
        taus = [d[2] for d in index_package]
        cov_new ./= sum(taus)
        return (inv=inv(cov_new), det=det(cov_new))
    else
        return (inv=value.inv, det=-1.0)
    end
end

@inline function normal_pdf_safe(X::Any, params::Dict)
    cov = val(params[:cov]::Parameter)::NamedTuple{(:inv, :det), Tuple{Matrix{Float64}, Float64}}
    if cov.det != -1.0
        p = length(X)
        mu = val(params[:mu]::Parameter)::Vector{Float64}
        return (2pi)^(-0.5 * p) * cov.det^(-0.5) * exp(-0.5 * _quadform_diff(X, mu, cov.inv))
    else
        return 0.0
    end
end

MtvNormalSafe(p) = ParsaDensity(normal_pdf_safe, normal_pdf_log, (x) -> normal_input(x, p),
    :mu => Parameter(zeros(p), normal_mean_update_safe),
    :cov => Parameter((inv=diagm(ones(p)), det=1.0), p * (p + 1) / 2, normal_covariance_update_safe))



####### NormalDouble #########
##### Normal Model #######

function normal_mean_update_double_1(value::Any, index_package::SummationPackage, log_pdf::Function)
    mu_new = zeros(length(value))
    cov = zeros(length(value), length(value))
    for (_, pr, params) in index_package
        cov += pr * val(params[:cov]).inv
    end
    for (x, pr, params) in index_package
        mu_new += pr * val(params[:cov]).inv * (val(x) - val(params[:mu2]))
    end
    mu_new = cov \ mu_new
    return mu_new
end

function normal_mean_update_double_2(value::Any, index_package::SummationPackage, log_pdf::Function)
    mu_new = zeros(length(value))
    cov = zeros(length(value), length(value))
    for (_, pr, params) in index_package
        cov += pr * val(params[:cov]).inv
    end
    for (x, pr, params) in index_package
        mu_new += pr * val(params[:cov]).inv * (val(x) - val(params[:mu1]))
    end
    mu_new = cov \ mu_new
    return mu_new
end

function normal_covariance_update_double(value::Any, index_package::SummationPackage, log_pdf::Function)
    cov_new = zeros(size(value.inv))
    for (x, pr, params) in index_package
        y = val(x) - (val(params[:mu1]) + val(params[:mu2]))
        cov_new += pr * (y * y')
    end
    taus = [d[2] for d in index_package]
    cov_new ./= sum(taus)
    return (inv=inv(cov_new), det=det(cov_new))
end

@inline function _quadform_diff_double(X::AbstractVector, mu1::AbstractVector, mu2::AbstractVector, cov_inv::AbstractMatrix)
    q = 0.0
    @inbounds for j in eachindex(X, mu1, mu2)
        yj = X[j] - (mu1[j] + mu2[j])
        s = 0.0
        for i in eachindex(X, mu1, mu2)
            s += (X[i] - (mu1[i] + mu2[i])) * cov_inv[i, j]
        end
        q += s * yj
    end
    return q
end

@inline function normal_pdf_double(X::Any, params::Dict)
    p = length(X)
    mu1 = val(params[:mu1]::Parameter)::Vector{Float64}
    mu2 = val(params[:mu2]::Parameter)::Vector{Float64}
    cov = val(params[:cov]::Parameter)::NamedTuple{(:inv, :det), Tuple{Matrix{Float64}, Float64}}
    return (2pi)^(-0.5 * p) * cov.det^(-0.5) * exp(-0.5 * _quadform_diff_double(X, mu1, mu2, cov.inv))
end

MtvNormalDouble(p) = ParsaDensity(normal_pdf_double, normal_pdf_log, (x) -> normal_input(x, p),
    :mu1 => Parameter(zeros(p), normal_mean_update_double_1),
    :mu2 => Parameter(zeros(p), normal_mean_update_double_2),
    :cov => Parameter((inv=diagm(ones(p)), det=1.0), p * (p + 1) / 2, normal_covariance_update_double))

####### Normal Parsa #########

function normal_parsa_a_update(a, package_index, log_pdf)
    p = length(val(first(package_index)[1]))
    a_new = 0.0
    tau_sum = 0.0
    for (x, pr, params) in package_index
        y = val(x) - val(params[:mu]::Parameter)
        z = val(params[:V]::Parameter)' * y
        L = val(params[:L]::Parameter)
        a_new += pr * sum((z .^ 2) ./ L)
        tau_sum += pr
    end
    a_new = a_new / (tau_sum * p)
    return a_new
end

function normal_parsa_L_update(L, package_index, log_pdf)
    p = length(L)
    L_new = zeros(p)
    for (x, pr, params) in package_index
        y = val(x) - val(params[:mu]::Parameter)
        z = val(params[:V]::Parameter)' * y
        a = val(params[:a]::Parameter)
        L_new .+= (pr / a) .* (z .^ 2)
    end
    d = prod(L_new)
    if !(d > eps(Float64))
        @warn "Singular L matrix detected; adding regularization"
        reg = eps(Float64)
        while !(d > eps(Float64))
            L_new .+= reg
            reg *= 10.0
            d = prod(L_new)
        end
    end
    L_new ./= d^(1 / p)
    return L_new
end

function normal_parsa_V_update(V_, package_index, log_pdf)
    V = copy(V_)
    opt_new = sum(V)
    opt_old = sum(V) + 10
    AA = Vector{}()
    ZZ = Vector{}()
    p = length(val(collect(package_index)[1][1]))
    for (x, pr, params) in package_index
        push!(AA, val(params[:L]) * val(params[:a]))
        push!(ZZ, pr * (val(x) - val(params[:mu])) * (val(x) - val(params[:mu]))')
    end
    zipped_az = zip(AA, ZZ)
    # while abs(sum(opt_new .- opt_old)) / abs(sum(opt_old)) > 10^-5
    # println(V' * V)
    step = 0
    while (abs.(sum(V' * V - I)) > 10^-5 || step <= 3) && step < 10000
        step = step + 1

        if step == 10000
            # @warn "max iter on V reached"
            error("max iter on V reached")
        end

        # println(abs(sum(opt_new .- opt_old)) / abs(sum(opt_old)))
        for _ in 1:3
            i, j = sample(1:p, 2, replace=false)
            # for i in 1:(p - 1)
            #     for j in (i+1):p
            # d1 = V[:,i]
            # d2 = V[:,j]
            d12 = V[:, [i, j]]
            D = zeros(size(2, 2))
            for (A, Zz) in zipped_az
                Z = d12' * Zz * d12
                D = D .+ ((1 / A[i] - 1 / A[j]) .* Z)
            end
            eig = eigvecs(D)
            V[:, i] = d12 * eig[:, 1]
            V[:, j] = d12 * eig[:, 2]
            #     end
            # end
        end
        opt_old = opt_new
        opt_new = sum(V)
    end
    return V
end

function parsa_mean_update(value, index_package, log_pdf)
    mu_new = zeros(length(value))
    cov = zeros(length(value), length(value))
    for (x, pr, params) in index_package
        V = p_v(params[:V]::Parameter)
        L = p_v(params[:L]::Parameter)
        a = p_v(params[:a]::Parameter)
        cov_inv = (1 / a) * V * Diagonal(1 ./ L) * V'
        cov .+= pr .* cov_inv
        mu_new .+= pr .* (cov_inv * x_v(x))
    end
    mu_new = cov \ mu_new
    return mu_new
end

@inline function _parsa_quadform(X::AbstractVector, mu::AbstractVector, V::AbstractMatrix, L::AbstractVector, a::Float64)
    q = 0.0
    @inbounds for j in eachindex(L)
        zj = 0.0
        for i in eachindex(X, mu)
            zj += V[i, j] * (X[i] - mu[i])
        end
        q += (zj * zj) / L[j]
    end
    return q / a
end

@inline function normal_parsa_pdf_2(X, params)
    p = length(X)
    mu = val(params[:mu]::Parameter)::Vector{Float64}
    V = p_v(params[:V]::Parameter)::Matrix{Float64}
    L = p_v(params[:L]::Parameter)::Vector{Float64}
    a = p_v(params[:a]::Parameter)::Float64
    # (2pi)^(-p/2) * p_v(params[:a])^(-p/2) * exp((-1/2 * y' * cov_inv * y))

    return (-0.5 * p * log(2pi) - 0.5 * p * log(a) - 0.5 * _parsa_quadform(X, mu, V, L, a))
end

function normal_parsa_pdf_log_2(X, params)
    p = length((params[:mu]))
    a = (params[:a])
    eigval = (params[:L])
    mu = (params[:mu])
    V = (params[:V])
    cov_inv = 1 / a .* V * diagm(1 ./ eigval) * V'
    l = -p * log(a) - ((X - mu)' * cov_inv * (X - mu))
    return l
end


parsa_V(p) = Parameter(diagm(ones(p)), p * (p - 1) / 2, normal_parsa_V_update)
# parsa_V_opt(p) = Parameter(diagm(ones(p)), optimizeOrthogonal)
parsa_L(p) = Parameter(ones(p), p - 1, normal_parsa_L_update)
parsa_a() = Parameter(1.0, 1, normal_parsa_a_update)

# ParsimoniousNormal(p) = ParsaDensity(normal_parsa_pdf_2, normal_parsa_pdf_log_2, (x) -> normal_input(x, p),
#         :mu=> Parameter(zeros(p), parsa_mean_update),
#         :a => parsa_a(),
#         :L => parsa_L(p),
#         :V => parsa_V_opt(p))

ParsimoniousNormal(p) = ParsaDensity(normal_parsa_pdf_2, normal_parsa_pdf_log_2, (x) -> normal_input(x, p),
    :mu => Parameter(zeros(p), parsa_mean_update),
    :a => parsa_a(),
    :L => parsa_L(p),
    :V => parsa_V(p))
