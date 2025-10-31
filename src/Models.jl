using StatsBase

##### Normal Model #######

function normal_mean_update(value::Any, index_package::SummationPackage, log_pdf::Function)
    mu_new = zeros(length(value))
    cov = zeros(length(value), length(value))
    for (_, pr, params) in index_package
        cov += pr * val(params[:cov]).inv
    end
    for (x, pr, params) in index_package
        mu_new += pr * val(params[:cov]).inv * val(x)
    end
    mu_new = cov \ mu_new
    return mu_new
end

function normal_covariance_update(value::Any, index_package::SummationPackage, log_pdf::Function)
    cov_new = zeros(size(value.inv))
    for (x, pr, params) in index_package
        y = val(x) - val(params[:mu])
        cov_new += pr * (y * y')
    end
    taus = [d[2] for d in index_package]
    cov_new ./= sum(taus)
    return (inv = inv(cov_new), det = det(cov_new))
end

function normal_pdf(X::Any, params::Dict)
    p = length(X)
    y = (X - val(params[:mu]))
    (2pi)^(-p/2) * val(params[:cov]).det^(-1/2) * exp((-1/2 * y' * val(params[:cov]).inv * y))
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
                                :cov => Parameter((inv = diagm(ones(p)), det = 1), p * (p + 1) / 2, normal_covariance_update))


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
    if eff_n > (p+1)
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
    if eff_n > (p+1)
        taus = [d[2] for d in index_package]
        cov_new ./= sum(taus)
        return (inv = inv(cov_new), det = det(cov_new))
    else
        return (inv = value.inv, det = -1)
    end
end

function normal_pdf_safe(X::Any, params::Dict)
    if val(params[:cov]).det != -1
        p = length(X)
        y = (X - val(params[:mu]))
        (2pi)^(-p/2) * val(params[:cov]).det^(-1/2) * exp((-1/2 * y' * val(params[:cov]).inv * y))
    else
        return 0
    end
end

MtvNormalSafe(p) = ParsaDensity(normal_pdf_safe, normal_pdf_log, (x) -> normal_input(x, p),
                                :mu => Parameter(zeros(p), normal_mean_update_safe),
                                :cov => Parameter((inv = diagm(ones(p)), det = 1), p * (p + 1) / 2, normal_covariance_update_safe))



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
    return (inv = inv(cov_new), det = det(cov_new))
end

function normal_pdf_double(X::Any, params::Dict)
    p = length(X)
    y = (X - (val(params[:mu1]) + val(params[:mu2])))
    (2pi)^(-p/2) * val(params[:cov]).det^(-1/2) * exp((-1/2 * y' * val(params[:cov]).inv * y))
end

MtvNormalDouble(p) = ParsaDensity(normal_pdf_double, normal_pdf_log, (x) -> normal_input(x, p),
                                :mu1 => Parameter(zeros(p), normal_mean_update_double_1),
                                :mu2 => Parameter(zeros(p), normal_mean_update_double_2),
                                :cov => Parameter((inv = diagm(ones(p)), det = 1), p * (p + 1) / 2, normal_covariance_update_double))

####### Normal Parsa #########


# function get_objective_function(param, package_index, pdf_func)
#     X = [p[1] for p in package_index]
#     paramter_map = [Dict([ke => p_v(va) for (ke, va) in p[3]]) for p in package_index]
#     tau_map = [p[2] for p in package_index]
#     N_keys = Vector{}(undef, length(X))
#     for i in eachindex(X)
#         pars = paramter_map[i]
#         par_search = [t for t in values(pars)]
#         param_index = [i for i in 1:length(par_search) if (par_search[i]) == param][1]
#         N_keys[i] = collect(keys(pars))[param_index]
#     end
#     function(ve)
#         res = 0.0
#         for i in eachindex(X)
#             pars = paramter_map[i]
#             x = x_v(X[i])
#             tau = tau_map[i]
#             pars[N_keys[i]] = ve
#             res = res + tau * pdf_func(x, pars)
#         end
#         return res
#     end
# end

# function optimizeOrthogonal(param, package_index, log_pdf)
#     func = get_objective_function(param, package_index, log_pdf)
#     model = Model(Ipopt.Optimizer)
#     set_optimizer_attribute(model, "max_iter", 3)
#     set_silent(model)
#     p = size(param)[1]
#     @variable(model, V[i=1:p, j=1:p], start=param[i,j]);
#     @constraint(model, V*V' - I == zeros(p,p));
#     @objective(model, Max, func(V))
#     optimize!(model)
#     return value.(V)

# end

function normal_parsa_a_update(a, package_index, log_pdf)
    p = length(val(collect(package_index)[1][1]))
    a_new = 0
    for (x, pr, params) in package_index
        y = (val(x) - val(params[:mu]))
        a_new += pr *  (y' * val(params[:V]) * diagm(1 ./ val(params[:L])) * val(params[:V])' * y)
    end
    taus = [d[2] for d in package_index]
    a_new = a_new / (sum(taus) * p )
    return a_new
end

function normal_parsa_L_update(L, package_index, log_pdf)
    p = length(val(collect(package_index)[1][1]))
    L_new = zeros(size(diagm(L)))
    for (x, pr, params) in package_index
        y = (val(x) - val(params[:mu]))
        L_new += pr * ( val(params[:V])' * y * y' * val(params[:V]) * val(params[:a])^(-1))
    end
    L_new = diag(L_new) / prod(diag(L_new))^(1 / p)
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
    while (abs.(sum(V'*V - I)) > 10^-5 || step <= 3) && step < 10000
        step = step + 1

        if step == 10000
            @warn "max iter on V reached"
        end

        # println(abs(sum(opt_new .- opt_old)) / abs(sum(opt_old)))
        for _ in 1:4
                i,j = sample(1:p, 2, replace=false)
        # for i in 1:(p - 1)
        #     for j in (i+1):p
                # d1 = V[:,i]
                # d2 = V[:,j]
                d12 = V[:,[i,j]]
                D = zeros(size(2,2))
                for (A, Zz) in zipped_az
                    # Z = [d1 d2]' * Zz * [d1 d2]
                    Z = d12' * Zz * d12
                    D = D .+ ((1 / A[i] - 1 / A[j]) .* Z)
                end
                V[:,i] = d12 * eigvecs(D)[:,1]
                V[:,j] = d12 * eigvecs(D)[:,2]
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
    for (_, pr, params) in index_package
        cov_inv = 1/ p_v(params[:a]) * p_v(params[:V]) * diagm(1 ./ p_v(params[:L])) * p_v(params[:V])'
        cov += pr * cov_inv
    end
    for (x, pr, params) in index_package
        cov_inv = 1/ p_v(params[:a]) * p_v(params[:V]) * diagm(1 ./ p_v(params[:L])) * p_v(params[:V])'
        mu_new += pr * cov_inv * x_v(x)
    end
    mu_new = inv(cov) * mu_new
    return mu_new
end

function normal_parsa_pdf_2(X, params)
    cov_inv = (Symmetric(1 / p_v(params[:a]) * p_v(params[:V]) * diagm(1 ./ p_v(params[:L])) * p_v(params[:V])'))
    p = length(X)
    y = (X - val(params[:mu]))
    (2pi)^(-p/2) * p_v(params[:a])^(-p/2) * exp((-1/2 * y' * cov_inv * y))
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
parsa_L(p) = Parameter( ones(p), p-1, normal_parsa_L_update)
parsa_a() = Parameter(1, 1,  normal_parsa_a_update)

# ParsimoniousNormal(p) = ParsaDensity(normal_parsa_pdf_2, normal_parsa_pdf_log_2, (x) -> normal_input(x, p),
#         :mu=> Parameter(zeros(p), parsa_mean_update),
#         :a => parsa_a(),
#         :L => parsa_L(p),
#         :V => parsa_V_opt(p))

ParsimoniousNormal(p) = ParsaDensity(normal_parsa_pdf_2, normal_parsa_pdf_log_2, (x) -> normal_input(x, p),
        :mu=> Parameter(zeros(p), parsa_mean_update),
        :a => parsa_a(),
        :L => parsa_L(p),
        :V => parsa_V(p))
