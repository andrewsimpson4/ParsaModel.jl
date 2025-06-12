using Ipopt, StatsBase, Distributions
using JuMP: Model, set_silent, @variable, @constraint, @objective, optimize!, value, set_optimizer_attribute, @operator

##### Normal Model #######


# function normal_mean_update(value, index_package, log_pdf)
#     mu_new = zeros(length(value))
#     for (x, pr,) in index_package
#         mu_new += pr * x
#     end
#     taus = [d[2] for d in index_package]
#     mu_new /= sum(taus)
#     return mu_new
# end

# function normal_covariance_update(value, index_package, log_pdf)
#     cov_new = zeros(size(value))
#     for (x, pr, params) in index_package
#         cov_new += pr * ((x - (params[:mu])) * (x - (params[:mu]))')
#     end
#     taus = [d[2] for d in index_package]
#     cov_new ./= sum(taus)
#     # return cov_new
#     return cholesky(Hermitian(cov_new)).L
# end

# function normal_pdf(X, params)
#     # N = MvNormal(params[:mu], params[:cov])
#     # pdf(N, X)
#     p = length(X)
#     y = params[:cov] \ (X - params[:mu])
#     (2pi)^(-p/2) * det(params[:cov])^(-1) * exp(-1/2 * y' * y)
# end


function normal_mean_update(value, index_package, log_pdf)
    mu_new = zeros(length(value))
    cov = zeros(length(value), length(value))
    for (_, pr, params) in index_package
        cov += pr * p_v(params[:cov]).inv
    end
    for (x, pr, params) in index_package
        mu_new += pr * p_v(params[:cov]).inv * x_v(x)
    end
    mu_new = cov \ mu_new
    return mu_new
end

function normal_covariance_update(value, index_package, log_pdf)
    cov_new = zeros(size(value.inv))
    for (x, pr, params) in index_package
        y = x_v(x) - p_v(params[:mu])
        cov_new += pr * (y * y')
    end
    taus = [d[2] for d in index_package]
    cov_new ./= sum(taus)
    return (inv = inv(cov_new), det = det(cov_new))
end

function normal_pdf(X, params)
    p = length(X)
    (2pi)^(-p/2) * params[:cov].value.value.det^(-1/2) * exp((-1/2 * (X - params[:mu].value.value)' * params[:cov].value.value.inv * (X - params[:mu].value.value)))
end

function normal_pdf_log(X, params)
    N = MvNormal(params[:mu], params[:cov])
    logpdf(N, X)
end

function normal_cov_post(L)
    # inv(L.inv)
    L
end

normal_input(x, p) = length(x) == p && all(isa.(x, Real))

Normal_Model(p) = Parsa_density(normal_pdf, normal_pdf_log, (x) -> normal_input(x, p),
                                :mu => Parsa_Parameter(zeros(p), normal_mean_update),
                                :cov => Parsa_Parameter((inv = diagm(ones(p)), det = 1), p * (p + 1) / 2, normal_covariance_update, normal_cov_post))




### Normal singular


function normal_covariance_update_singular(value, index_package, log_pdf)
    cov_new = zeros(size(value))
    for (x, pr, params) in index_package
        cov_new += pr * ((x - params[:mu]) * (x - params[:mu])')
    end
    taus = [d[2] for d in index_package]
    cov_new ./= sum(taus)
    # return cov_new
    return cholesky(Hermitian(cov_new .+ diagm(zeros(size(cov_new)[1]) .+ 10^-10))).L
end


Normal_Model_singular(p) = Parsa_density(normal_pdf, normal_pdf_log, (x) -> normal_input(x, p),
                                :mu => Parsa_Parameter(zeros(p), normal_mean_update),
                                :cov => Parsa_Parameter(diagm(ones(p)), p * (p + 1) / 2, normal_covariance_update_singular, normal_cov_post))



### Double mean

function normal_mean_update_2_1(value, index_package, log_pdf)
    mu_new = zeros(length(value))
    for (x, pr, params) in index_package
        mu_new += pr * (x .- params[:mu2])
    end
    taus = [d[2] for d in index_package]
    mu_new /= sum(taus)
    return mu_new
end
function normal_mean_update_2_2(value, index_package, log_pdf)
    mu_new = zeros(length(value))
    for (x, pr, params) in index_package
        mu_new += pr * (x .- params[:mu1])
    end
    taus = [d[2] for d in index_package]
    mu_new /= sum(taus)
    return mu_new
end

function normal_covariance_update_2(value, index_package, log_pdf)
    cov_new = zeros(size(value))
    for (x, pr, params) in zip(X, taus, parameter_maps)
        cov_new += pr * ((x - (params[:mu1] .+ params[:mu2])) * (x - (params[:mu1] .+ params[:mu2]))')
    end
    taus = [d[2] for d in index_package]
    cov_new ./= sum(taus)
    cov.value = cov_new
end

function normal_pdf_2(X, params)
    N = MvNormal(params[:mu1] .+ params[:mu2], params[:cov])
    pdf(N, X)
end

function normal_pdf_log_2(X, params)
    N = MvNormal(params[:mu1] .+ params[:mu2], params[:cov])
    logpdf(N, X)
end

Double_Normal_Model(p) = LMMM_Base(pdf=normal_pdf_2, pdf_log=normal_pdf_log_2, (x) -> normal_input(x, p),
                            :mu1 => Parsa_Parameter(ones(p), normal_mean_update_2_1),
                            :mu2 => Parsa_Parameter(ones(p), normal_mean_update_2_2),
                            :cov => Parsa_Parameter(diagm(ones(p)), p * (p + 1) / 2, normal_covariance_update_2))


####### Normal Parsa #########


function get_objective_function(param, package_index, pdf_func)
    X = [p[1] for p in package_index]
    paramter_map = [Dict([ke => p_v(va) for (ke, va) in p[3]]) for p in package_index]
    tau_map = [p[2] for p in package_index]
    N_keys = Vector{}(undef, length(X))
    for i in eachindex(X)
        pars = paramter_map[i]
        par_search = [t for t in values(pars)]
        param_index = [i for i in 1:length(par_search) if (par_search[i]) == param][1]
        N_keys[i] = collect(keys(pars))[param_index]
    end
    function(ve)
        res = 0.0
        for i in eachindex(X)
            pars = paramter_map[i]
            x = x_v(X[i])
            tau = tau_map[i]
            pars[N_keys[i]] = ve
            res = res + tau * pdf_func(x, pars)
        end
        return res
    end
end

function optimizeOrthogonal(param, package_index, log_pdf)
    func = get_objective_function(param, package_index, log_pdf)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 3)
    set_silent(model)
    p = size(param)[1]
    @variable(model, V[i=1:p, j=1:p], start=param[i,j]);
    @constraint(model, V*V' - I == zeros(p,p));
    @objective(model, Max, func(V))
    optimize!(model)
    return value.(V)

end



function normal_parsa_a_update(a, package_index, log_pdf)
    p = length(x_v(collect(package_index)[1][1]))
    a_new = 0
    for (x, pr, params) in package_index
        y = (x_v(x) - p_v(params[:mu]))
        a_new += pr *  (y' * p_v(params[:V]) * diagm(1 ./ p_v(params[:L])) * p_v(params[:V])' * y)
    end
    taus = [d[2] for d in package_index]
    a_new = a_new / (sum(taus) * p )
    return a_new
end

function normal_parsa_L_update(L, package_index, log_pdf)
    p = length(x_v(collect(package_index)[1][1]))
    L_new = zeros(size(diagm(L)))
    for (x, pr, params) in package_index
        y = (x_v(x) - p_v(params[:mu]))
        L_new += pr * ( p_v(params[:V])' * y * y' * p_v(params[:V]) * p_v(params[:a])^(-1))
    end
    L_new = diag(L_new) / prod(diag(L_new))^(1 / p)
    return L_new
end

function normal_parsa_V_update(V, package_index, log_pdf)
    opt_new = sum(V)
    opt_old = sum(V) + 10
    while abs(sum(opt_new .- opt_old)) / abs(sum(opt_old)) > 10^-10
        opt_old = opt_new
        # i,j = sample(1:p, 2; replace=false)
        for i in 1:p
            for j in 1:p
                if i != j
                    d1 = V[:,i]
                    d2 = V[:,j]
                    D = zeros(size(2,2))
                    for (x, pr, params) in package_index
                        A =  sort(params[:L]; rev=true) * params[:a]
                        Z = [d1 d2]' * pr * (x - params[:mu]) * (x - params[:mu])' * [d1 d2]
                        # D = D .+ ((1 / A[i] - 1 / A[j]) .* Z)
                        D = D .+ Z
                        # println((1 / A[i] - 1 / A[j]))
                    end
                    V[:,i] = [d1 d2] * eigvecs(D)[:,2]
                    V[:,j] = [d1 d2] * eigvecs(D)[:,1]
                    opt_new = 0
                    for (x, pr, params) in package_index
                        opt_new = opt_new + pr * tr(V * diagm(1 ./ (params[:L] * params[:a])) * V' * (x - params[:mu]) * (x - params[:mu])')
                    end
                end
            end
        end
    end
    return V
end

parsa_V(p) = Parameter(value = ParameterValue(value = diagm(ones(p))), update = normal_parsa_V_update)
parsa_V_opt(p) = Parsa_Parameter(diagm(ones(p)), optimizeOrthogonal)
parsa_L(p) = Parameter(value = ParameterValue(value = ones(p)), update = normal_parsa_L_update)
parsa_a() = Parameter(value = ParameterValue(value = 1), update = normal_parsa_a_update)


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
    # taus = [d[2] for d in index_package]
    # mu_new /= sum(taus)
    mu_new = inv(cov) * mu_new
    return mu_new
end

function normal_parsa_pdf_2(X, params)
    N = MvNormal(p_v(params[:mu]), Symmetric(p_v(params[:a]) .* p_v(params[:V]) * diagm(p_v(params[:L])) * p_v(params[:V])'))
    pdf(N, X)
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


Normal_Parsa_Model(p) = Parsa_density(normal_parsa_pdf_2, normal_parsa_pdf_log_2, (x) -> normal_input(x, p),
        :mu=> Parsa_Parameter(zeros(p), parsa_mean_update),
        :a => parsa_a(),
        :L => parsa_L(p),
        :V => parsa_V_opt(p))

