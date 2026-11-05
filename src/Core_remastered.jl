using UnicodePlots
using Distributions
using OrderedCollections
using Base: IdSet
using CSV, DataFrames



function prime_X(X::Observation)
    domains = GetDependentVariable(X)
    for LV in domains
        push!(LV.dependent_X, X)
    end
end



function rawCallDomain(map::Vector{<:Pair})
    LV = Dict()
    for (ke, va) in map
        if isa(va, LV_wrap)
            LV[ke] = va.LV()
        elseif isa(va, Vector)
            LV[ke] = [isa(lv, Int) ? lv : lv.LV() for lv in va]
        else
            LV[ke] = va
        end
    end
    return LV
end

function callDomain(map::Vector{<:Pair})
    LV = Vector{LatentVaraible}()
    for (ke, va) in map
        if isa(va, LV_wrap)
            push!(LV, va.LV())
        elseif isa(va, Vector)
            for lv in va
                if !isa(lv, Int)
                    push!(LV, lv.LV())
                end
            end
        end
    end
    return (LV)
end

function callDomain(X::Observation)
    for LV in callDomain(X.T)
        if typeof(LV) == LatentVaraible
            if !LV.active
                return LV
            end
        end
    end
end

function exhaustiveIndependenceCheck(G::Vector{Vector{Observation}})
    G_depo = [unique(reduce(vcat, GetDependentVariable.(X))) for X in G]
    n = length(G)
    for i in 1:(n-1)
        for j in (i+1):(n)
            if !isempty(intersect(G_depo[i], G_depo[j]))
                for LV in intersect(G_depo[i], G_depo[j])
                    display(LV)
                end
                error("Oh no you found a bug... conditional non independent sets found")
            end
        end
    end
end

function GetDependentVariable(X_i::Observation)
    used_conditions = Vector{LatentVaraible}()
    GetDependentVariable!(X_i, used_conditions)
    return used_conditions
end
function GetDependentVariable!(X_i::Observation, used_conditions::Vector)
    condition = callDomain(X_i)
    if !isnothing(condition)
        push!(used_conditions, condition)
        K = typeof(condition.value_) == Unknown ? (1:condition.Z.K) : lv_v(condition)
        for k in K
            lv_set(condition, k)
            GetDependentVariable!(X_i, used_conditions)
            lv_set(condition, 0)
        end
    end
end

function getIndependentSets(X::Vector{Observation})
    domains = Dict([x => Set([lv for lv in GetDependentVariable(x) if !lv_isKnown(lv)]) for x in X])
    Xset = Set(X)
    domainSet = Set{LatentVaraible}()
    independent_sets = Vector{Vector{Observation}}()
    for x in X
        if x in Xset
            current_set = Vector{Observation}()
            push!(independent_sets, current_set)
            independent_search!(x, Xset, current_set, domains, domainSet)
        end
    end
    return independent_sets
end

function independent_search!(x, Xset, current_set, domains, domainSet)
    push!(current_set, x)
    delete!(Xset, x)
    for condition in setdiff(domains[x], domainSet)
        push!(domainSet, condition)
        for va in condition.dependent_X
            if (va in Xset)
                independent_search!(va, Xset, current_set, domains, domainSet)
            end
        end
    end
end

function wild_tau(tau)
    G = Categorical(tau)
    g = rand(G)
    tau_new = zeros(length(tau)) .+ 0
    tau_new[g] = 1
    tau_new /= sum(tau_new)
    return tau_new
end

Base.@kwdef mutable struct Node
    condition::Any
    parent::Any
    correction_factor::Any
    all_X::Vector{Observation}
    group_id = 0
end

Base.@kwdef mutable struct Package
    likelihood::Any
    pi_used::Any
    params_used::Any
    group_id::Any
    X::Observation
    LV_used::Any
end


function normalize_log_post(L)
    po = zeros(length(L))
    for (i, l) in enumerate(L)
        for ll in L
            po[i] = po[i] + exp(ll - l)
        end
    end
    return 1 ./ po
end

function E_step(X, density)
    all_packages = Vector{}()
    E_step!(X, nothing, all_packages, density)

    likelihoods = Vector{Float64}(undef, length(all_packages))
    posteriors = Vector{Float64}(undef, length(all_packages))
    group_id = [pack.group_id for pack in all_packages]
    eval_posterior = function()
        for (i, pack) in enumerate(all_packages)
            likelihoods[i] = pack.likelihood()
        end

        for g in unique(group_id)
           wi = findall(group_id .== g)
           posteriors[wi] = normalize_log_post(likelihoods[wi])
        end
        return posteriors
    end

    random_posterior = function()
        for (i, pack) in enumerate(all_packages)
            likelihoods[i] = is_feasible_init(pack.LV_used) ? Float64(sample(1:10^5, 1)[1]) : 0
        end
        for (i, pack) in enumerate(all_packages)
            posteriors[i] = likelihoods[i] / sum(likelihoods[findall(group_id .== pack.group_id)])
        end
        return posteriors
    end

    return eval_posterior, random_posterior, all_packages
end

function is_feasible_init(LVs)
    for (LV, k) in LVs
        if !isnothing(LV.init_value)
            if LV.init_value != k
                return false
            end
        end
    end
    return true
end

function E_step!(X, parent_node, all_packages, density)
    independent_sets = getIndependentSets(X)
    is_first = isnothing(parent_node)
    for (i_g, G) in enumerate(independent_sets)
        if is_first
            parent_node = Node(nothing, nothing, 0, Vector{}(), i_g)
        end
        # get next best condition
        all_domains = Vector{Vector{LatentVaraible}}(undef, length(G))
        for (i, x) in enumerate(G)
            all_domains[i] = GetDependentVariable(x)
        end
        domains = [LV for LV in (reduce(vcat, all_domains))]
        domain_lengths = [length(d) for d in all_domains]
        lv_freq_map = countmap(domains)
        if maximum(domain_lengths, init=0) > maximum(values(lv_freq_map), init=0)
            next_conditions = all_domains[argmax(domain_lengths)]
        else
            next_conditions = domains
            lv_freq_map = filter(x -> x[1] in next_conditions, lv_freq_map)
            top_order = sortperm(collect(values(lv_freq_map)); rev=true)
            next_conditions = collect(keys(lv_freq_map))[top_order]
        end
        ######

        if length(next_conditions) != 0
            next_condition = next_conditions[1]
            K = typeof(next_condition.value_) == Unknown ? (1:next_condition.Z.K) : lv_v(next_condition)
            # K = 1:next_condition.Z.K
            for (i_k, k) in enumerate(K)
                lv_set(next_condition, k)
                new_node = Node((next_condition, k), parent_node, length(independent_sets), [parent_node.all_X; G], parent_node.group_id)
                E_step!(G, new_node, all_packages, density)
                lv_set(next_condition, 0)
            end
        else
            g = G[1]
            params_used = index_to_parameters(rawCallDomain(g.T), density.parameters)
            new_node = Node(nothing, parent_node, length(independent_sets), [parent_node.all_X; G], parent_node.group_id)
            pis_used = Vector{}()
            get_pis_used!(new_node,pis_used, 1)
            pre_sets = Vector{}()
            get_needed_sets!(new_node, pre_sets)
            LV_used = Vector{}()
            get_LVs_used!(new_node, LV_used)
            conditional_likelihood = initialize_density_evaluation(new_node.all_X, density)
            likelihood = function ()
                log_pi = 0
                for (c, k) in pre_sets
                    log_pi = log_pi + safe_log(c.Z.Pi[k])
                end
                ll = log_pi + conditional_likelihood()
                return ll
            end
            push!(all_packages, Package(likelihood, pis_used, params_used, new_node.group_id, g, LV_used))
        end
    end
end


function get_pis_used!(node::Node, collection, factor)
    condition_used = node.condition
    if !isnothing(condition_used)
        push!(collection, ((condition_used[1].Z,condition_used[2]), factor))
    end
    if !isnothing(node.parent)
        get_pis_used!(node.parent, collection, factor*node.correction_factor)
    end
end

function get_LVs_used!(node::Node, collection)
    condition_used = node.condition
    if !isnothing(condition_used)
        push!(collection, (condition_used[1], condition_used[2]))
    end
    if !isnothing(node.parent)
        get_LVs_used!(node.parent, collection)
    end
end

function get_needed_sets!(node::Node, collection)
    condition_used = node.condition
    if !isnothing(condition_used)
        push!(collection, condition_used)
    end
    if !isnothing(node.parent)
        get_needed_sets!(node.parent, collection)
    end
end

prod_foldl(G) = foldl((a, g) -> a + g(), G; init=0)
function sum_foldl(G)
    vv = Vector{Float64}(undef, length(G))
    function ()
        for (i, g) in enumerate(G)
            vv[i] = g()
        end
        mm = maximum(vv)
        return mm + safe_log(sum(exp.(vv .- mm)))
    end
end
safe_log(x) = log(max(x, eps(Float64)))
# safe_log(x) = log(x)
function initialize_density_evaluation(X::Vector{Observation}, density)
    independent_sets = getIndependentSets(X)
    mult_list = Vector{}()
    for G in independent_sets
        all_domains = Vector{Vector{LatentVaraible}}(undef, length(G))
        for (i, x) in enumerate(G)
            all_domains[i] = GetDependentVariable(x)
        end
        domains = [LV for LV in (reduce(vcat, all_domains))]
        domain_lengths = [length(d) for d in all_domains]
        lv_freq_map = countmap(domains)
        if maximum(domain_lengths, init=0) > maximum(values(lv_freq_map), init=0)
            next_conditions = all_domains[argmax(domain_lengths)]
        else
            next_conditions = domains
            lv_freq_map = filter(x -> x[1] in next_conditions, lv_freq_map)
            top_order = sortperm(collect(values(lv_freq_map)); rev=true)
            next_conditions = collect(keys(lv_freq_map))[top_order]
        end

        if length(next_conditions) != 0
            next_condition = next_conditions[1]
            K = typeof(next_condition.value_) == Unknown ? (1:next_condition.Z.K) : lv_v(next_condition)
            # K = 1:next_condition.Z.K
            sum_list = Vector{}(undef, length(K))
            for (i_k, k) in enumerate(K)
                lv_set(next_condition, k)
                lik_new = initialize_density_evaluation(G, density)
                # pi_c = () -> (typeof(next_condition.value_) == Unknown ? next_condition.Z.Pi[k] : lv_v(next_condition) == k)
                pi_c = () -> (typeof(next_condition.value_) == Unknown ? next_condition.Z.Pi[k] : next_condition.Z.Pi[k])
                sum_list[i_k] = (() -> (safe_log(pi_c()) + lik_new()))
                lv_set(next_condition, 0)
            end
            ss = sum_foldl(sum_list)
            push!(mult_list, (() -> ss()))
        else
            for g in G
                ma = rawCallDomain(g.T)
                mm = index_to_parameters(ma, density.parameters)
                push!(mult_list, (() -> (density.evaluate(g, mm))))
            end
        end
    end
    return () -> prod_foldl(mult_list)
end


function M_step_init(packages, base::Parsa_Base)
    mappings = Dict{}()
    for ke in base.parameter_order
        for (ke2, params) in base.parameters[ke].parameter_map
            if !params.is_const
                mappings[params] = Vector{}()
                for (i, pack) in enumerate(packages)
                    if params in values(pack.params_used)
                        push!(mappings[params], i)
                    end
                end
            end
        end
    end
    return function (tau, base)
        for (param, va) in mappings
            x::Vector{Observation} = [pack.X for pack in packages[va]]
            pr::Vector{Real} = tau[va]
            map::Vector{Dict{Any,Any}} = [pack.params_used for pack in packages[va]]
            param.run_update(x, pr, map, base.log_pdf)
        end
    end
end

function Pi_init(packages, base::Parsa_Base)
    pis_used = [[p[1] for p in pack.pi_used] for pack in packages]
    X = unique([pack.X for pack in packages])
    domains = unique(reduce(vcat, [GetDependentVariable(x) for x in X]))
    all_Z = unique([LV.Z for LV in domains])
    mappings = Dict{}()
    for (i, Z_cat) in enumerate(all_Z)
        mappings[Z_cat] = Dict{}()
        for k in 1:Z_cat.K
            mappings[Z_cat][k] = Vector{}()
             for i in eachindex(packages)
                if (Z_cat,k) in pis_used[i]
                    correction = [p[2] for p in packages[i].pi_used if p[1] == (Z_cat,k)][1]
                    push!(mappings[Z_cat][k], (i, correction))
                end
            end
        end
    end
    return function (tau)
        for (i, Z_cat) in enumerate(all_Z)
            Pi_new = zeros(Z_cat.K)
            for k in 1:Z_cat.K
                z_map = mappings[Z_cat][k]
                is = [m[1] for m in z_map]
                co = [m[2] for m in z_map]
                Pi_new[k] = sum(tau[is] ./ co; init=0)
            end
            Pi_new = Pi_new ./ sum(Pi_new)
            if !Z_cat.constant
                Z_cat.Pi = Pi_new
            end
        end
    end
end


function LMEM(X::Set{Observation}, base::Parsa_Base; eps=10^-6,
                                                        max_steps = 10000,
                                                        n_init = 30,
                                                        init_eps=10^-3,
                                                        verbose=true)
    X = collect(X)
    prime_X.(X)
    posteriors, random_posterior, packages = E_step(X, base)
    m = M_step_init(packages, base)
    p = Pi_init(packages, base)
    likelihood = initialize_density_evaluation(X, base)
    param_reset = set_original_parameters(base)
    pi_reset = reset_Pi(X)


    init_likelihoods = [[] for _ in 1:n_init]
    best_likelihood = -Inf
    best_tau = nothing
    best_prams = nothing
    best_Pi = nothing
    for i_init in 1:n_init
        param_reset()
        pi_reset()
        tau_wild = random_posterior()
        p(tau_wild)
        m(tau_wild, base)
        lik_old = -Inf
        lik_new = likelihood()
        for i_wild in 1:max_steps
            tau_wild = posteriors()
            p(tau_wild)
            m(tau_wild, base)
            lik_old = lik_new
            lik_new = likelihood()
            if (lik_new < lik_old && i_wild >= 3)
                error("init error... decreasing likelihood with rel diff of " * string(abs(lik_new - lik_old) / abs(lik_old)) * "on step " * string(i_wild))
            end
            if i_wild > 0
                push!(init_likelihoods[i_init], (lik_new))
                verbose ? plotit(init_likelihoods, Vector{}()) : nothing
            end
            if abs(lik_new - lik_old) / abs(lik_old) < init_eps
                break
            end
        end
        if lik_new > best_likelihood
            best_likelihood = lik_new
            best_tau = tau_wild
            best_prams = save_parameters(base)
            best_Pi = save_Pi(X)
        end
    end

    best_Pi()
    best_prams()
    for _ in 1:2
        tau_wild = posteriors()
        p(tau_wild)
        m(tau_wild, base)
    end
    lik_old = -Inf
    lik_new = likelihood()
    all_likelihoods::Vector{Real} = [lik_new]
    all_steps::Vector{Real} = [1]
    i = 1
    while ((abs(lik_new - lik_old) / abs(lik_new)) > eps && max_steps > 0)
        max_steps = max_steps - 1
        tau = posteriors()
        p(tau)
        m(tau, base)
        lik_old = lik_new
        lik_new = likelihood()
        all_likelihoods = [all_likelihoods; Float64(lik_new)]
        all_steps = [all_steps; i]
        verbose ? plotit(init_likelihoods, all_likelihoods) : nothing
        i = i + 1
    end
    (log_likelihood=likelihood, n=length(X))
end


function plotit(lines, final_lines)
    println("\33[H")
    print("\33c\e[3J")
    terminal = displaysize(stdout)
    lll = [l for l in [reduce(vcat, lines); final_lines] if l != 0]
    ymax = maximum(lll)
    ymin = minimum(lll)
    xmin = 0
    xmax_init = argmax([maximum(l, init=1) for l in lines]) #maximum([length(ll) for ll in lines])
    xmax_init = length(lines[xmax_init])
    xmax = xmax_init + length(final_lines)
    plt = UnicodePlots.lineplot([0], [0]; ylim=(ymin, ymax), xlim=(xmin, xmax), xlabel="steps", ylabel="log-likelihood", height=Int(round(terminal[1] / 2)), width=Int(round(terminal[2] / 2)), color=:red)
    for i in 1:size(lines)[1]
        n_l = lines[i] #[l for l in lines[i, :] if l != 0]
        UnicodePlots.lineplot!(plt, 1:length(n_l), n_l, color=:blue)
    end
    if length(final_lines) > 0
        n_l = final_lines
        UnicodePlots.lineplot!(plt, xmax_init:(length(n_l)+xmax_init-1), n_l)
    end
    display(plt)
    sleep(0.001)
end


function set_original_parameters(base::Parsa_Base)
    for ke in base.parameter_order
        for (ke2, param) in base.parameters[ke].parameter_map
            param.value_original = deepcopy(param.value.value)
        end
    end

    return function ()
        for ke in base.parameter_order
            for (ke2, param) in base.parameters[ke].parameter_map
                param.value.value = param.value_original
            end
        end
    end
end

function reset_Pi(X::Vector{Observation})
    domains = unique(reduce(vcat, [GetDependentVariable(x) for x in X]))
    all_Z = unique([LV.Z for LV in domains])
    return function ()
        for Z in all_Z
            k = length(Z.Pi)
            if !Z.constant
                Z.Pi = zeros(k) .+ 1 / k
            end
        end
    end
end

function save_parameters(base::Parsa_Base)
    all_params = Vector{}()
    for ke in base.parameter_order
        for (_, param) in base.parameters[ke].parameter_map
            if !param.is_const
                push!(all_params, deepcopy(param.value.value))
            end
        end
    end

    return function ()
        i = 1
        for ke in base.parameter_order
            for (_, param) in base.parameters[ke].parameter_map
                if !param.is_const
                    param.value.value = all_params[i]
                    i = i + 1
                end
            end
        end
    end
end

function save_Pi(X::Vector{Observation})
    domains = unique(reduce(vcat, [GetDependentVariable(x) for x in X]))
    all_Z = unique([LV.Z for LV in domains])
    vals = Vector{}()
    for Z in all_Z
        push!(vals, deepcopy(Z.Pi))
    end
    return function ()
        for (Z, v) in zip(all_Z, vals)
            Z.Pi = v
        end
    end
end

function posterior_initialize(conditions, X::Vector{Observation}, density::Parsa_Base)
    prime_X.(X)
    domains = conditions
    all_domains = [GetDependentVariable(x) for x in X]
    X_sub = [(x, xf) for (x, xf) in zip(X, all_domains) if !isdisjoint(domains, xf)]
    X_full::Vector{Observation} = []
    for (_, xf) in X_sub
        for LV in xf
            if !lv_isKnown(LV)
                X_full = [X_full; collect(values(LV.dependent_X))]
            end
        end
    end
    X_sub = unique(X_full)
    if length(X_sub) == 0
        X_sub = [X[1]]
    end
    domain_map = Dict([x => unique(GetDependentVariable(x)) for x in X_sub])
    map_collector = OrderedDict()
    independent_map = Dict()
    tau = Vector{}()
    Pi = Vector{}()
    posterior_initialize!(domains, X_sub, density, tau, Pi, domains)
    tau_l = length(tau)
    vv = Vector{Float64}(undef, tau_l)
    function ()
        for i in 1:tau_l
            vv[i] = tau[i]()
        end
        tau_vec = normalize_log_post(vv)
        Pi_val = [[y for (_, y) in pi_i] for pi_i in Pi]
        PV = Dict([length(pp) == 1 ? pp[1] => vv : pp => vv for (pp, vv) in zip(Pi_val, tau_vec)])
        return (max=max_posterior(PV), probability=PV)
    end
end

function posterior_initialize!(domains, X::Vector{Observation}, density::Parsa_Base, tau_chain, Pi_used, all_domains)
    condition = domains[1]
    K = typeof(condition.value_) == Unknown ? (1:condition.Z.K) : lv_v(condition)
    for k in K
        lv_set(condition, k)
        if length(domains[2:end]) >= 1
            posterior_initialize(domains[2:end], X, density, tau_chain, Pi_used, all_domains)
        else
            tau = initialize_density_evaluation(X, density)
            push!(Pi_used, Tuple([(d.Z, lv_v(d)) for d in all_domains]))
            current_state = [lv_v(LV) for LV in all_domains]
            Pi_val = () -> prod([d.Z.Pi[current_state[i]] for (i, d) in enumerate(all_domains)])
            push!(tau_chain, () -> safe_log(Pi_val()) + tau())
        end
        lv_set(condition, 0)

    end
end

function max_posterior(post)
    top_prob = 0
    cond = nothing
    for p in post
        if p[2] > top_prob
            top_prob = p[2]
            cond = p[1]
        end
    end
    if length(cond) == 1
        return cond[1]
    end
    return (cond)
end
