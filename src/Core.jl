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

function ordered_countmap(v::Vector{LatentVaraible})
    counts = OrderedDict{LatentVaraible, Int}()
    for x in v
        counts[x] = get(counts, x, 0) + 1
    end
    return counts
end

function getIndependentSets(X::Vector{Observation})
    domains = Dict([x => OrderedSet([lv for lv in GetDependentVariable(x) if !lv_isKnown(lv)]) for x in X])
    Xset = OrderedSet(X)
    domainSet = OrderedSet{LatentVaraible}()
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

@inline function normalize_log_post!(posteriors, likelihoods, wi)
    @inbounds for i in wi
        l = likelihoods[i]
        s = 0.0
        for j in wi
            s += exp(likelihoods[j] - l)
        end
        posteriors[i] = 1.0 / s
    end
end

struct DensityCache
    evals::Vector{Tuple{Observation, Dict{Symbol, Any}}}
    index::Dict{Tuple{Observation, Dict{Symbol, Any}}, Int}
    values::Vector{Float64}
    leaf_closures::Dict{Vector{Int}, Function}
    sum_evals::Vector{Function}
    sum_index::Dict{Tuple{CategoricalZ, Vector{Int}, Vector{Function}}, Int}
    sum_values::Vector{Float64}
    sum_closures::Vector{Function}
end

DensityCache() = DensityCache(
    Vector{Tuple{Observation, Dict{Symbol, Any}}}(),
    Dict{Tuple{Observation, Dict{Symbol, Any}}, Int}(),
    Vector{Float64}(),
    Dict{Vector{Int}, Function}(),
    Vector{Function}(),
    Dict{Tuple{CategoricalZ, Vector{Int}, Vector{Function}}, Int}(),
    Vector{Float64}(),
    Vector{Function}()
)

function register_leaf!(cache::DensityCache, g::Observation, mm::Dict{Symbol, Any})
    return get!(cache.index, (g, mm)) do
        push!(cache.evals, (g, mm))
        push!(cache.values, 0.0)
        return length(cache.evals)
    end
end

function register_sum!(cache::DensityCache, Z_cond::CategoricalZ, K_vec::Vector{Int}, children::Vector{Function}, sum_list)
    idx = get!(cache.sum_index, (Z_cond, K_vec, children)) do
        fn = length(sum_list) == 1 ? sum_list[1] : sum_foldl(sum_list)
        push!(cache.sum_evals, fn)
        push!(cache.sum_values, 0.0)
        new_idx = length(cache.sum_evals)
        svals = cache.sum_values
        push!(cache.sum_closures, @inline () -> svals[new_idx])
        return new_idx
    end
    return cache.sum_closures[idx]
end

function E_step(X, density)
    all_packages = Vector{Package}()
    cache = DensityCache()
    E_step!(X, nothing, all_packages, density, cache)

    likelihoods = Vector{Float64}(undef, length(all_packages))
    posteriors = Vector{Float64}(undef, length(all_packages))
    group_id = [pack.group_id for pack in all_packages]
    group_indices = [findall(group_id .== g) for g in unique(group_id)]
    eval_posterior = function()
        @inbounds for i in eachindex(cache.evals)
            g, mm = cache.evals[i]
            cache.values[i] = density.evaluate(g, mm)::Float64
        end
        @inbounds for i in eachindex(cache.sum_evals)
            cache.sum_values[i] = cache.sum_evals[i]()::Float64
        end

        @inbounds for (i, pack) in enumerate(all_packages)
            if i > 1 && pack.likelihood === all_packages[i - 1].likelihood
                likelihoods[i] = likelihoods[i - 1]
            else
                likelihoods[i] = pack.likelihood()::Float64
            end
        end

        @inbounds for wi in group_indices
            normalize_log_post!(posteriors, likelihoods, wi)
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

function E_step!(X, parent_node, all_packages, density, cache=nothing)
    independent_sets = getIndependentSets(X)
    is_first = isnothing(parent_node)
    shared_likelihood = nothing
    for (i_g, G) in enumerate(independent_sets)
        if is_first
            parent_node = Node(nothing, nothing, 0, G, i_g)
        end
        # get next best condition
        all_domains = Vector{Vector{LatentVaraible}}(undef, length(G))
        for (i, x) in enumerate(G)
            all_domains[i] = GetDependentVariable(x)
        end
        domains = reduce(vcat, all_domains)
        domain_lengths = [length(d) for d in all_domains]
        lv_freq_map = ordered_countmap(domains)
        if maximum(domain_lengths, init=0) > maximum(values(lv_freq_map), init=0)
            next_condition = all_domains[argmax(domain_lengths)][1]
            has_next = true
        elseif !isempty(lv_freq_map)
            next_condition = argmax(lv_freq_map)
            has_next = true
        else
            has_next = false
        end
        ######

        if has_next
            K = typeof(next_condition.value_) == Unknown ? (1:next_condition.Z.K) : lv_v(next_condition)
            # K = 1:next_condition.Z.K
            for (i_k, k) in enumerate(K)
                lv_set(next_condition, k)
                new_node = Node((next_condition, k), parent_node, length(independent_sets), parent_node.all_X, parent_node.group_id)
                E_step!(G, new_node, all_packages, density, cache)
                lv_set(next_condition, 0)
            end
        else
            g = G[1]
            params_used = index_to_parameters(rawCallDomain(g.T), density.parameters)
            new_node = Node(nothing, parent_node, length(independent_sets), parent_node.all_X, parent_node.group_id)
            pis_used = Vector{}()
            get_pis_used!(new_node,pis_used, 1)
            LV_used = Vector{}()
            get_LVs_used!(new_node, LV_used)
            if is_first || isnothing(shared_likelihood)
                pre_sets = Vector{Tuple{LatentVaraible, Int}}()
                get_needed_sets!(new_node, pre_sets)
                conditional_likelihood = initialize_density_evaluation(new_node.all_X, density, cache)
                shared_likelihood = @inline function ()
                    log_pi = 0.0
                    @inbounds for (c, k) in pre_sets
                        log_pi = log_pi + safe_log(Float64(c.Z.Pi[k]))
                    end
                    ll = log_pi + conditional_likelihood()::Float64
                    return ll
                end
            end
            push!(all_packages, Package(shared_likelihood, pis_used, params_used, new_node.group_id, g, LV_used))
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

@inline function prod_foldl(G)
    s = 0.0
    @inbounds for g in G
        s += g()::Float64
    end
    return s
end
function sum_foldl(G)
    vv = Vector{Float64}(undef, length(G))
    @inline function ()
        mm = -Inf
        @inbounds for i in eachindex(G)
            v = G[i]()::Float64
            vv[i] = v
            if v > mm
                mm = v
            end
        end
        s = 0.0
        @inbounds for v in vv
            s += exp(v - mm)
        end
        return mm + safe_log(s)
    end
end
@inline safe_log(x) = log(max(x, eps(Float64)))
# safe_log(x) = log(x)
function initialize_density_evaluation(X::Vector{Observation}, density, cache=nothing)
    independent_sets = getIndependentSets(X)
    mult_list = Vector{}()
    leaf_list = Vector{Tuple{Observation, Dict{Symbol, Any}}}()
    for G in independent_sets
        all_domains = Vector{Vector{LatentVaraible}}(undef, length(G))
        for (i, x) in enumerate(G)
            all_domains[i] = GetDependentVariable(x)
        end
        domains = reduce(vcat, all_domains)
        domain_lengths = [length(d) for d in all_domains]
        lv_freq_map = ordered_countmap(domains)
        if maximum(domain_lengths, init=0) > maximum(values(lv_freq_map), init=0)
            next_condition = all_domains[argmax(domain_lengths)][1]
            has_next = true
        elseif !isempty(lv_freq_map)
            next_condition = argmax(lv_freq_map)
            has_next = true
        else
            has_next = false
        end

        if has_next
            K = typeof(next_condition.value_) == Unknown ? (1:next_condition.Z.K) : lv_v(next_condition)
            # K = 1:next_condition.Z.K
            sum_list = Vector{}(undef, length(K))
            children = Vector{Function}(undef, length(K))
            Z_cond = next_condition.Z
            for (i_k, k) in enumerate(K)
                lv_set(next_condition, k)
                lik_new = initialize_density_evaluation(G, density, cache)
                children[i_k] = lik_new
                # pi_c = () -> (typeof(next_condition.value_) == Unknown ? next_condition.Z.Pi[k] : lv_v(next_condition) == k)
                sum_list[i_k] = @inline () -> (safe_log(Float64(Z_cond.Pi[k])) + lik_new()::Float64)
                lv_set(next_condition, 0)
            end
            if isnothing(cache)
                ss = length(sum_list) == 1 ? sum_list[1] : sum_foldl(sum_list)
            else
                ss = register_sum!(cache, Z_cond, K isa Int ? [K] : collect(K), children, sum_list)
            end
            push!(mult_list, ss)
        else
            for g in G
                ma = rawCallDomain(g.T)
                mm = index_to_parameters(ma, density.parameters)
                push!(leaf_list, (g, mm))
            end
        end
    end
    if !isempty(leaf_list)
        if isnothing(cache)
            if length(leaf_list) == 1
                g1, mm1 = leaf_list[1]
                push!(mult_list, @inline () -> (density.evaluate(g1, mm1)))
            else
                push!(mult_list, @inline function ()
                    s = 0.0
                    @inbounds for (g, mm) in leaf_list
                        s += density.evaluate(g, mm)::Float64
                    end
                    return s
                end)
            end
        else
            leaf_indices = [register_leaf!(cache, g, mm) for (g, mm) in leaf_list]
            leaf_fn = get!(cache.leaf_closures, leaf_indices) do
                vals = cache.values
                if length(leaf_indices) == 1
                    idx1 = leaf_indices[1]
                    return @inline () -> vals[idx1]
                else
                    return @inline function ()
                        s = 0.0
                        @inbounds for idx in leaf_indices
                            s += vals[idx]
                        end
                        return s
                    end
                end
            end
            push!(mult_list, leaf_fn)
        end
    end
    return length(mult_list) == 1 ? mult_list[1] : (@inline () -> prod_foldl(mult_list))
end


function M_step_init(packages, base::Parsa_Base)
    mappings = OrderedDict{Parameter, Vector{Int}}()
    for ke in base.parameter_order
        for (ke2, params) in base.parameters[ke].parameter_map
            if !params.is_const
                mappings[params] = Vector{Int}()
                for (i, pack) in enumerate(packages)
                    if params in values(pack.params_used)
                        push!(mappings[params], i)
                    end
                end
            end
        end
    end
    m_entries = [
        (param, va, Observation[pack.X for pack in packages[va]], Vector{Float64}(undef, length(va)), Dict{Symbol,Any}[pack.params_used for pack in packages[va]])
        for (param, va) in mappings
    ]
    return function (tau, base)
        for (param, va, x, pr, map) in m_entries
            @inbounds for (j, idx) in enumerate(va)
                pr[j] = tau[idx]
            end
            param.run_update(x, pr, map, base.log_pdf)
        end
    end
end

function Pi_init(packages, base::Parsa_Base)
    pis_used = [[p[1] for p in pack.pi_used] for pack in packages]
    X = unique([pack.X for pack in packages])
    domains = unique(reduce(vcat, [GetDependentVariable(x) for x in X]))
    all_Z = unique([LV.Z for LV in domains])
    mappings = Dict{CategoricalZ, Vector{Vector{Tuple{Int, Int}}}}()
    for (i, Z_cat) in enumerate(all_Z)
        mappings[Z_cat] = Vector{Vector{Tuple{Int, Int}}}(undef, Z_cat.K)
        for k in 1:Z_cat.K
            mappings[Z_cat][k] = Vector{Tuple{Int, Int}}()
            for i in eachindex(packages)
                if (Z_cat,k) in pis_used[i]
                    correction = [p[2] for p in packages[i].pi_used if p[1] == (Z_cat,k)][1]
                    push!(mappings[Z_cat][k], (i, correction))
                end
            end
        end
    end
    return function (tau)
        for Z_cat in all_Z
            Z_cat.constant && continue
            Pi_new = Z_cat.Pi
            total = 0.0
            z_maps = mappings[Z_cat]
            @inbounds for k in 1:Z_cat.K
                s = 0.0
                for (idx, co) in z_maps[k]
                    s += tau[idx] / co
                end
                Pi_new[k] = s
                total += s
            end
            @inbounds for k in 1:Z_cat.K
                Pi_new[k] /= total
            end
        end
    end
end


function LMEM(X::AbstractSet{Observation}, base::Parsa_Base; eps=10^-6,
                                                        max_steps = 10000,
                                                        n_init = 30,
                                                        init_eps=10^-3,
                                                        verbose=true)
    X = collect(X)
    tui = EM_TUI(length(X), n_init, eps, init_eps, verbose)
    verbose && render_build!(tui, 1, "Priming observations")
    prime_X.(X)
    verbose && render_build!(tui, 2, "Building E-step graph")
    posteriors, random_posterior, packages = E_step(X, base)
    verbose && render_build!(tui, 3, "Mapping M-step & π", length(packages))
    m = M_step_init(packages, base)
    p = Pi_init(packages, base)
    verbose && render_build!(tui, 4, "Building likelihood tree", length(packages))
    likelihood = initialize_density_evaluation(X, base)
    param_reset = set_original_parameters(base)
    pi_reset = reset_Pi(X)
    verbose && finish_build!(tui, length(packages))


    init_likelihoods = [[] for _ in 1:n_init]
    best_likelihood = -Inf
    best_tau = nothing
    best_prams = nothing
    best_Pi = nothing
    for i_init in 1:n_init
        tui.had_warning = false
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
            tui.had_warning && break
            lik_old = lik_new
            lik_new = likelihood()
            if (lik_new < lik_old && i_wild >= 3)
                error("init error... decreasing likelihood with rel diff of " * string(abs(lik_new - lik_old) / abs(lik_old)) * "on step " * string(i_wild))
            end
            if i_wild > 0
                push!(init_likelihoods[i_init], (lik_new))
                verbose ? plotit(init_likelihoods, Vector{}(); tui=tui, phase=:init, i_init=i_init, step=i_wild, max_steps=max_steps, rel_diff=abs(lik_new - lik_old) / abs(lik_old), target_eps=init_eps) : nothing
            end
            if abs(lik_new - lik_old) / abs(lik_old) < init_eps
                break
            end
        end
        if !tui.had_warning && lik_new > best_likelihood
            best_likelihood = lik_new
            best_tau = tau_wild
            best_prams = save_parameters(base)
            best_Pi = save_Pi(X)
        end
    end
    if isnothing(best_prams)
        best_prams = save_parameters(base)
        best_Pi = save_Pi(X)
    end

    tui.in_final = true
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
        verbose ? plotit(init_likelihoods, all_likelihoods; tui=tui, phase=:final, i_init=n_init, step=i, max_steps=max_steps+i, rel_diff=abs(lik_new - lik_old) / abs(lik_new), target_eps=eps) : nothing
        i = i + 1
    end
    verbose ? plotit(init_likelihoods, all_likelihoods; tui=tui, phase=:done, i_init=n_init, step=max(1, i-1), max_steps=max_steps+i-1, rel_diff=abs(lik_new - lik_old) / abs(lik_new), target_eps=eps, force=true) : nothing
    restore_logger!(tui)
    (log_likelihood=likelihood, n=length(X))
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
                Z.Pi = (zeros(k) .+ 1) / k
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

function load_iris_dataset()
    path = joinpath(@__DIR__, "..", "examples/datasets")
    file = joinpath(path, "Iris.csv")
    iris = CSV.read(file, DataFrame)
    iris_matrix = Matrix(iris[:, 2:5])
    class_string = vec(iris[:, 6])
    mapping = Dict(val => i for (i, val) in enumerate(unique(class_string)))
    class = [mapping[val] for val in class_string]
    return iris_matrix, class
end
