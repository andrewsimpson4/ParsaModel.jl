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


const _SUNSET_BAR = ["#FF5E62", "#FF7560", "#FF8C5E", "#FFA35C", "#FFBA60", "#FFD166"]
const _OCEAN_BAR  = ["#FF7E5F", "#E5748F", "#C471ED", "#7A8CE2", "#2CA5E0", "#00B4DB"]

mutable struct EM_TUI
    n_obs::Int
    n_init::Int
    eps::Float64
    init_eps::Float64
    n_packages::Int
    build_stage::Int
    build_msg::String
    build_start_ns::UInt64
    build_elapsed::Float64
    em_start_ns::UInt64
    last_render_ns::UInt64
    total_iters::Int
    last_lines::Int
    init_warnings::OrderedDict{String, Int}
    final_warnings::OrderedDict{String, Int}
    prev_logger::Any
    verbose::Bool
    had_warning::Bool
    in_final::Bool
end

struct TUIWarningLogger <: Base.CoreLogging.AbstractLogger
    tui::EM_TUI
end

Base.CoreLogging.min_enabled_level(::TUIWarningLogger) = Base.CoreLogging.Warn
Base.CoreLogging.shouldlog(::TUIWarningLogger, level, _module, group, id) = level >= Base.CoreLogging.Warn
Base.CoreLogging.catch_exceptions(::TUIWarningLogger) = true

function Base.CoreLogging.handle_message(logger::TUIWarningLogger, level, message, _module, group, id, file, line; kwargs...)
    logger.tui.had_warning = true
    if logger.tui.verbose
        msg = strip(string(message))
        if !isempty(msg)
            target = logger.tui.in_final ? logger.tui.final_warnings : logger.tui.init_warnings
            target[msg] = get(target, msg, 0) + 1
        end
    elseif !isnothing(logger.tui.prev_logger)
        Base.CoreLogging.handle_message(logger.tui.prev_logger, level, message, _module, group, id, file, line; kwargs...)
    end
    return nothing
end

function EM_TUI(n_obs::Int, n_init::Int, eps::Real, init_eps::Real, verbose::Bool=true)
    now_ns = time_ns()
    tui = EM_TUI(n_obs, n_init, Float64(eps), Float64(init_eps), 0, 0, "Initializing...", now_ns, 0.0, now_ns, UInt64(0), 0, 0, OrderedDict{String, Int}(), OrderedDict{String, Int}(), nothing, verbose, false, false)
    tui.prev_logger = Base.CoreLogging.global_logger(TUIWarningLogger(tui))
    return tui
end

function restore_logger!(tui::EM_TUI)
    if !isnothing(tui.prev_logger)
        Base.CoreLogging.global_logger(tui.prev_logger)
        tui.prev_logger = nothing
    end
end

function _tui_bar(frac::Real, width::Int=20; palette::Vector{String}=_SUNSET_BAR)
    f = clamp(Float64(frac), 0.0, 1.0)
    filled = Int(round(f * width))
    empty = width - filled
    buf = IOBuffer()
    n_col = length(palette)
    for i in 1:filled
        c_idx = clamp(Int(ceil((i / max(1, width)) * n_col)), 1, n_col)
        col = palette[c_idx]
        print(buf, "{$(col)}█{/$(col)}")
    end
    pct = lpad(string(Int(round(f * 100))), 3) * "%"
    print(buf, "{dim}", "░"^empty, "{/dim} ", pct)
    return String(take!(buf))
end

function _render_frame!(tui::Union{Nothing, EM_TUI}, dashboard, total_w::Int)
    if !isnothing(tui) && (!isempty(tui.init_warnings) || !isempty(tui.final_warnings))
        warn_lines = String[]
        for (msg, cnt) in tui.init_warnings
            suffix = cnt > 1 ? " {dim}(×$(cnt)){/dim}" : ""
            push!(warn_lines, "  {#FFB347}⚠ [Init] {/#FFB347} $(msg)$(suffix)")
        end
        for (msg, cnt) in tui.final_warnings
            suffix = cnt > 1 ? " {dim}(×$(cnt)){/dim}" : ""
            push!(warn_lines, "  {#FF4D6D}⚠ [Final]{/#FF4D6D} $(msg)$(suffix)")
        end
        w_style = isempty(tui.final_warnings) ? "#FFA647" : "#FF4D6D"
        warn_panel = Term.Panel(join(warn_lines, "\n"); title="Warnings", style=w_style, width=total_w)
        dashboard = dashboard / warn_panel
    end
    frame_str = string(dashboard)
    if !isnothing(tui) && tui.last_lines > 0
        print(stdout, "\e[$(tui.last_lines)A\r\e[J", frame_str, "\n")
    else
        print(stdout, frame_str, "\n")
    end
    if !isnothing(tui)
        tui.last_lines = count(==('\n'), frame_str) + 1
    end
    flush(stdout)
end

function render_build!(tui::EM_TUI, stage::Int, msg::String, n_pkgs::Int=0)
    tui.build_stage = stage
    tui.build_msg = msg
    if n_pkgs > 0
        tui.n_packages = n_pkgs
    end
    tui.build_elapsed = (time_ns() - tui.build_start_ns) / 1e9
    terminal = displaysize(stdout)
    total_w = clamp(terminal[2] - 2, 76, 120)
    right_w = 34
    left_w = total_w - right_w
    panel_h = clamp(Int(round(terminal[1] * 0.55)), 14, 24)

    build_frac = (stage - 0.5) / 4.0
    b_bar = _tui_bar(build_frac, 20; palette=_SUNSET_BAR)
    e_bar = _tui_bar(0.0, 20; palette=_OCEAN_BAR)
    top_text = "  {bold #FFD166}Build {/bold #FFD166} $(b_bar)  {dim}[$(stage)/4: $(msg)]{/dim}\n" *
               "  {bold #48CAE4}EM Fit{/bold #48CAE4} $(e_bar)  {dim}[Waiting for compilation]{/dim}"
    top_panel = Term.Panel(top_text; title="🌅 ParsaModel EM!", style="#FF7E5F", width=total_w)

    steps = [
        (1, "Prime Observations (X)"),
        (2, "E-Step Conditioning Graph"),
        (3, "M-Step & π Parameter Mappings"),
        (4, "Log-Likelihood Evaluation Tree")
    ]
    checklist = IOBuffer()
    println(checklist, "")
    for (idx, label) in steps
        if idx < stage
            println(checklist, "  {#FFD166}✓{/#FFD166} $(label)")
        elseif idx == stage
            println(checklist, "  {#FF7E5F}⟳{/#FF7E5F} {bold #FFD166}$(label){/bold #FFD166}")
        else
            println(checklist, "  {dim}• $(label){/dim}")
        end
    end
    println(checklist, "\n  {dim}Elapsed build time: $(round(tui.build_elapsed, digits=2))s{/dim}")
    left_panel = Term.Panel(String(take!(checklist)); title="Structure Compilation", style="#FFA647", width=left_w, height=panel_h)

    info_text = join([
        "• {bold #90E0EF}Status:{/bold #90E0EF}     {#FFB347}Compiling{/#FFB347}",
        "• {bold #90E0EF}Stage:{/bold #90E0EF}      $(stage) / 4",
        "• {bold #90E0EF}Obs (N):{/bold #90E0EF}    $(tui.n_obs)",
        "• {bold #90E0EF}Packages:{/bold #90E0EF}   $(tui.n_packages > 0 ? string(tui.n_packages) : "...")",
        "• {bold #90E0EF}Restarts:{/bold #90E0EF}   0 / $(tui.n_init)",
        "• {bold #90E0EF}Init ε:{/bold #90E0EF}     $(round(tui.init_eps, sigdigits=2))",
        "• {bold #90E0EF}Final ε:{/bold #90E0EF}    $(round(tui.eps, sigdigits=2))",
        "• {bold #90E0EF}Elapsed:{/bold #90E0EF}    $(round(tui.build_elapsed, digits=2))s"
    ], "\n")
    right_panel = Term.Panel(info_text; title="Model & Convergence", style="#0077B6", width=right_w, height=panel_h)

    _render_frame!(tui, top_panel / (left_panel * right_panel), total_w)
end

function finish_build!(tui::EM_TUI, n_pkgs::Int)
    tui.n_packages = n_pkgs
    tui.build_stage = 4
    tui.build_elapsed = (time_ns() - tui.build_start_ns) / 1e9
    tui.em_start_ns = time_ns()
end

function plotit(lines, final_lines;
                tui::Union{Nothing, EM_TUI}=nothing,
                phase::Symbol=:init,
                i_init::Int=1,
                step::Int=1,
                max_steps::Int=10000,
                rel_diff::Real=NaN,
                target_eps::Real=1e-6,
                force::Bool=false)
    now_ns = time_ns()
    if !isnothing(tui)
        tui.total_iters += 1
        if !force && step > 1 && (now_ns - tui.last_render_ns) < 33_000_000
            return nothing
        end
        tui.last_render_ns = now_ns
    end

    terminal = displaysize(stdout)
    total_w = clamp(terminal[2] - 2, 76, 120)
    right_w = 34
    left_w = total_w - right_w
    panel_h = clamp(Int(round(terminal[1] * 0.62)), 16, 26)
    plot_w = max(16, left_w - 18)
    plot_h = max(6, panel_h - 6)

    lll = [Float64(l) for l in [reduce(vcat, lines); final_lines] if l != 0 && isfinite(l)]
    if isempty(lll)
        lll = [0.0]
    end
    ymax = maximum(lll)
    ymin = minimum(lll)
    if ymin == ymax
        ymin -= 1.0
        ymax += 1.0
    end
    xmin = 0

    best_init_idx = 1
    best_init_ll = -Inf
    for (idx, l) in enumerate(lines)
        if !isempty(l)
            m_val = maximum(l)
            if m_val > best_init_ll
                best_init_ll = m_val
                best_init_idx = idx
            end
        end
    end
    xmax_init = isempty(lines[best_init_idx]) ? 1 : length(lines[best_init_idx])
    max_any_init = maximum([length(l) for l in lines]; init=1)
    xmax = max(max_any_init, xmax_init + length(final_lines))

    plt = UnicodePlots.lineplot([0], [ymin];
        ylim=(round(ymin, digits=1), round(ymax, digits=1)),
        xlim=(xmin, xmax),
        xlabel="steps",
        margin=1,
        padding=0,
        height=plot_h,
        width=plot_w,
        color=39)

    for i in 1:size(lines)[1]
        n_l = lines[i]
        if !isempty(n_l) && i != best_init_idx && i != i_init
            UnicodePlots.lineplot!(plt, 1:length(n_l), Float64.(n_l), color=39)
        end
    end
    if !isempty(lines[best_init_idx])
        UnicodePlots.lineplot!(plt, 1:length(lines[best_init_idx]), Float64.(lines[best_init_idx]), color=45)
    end
    if phase == :init && i_init <= length(lines) && !isempty(lines[i_init]) && i_init != best_init_idx
        UnicodePlots.lineplot!(plt, 1:length(lines[i_init]), Float64.(lines[i_init]), color=208)
    end
    if length(final_lines) > 0
        n_l = Float64.(final_lines)
        UnicodePlots.lineplot!(plt, xmax_init:(length(n_l)+xmax_init-1), n_l, color=220)
    end

    str_plt = sprint(show, MIME("text/plain"), plt; context=:color=>true)
    left_panel = Term.Panel(str_plt; title="🌊 Log-Likelihood", style="#00B4DB", width=left_w, height=panel_h)

    curr_ll = !isempty(final_lines) ? Float64(final_lines[end]) :
              (i_init <= length(lines) && !isempty(lines[i_init]) ? Float64(lines[i_init][end]) : best_init_ll)

    n_init_total = !isnothing(tui) ? tui.n_init : length(lines)
    if phase == :init
        em_frac = (i_init - 0.5) / max(1, n_init_total) * 0.75
        em_status = "Init $(i_init)/$(n_init_total) • Step $(step)"
        phase_str = "{#FF9966}Init ($(i_init)/$(n_init_total)){/#FF9966}"
    elseif phase == :final
        init_e = !isnothing(tui) ? tui.init_eps : 1e-3
        final_e = !isnothing(tui) ? tui.eps : 1e-6
        conv_frac = (isfinite(rel_diff) && rel_diff > 0) ?
            clamp((log10(init_e) - log10(rel_diff)) / max(1e-9, log10(init_e) - log10(final_e)), 0.0, 0.99) : 0.5
        em_frac = 0.75 + 0.25 * conv_frac
        em_status = "Refining • Step $(step)"
        phase_str = "{#48CAE4}Final EM{/#48CAE4}"
    else
        em_frac = 1.0
        em_status = "Converged at step $(step)"
        phase_str = "{#FFD166}Converged ☀{/#FFD166}"
    end

    build_time_str = !isnothing(tui) ? "$(round(tui.build_elapsed, digits=2))s" : "done"
    pkgs_str = (!isnothing(tui) && tui.n_packages > 0) ? "$(tui.n_packages) pkgs" : "ready"
    b_bar = _tui_bar(1.0, 20; palette=_SUNSET_BAR)
    e_bar = _tui_bar(em_frac, 20; palette=_OCEAN_BAR)
    top_text = "  {bold #FFD166}Build {/bold #FFD166} $(b_bar)  {dim}[Done in $(build_time_str) • $(pkgs_str)]{/dim}\n" *
               "  {bold #48CAE4}EM Fit{/bold #48CAE4} $(e_bar)  {dim}[$(em_status)]{/dim}"
    top_panel = Term.Panel(top_text; title="🌅 ParsaModel EM!", style="#FF7E5F", width=total_w)

    em_elapsed = !isnothing(tui) ? (now_ns - tui.em_start_ns) / 1e9 : 0.0
    speed = ( !isnothing(tui) && em_elapsed > 0 ) ? Int(round(tui.total_iters / em_elapsed)) : 0
    rel_str = isfinite(rel_diff) ? string(round(rel_diff, sigdigits=3)) : "..."

    info_lines = [
        "• {bold #90E0EF}Phase:{/bold #90E0EF}      $(phase_str)",
        "• {bold #90E0EF}Step:{/bold #90E0EF}       $(step) / $(max_steps)",
        "• {bold #90E0EF}Best Init:{/bold #90E0EF}  #$(best_init_idx) / $(n_init_total)",
        "• {bold #90E0EF}Best LL:{/bold #90E0EF}    {#FFD166}$(round(max(best_init_ll, curr_ll), digits=4)){/#FFD166}",
        "• {bold #90E0EF}Curr LL:{/bold #90E0EF}    $(round(curr_ll, digits=4))",
        "• {bold #90E0EF}ΔLL / |LL|:{/bold #90E0EF} $(rel_str)",
        "• {bold #90E0EF}Target ε:{/bold #90E0EF}   $(round(target_eps, sigdigits=2))",
        "• {bold #90E0EF}Obs (N):{/bold #90E0EF}    $(!isnothing(tui) ? string(tui.n_obs) : "...")",
        "• {bold #90E0EF}Packages:{/bold #90E0EF}   $(!isnothing(tui) ? string(tui.n_packages) : "...")",
        "• {bold #90E0EF}Speed:{/bold #90E0EF}      $(speed) iter/s",
        "• {bold #90E0EF}Elapsed:{/bold #90E0EF}    $(round(em_elapsed, digits=2))s"
    ]
    right_panel = Term.Panel(join(info_lines, "\n"); title="Model & Convergence", style="#0077B6", width=right_w, height=panel_h)

    _render_frame!(tui, top_panel / (left_panel * right_panel), total_w)
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
