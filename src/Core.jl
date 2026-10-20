using UnicodePlots
using Distributions
using OrderedCollections
using Base: IdSet
using CSV, DataFrames

@inline safe_log(x::Real) = log(max(Float64(x), eps(Float64)))

# ============================================================================
# Parameter & Tau Index Helpers
# ============================================================================

function getRelaventTaus(
    parameter::Parameter,
    X::Vector{Observation},
    tau::AbstractVector{<:AbstractVector{<:Real}},
    parameter_map::Vector{Vector{Any}},
)
    result = Tuple{Observation, Float64, Any}[]
    for (x, tau_x, params_x) in zip(X, tau, parameter_map)
        for (pr, param) in zip(tau_x, params_x)
            if _param_in_dict(parameter, param)
                push!(result, (x, Float64(pr), param))
            end
        end
    end
    return result
end

@inline function _param_in_dict(target::Parameter, params)
    for (k, v) in pairs(params)
        if k !== :index_key && v === target
            return true
        end
    end
    return false
end

function getRelaventTausIndex(
    parameter::Parameter,
    X::Vector{Observation},
    tau::AbstractVector{<:AbstractVector{<:Real}},
    parameter_map::Vector{Vector{Any}},
)
    indx = Vector{Tuple{Int, Int}}()
    for (i_1, params_x) in enumerate(parameter_map)
        for (i_2, params) in enumerate(params_x)
            if _param_in_dict(parameter, params)
                push!(indx, (i_1, i_2))
            end
        end
    end
    return indx
end

function getRelaventTaus(
    parameter::Tuple{CategoricalZ, Int},
    X::Vector{Observation},
    tau::AbstractVector{<:AbstractVector{<:Real}},
    parameter_map::Vector{Vector{Any}},
)
    result = Float64[]
    for (_, tau_x, params_x) in zip(X, tau, parameter_map)
        for (pr, param) in zip(tau_x, params_x)
            if parameter in param
                push!(result, Float64(pr))
            end
        end
    end
    return result
end

function getRelaventTausIndex(
    parameter::Tuple{CategoricalZ, Int},
    X::Vector{Observation},
    tau::AbstractVector{<:AbstractVector{<:Real}},
    parameter_map::Vector{Vector{Any}},
)
    indx = Vector{Tuple{Int, Int}}()
    for (i_1, params_x) in enumerate(parameter_map)
        for (i_2, params) in enumerate(params_x)
            if parameter in params
                push!(indx, (i_1, i_2))
            end
        end
    end
    return indx
end

# ============================================================================
# Domain Resolution & Dependency Graph Traversal
# ============================================================================

function rawCallDomain(map::Vector{<:Pair})
    LV = Dict{Symbol, Any}()
    for (ke, va) in map
        if va isa LV_wrap
            LV[ke] = resolve_lv(va)
        elseif va isa Vector
            LV[ke] = [lv isa Int ? lv : resolve_lv(lv) for lv in va]
        else
            LV[ke] = va
        end
    end
    return LV
end

function callDomain(map::Vector{<:Pair})
    LV = Vector{LatentVaraible}()
    for (_, va) in map
        if va isa LV_wrap
            res = resolve_lv(va)
            res isa LatentVaraible && push!(LV, res)
        elseif va isa Vector
            for lv in va
                if !(lv isa Int)
                    res = resolve_lv(lv)
                    res isa LatentVaraible && push!(LV, res)
                end
            end
        end
    end
    return LV
end

function callDomain(X::Observation)
    T = X.T
    isnothing(T) && return nothing
    for (_, va) in T
        if va isa LV_wrap
            res = resolve_lv(va)
            if res isa LatentVaraible && !res.active
                return res
            end
        elseif va isa Vector
            for lv in va
                if !(lv isa Int)
                    res = resolve_lv(lv)
                    if res isa LatentVaraible && !res.active
                        return res
                    end
                end
            end
        end
    end
    return nothing
end

const EMPTY_LV_VEC = LatentVaraible[]

function callUnknownDomain(X::Observation)
    T = X.T
    isnothing(T) && return nothing
    for (_, va) in T
        if va isa LV_wrap
            res = resolve_lv(va)
            if res isa LatentVaraible && !lv_isKnown(res)
                return res
            end
        elseif va isa Vector
            for lv in va
                if !(lv isa Int)
                    res = resolve_lv(lv)
                    if res isa LatentVaraible && !lv_isKnown(res)
                        return res
                    end
                end
            end
        end
    end
    return nothing
end

function exhaustiveIndependenceCheck(G::Vector{Vector{Observation}})
    G_depo = [unique(reduce(vcat, GetDependentVariable.(X))) for X in G]
    n = length(G)
    for i in 1:(n - 1)
        for j in (i + 1):n
            shared = intersect(G_depo[i], G_depo[j])
            if !isempty(shared)
                for lv in shared
                    display(lv)
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

function GetDependentVariable!(X_i::Observation, used_conditions::Vector{LatentVaraible})
    condition = callDomain(X_i)
    if !isnothing(condition)
        if !(condition in used_conditions)
            push!(used_conditions, condition)
        end
        if condition.value_ isa Unknown
            for k in 1:condition.Z.K
                lv_set(condition, k)
                GetDependentVariable!(X_i, used_conditions)
                lv_set(condition, 0)
                if !condition.is_index
                    break
                end
            end
        else
            lv_set(condition, lv_v(condition))
            GetDependentVariable!(X_i, used_conditions)
            lv_set(condition, 0)
        end
    end
    return nothing
end

function _get_unknown_dependent_variables(X_i::Observation)
    out = Vector{LatentVaraible}()
    _get_unknown_dependent_variables!(X_i, out)
    return out
end

function _get_unknown_dependent_variables!(X_i::Observation, out::Vector{LatentVaraible})
    condition = callUnknownDomain(X_i)
    if !isnothing(condition)
        if !(condition in out)
            push!(out, condition)
        end
        for k in 1:condition.Z.K
            lv_set(condition, k)
            _get_unknown_dependent_variables!(X_i, out)
            lv_set(condition, 0)
            if !condition.is_index
                break
            end
        end
    end
    return out
end

@inline function _get_unknown_dependent_variables_fast(
    x::Observation,
    dom_x::Vector{LatentVaraible},
)::Vector{LatentVaraible}
    n_unk = 0
    has_active_index = false
    @inbounds for i in eachindex(dom_x)
        lv = dom_x[i]
        u = lv.value_
        if u isa Unknown
            v = u.value
            if v == 0
                n_unk += 1
            elseif lv.is_index
                has_active_index = true
            end
        end
    end
    n_unk == 0 && return EMPTY_LV_VEC
    if !has_active_index
        out = Vector{LatentVaraible}(undef, n_unk)
        idx = 1
        @inbounds for i in eachindex(dom_x)
            lv = dom_x[i]
            if !lv_isKnown(lv)
                out[idx] = lv
                idx += 1
            end
        end
        return out
    end
    return _get_unknown_dependent_variables(x)
end

@inline function _first_unknown_variable(
    x::Observation,
    dom_x::Vector{LatentVaraible},
)::Union{Nothing, LatentVaraible}
    first_unk = nothing
    has_active_index = false
    @inbounds for i in eachindex(dom_x)
        lv = dom_x[i]
        u = lv.value_
        if u isa Unknown
            v = u.value
            if v == 0
                if isnothing(first_unk)
                    first_unk = lv
                end
            elseif lv.is_index
                has_active_index = true
            end
        end
    end
    isnothing(first_unk) && return nothing
    !has_active_index && return first_unk
    return callUnknownDomain(x)
end

@inline function _may_have_shared_unknown(
    X::Vector{Observation},
    domain_map::Dict{Observation, Vector{LatentVaraible}},
)::Bool
    @inbounds for x in X
        dom_x = domain_map[x]
        for i in eachindex(dom_x)
            lv = dom_x[i]
            if !lv_isKnown(lv) && length(lv.dependent_X) > 1
                return true
            end
        end
    end
    return false
end

function getIndependentSets(
    X::Vector{Observation},
    conditions,
    independent_map,
    domain_map::Dict{Observation, Vector{LatentVaraible}},
)
    n = length(X)
    if n <= 1
        return Vector{Observation}[X]
    end
    if !_may_have_shared_unknown(X, domain_map)
        return Vector{Observation}[[x] for x in X]
    end
    domains = Dict{Observation, Vector{LatentVaraible}}()
    has_shared = false
    for x in X
        u_lvs = _get_unknown_dependent_variables_fast(x, domain_map[x])
        domains[x] = u_lvs
        for lv in u_lvs
            if length(lv.dependent_X) > 1
                has_shared = true
            end
        end
    end
    if !has_shared
        return Vector{Observation}[[x] for x in X]
    end

    Xset = Set{Observation}(X)
    domainSet = Set{LatentVaraible}()
    independent_sets = Vector{Vector{Observation}}()
    for x in X
        if x in Xset
            u_lvs = domains[x]
            if isempty(u_lvs) || all(lv -> length(lv.dependent_X) <= 1, u_lvs)
                delete!(Xset, x)
                push!(independent_sets, Observation[x])
            else
                current_set = Vector{Observation}()
                independent_search!(x, Xset, current_set, domains, domainSet)
                if length(current_set) > 1
                    sort!(current_set, by = objectid)
                end
                push!(independent_sets, current_set)
            end
        end
    end
    return independent_sets
end

function independent_search!(
    x::Observation,
    Xset::Set{Observation},
    current_set::Vector{Observation},
    domains::Dict{Observation, Vector{LatentVaraible}},
    domainSet::Set{LatentVaraible},
)
    push!(current_set, x)
    delete!(Xset, x)
    for condition in domains[x]
        if length(condition.dependent_X) > 1 && !(condition in domainSet)
            push!(domainSet, condition)
            for va in condition.dependent_X
                if va in Xset && (condition in domains[va])
                    independent_search!(va, Xset, current_set, domains, domainSet)
                end
            end
        end
    end
end

function conditional_dependent_search!(x::Observation, adj::Dict{Observation, Vector{Observation}}, visited::Set{Observation})
    for y in adj[x]
        if !(y in visited)
            push!(visited, y)
            conditional_dependent_search!(y, adj, visited)
        end
    end
end

function getDependentOnX(
    X::Vector{Observation},
    domain_map::Union{Nothing, Dict{Observation, Vector{LatentVaraible}}} = nothing,
)
    adj = Dict{Observation, Vector{Observation}}(x => Observation[x] for x in X)
    for x in X
        u_lvs = isnothing(domain_map) ? _get_unknown_dependent_variables(x) : _get_unknown_dependent_variables_fast(x, domain_map[x])
        neighbors = adj[x]
        for condition in u_lvs
            for va in condition.dependent_X
                if !(va in neighbors)
                    push!(neighbors, va)
                end
            end
        end
    end
    indo_sets = Dict{Observation, Set{Observation}}()
    for x in X
        visited = Set{Observation}()
        push!(visited, x)
        conditional_dependent_search!(x, adj, visited)
        indo_sets[x] = visited
    end
    return indo_sets
end

# ============================================================================
# Compiled Topological Density Evaluation DAG
# ============================================================================

struct LeafNode{E, P}
    out_idx::Int
    obs::Observation
    params::P
    evaluate::E
    is_const::Bool
end

@inline _eval_leaf(leaf::LeafNode)::Float64 = Float64(leaf.evaluate(leaf.obs, leaf.params))

struct OpNode
    out_idx::Int
    is_sum::Bool
    Z::Union{Nothing, CategoricalZ}
    child_indices::Vector{Int}
    k_indices::Vector{Int}
    is_const::Bool
end

mutable struct DensityDAG
    values::Vector{Float64}
    is_const_node::Vector{Bool}
    leaf_nodes::Vector{LeafNode}
    typed_leaves::Any
    op_nodes::Vector{OpNode}
    leaf_cache::Dict{UInt64, Int}
    subproblem_cache::Dict{UInt64, Int}
    mutable_obs::Set{Observation}
end

function DensityDAG(; mutable_obs::Set{Observation} = Set{Observation}())
    return DensityDAG(
        Float64[],
        Bool[],
        LeafNode[],
        nothing,
        OpNode[],
        Dict{UInt64, Int}(),
        Dict{UInt64, Int}(),
        mutable_obs,
    )
end

@inline function _alloc_slot!(dag::DensityDAG, is_const::Bool = false)
    push!(dag.values, 0.0)
    push!(dag.is_const_node, is_const)
    return length(dag.values)
end

struct DAGNodeRef
    dag::DensityDAG
    idx::Int
end

@inline (r::DAGNodeRef)() = @inbounds r.dag.values[r.idx]

struct EVAL{F}
    f::F
    is_eval::Bool
end
(f::EVAL)() = f.f()

function prod_foldl(G)
    s = 0.0
    for g in G
        s += g()
    end
    return s
end

function sum_foldl(G)
    vv = Vector{Float64}(undef, length(G))
    return function ()
        @inbounds for (i, g) in enumerate(G)
            vv[i] = g()
        end
        mm = maximum(vv)
        s = 0.0
        @inbounds for v in vv
            s += exp(v - mm)
        end
        return mm + safe_log(s)
    end
end

Base.@kwdef mutable struct HOLDER
    f::Any
    is_eval::Bool
    val::Float64 = 0.0
end

@inline function _get_dag(map_collector::OrderedDict)
    if haskey(map_collector, :__dag__)
        return map_collector[:__dag__]::DensityDAG
    else
        dag = DensityDAG()
        map_collector[:__dag__] = dag
        return dag
    end
end
@inline _get_dag(dag::DensityDAG) = dag

@inline function _eval_op_node!(vals::Vector{Float64}, node::OpNode)
    if node.is_sum
        Z_pi = (node.Z::CategoricalZ).Pi
        c_idxs = node.child_indices
        k_idxs = node.k_indices
        n_c = length(c_idxs)
        max_v = -Inf
        @inbounds for i in 1:n_c
            k = k_idxs[i]
            lp = k > 0 ? safe_log(Z_pi[k]) : 0.0
            v = lp + vals[c_idxs[i]]
            if v > max_v
                max_v = v
            end
        end
        s = 0.0
        @inbounds for i in 1:n_c
            k = k_idxs[i]
            lp = k > 0 ? safe_log(Z_pi[k]) : 0.0
            v = lp + vals[c_idxs[i]]
            s += exp(v - max_v)
        end
        @inbounds vals[node.out_idx] = max_v + safe_log(s)
    else
        s = 0.0
        @inbounds for c_idx in node.child_indices
            s += vals[c_idx]
        end
        @inbounds vals[node.out_idx] = s
    end
    return nothing
end

function _eval_all_leaves!(vals::Vector{Float64}, leaves::Vector{L}) where {L<:LeafNode}
    @inbounds for leaf in leaves
        if !leaf.is_const
            vals[leaf.out_idx] = _eval_leaf(leaf)
        end
    end
    return nothing
end

function call_collection(dag::DensityDAG)
    vals = dag.values
    tl = dag.typed_leaves
    if !isnothing(tl)
        _eval_all_leaves!(vals, tl)
    else
        for leaf in dag.leaf_nodes
            if !leaf.is_const
                @inbounds vals[leaf.out_idx] = _eval_leaf(leaf)
            end
        end
    end
    for node in dag.op_nodes
        if !node.is_const
            _eval_op_node!(vals, node)
        end
    end
    return nothing
end

function call_collection(C::OrderedDict)
    if haskey(C, :__dag__)
        call_collection(C[:__dag__]::DensityDAG)
    else
        for (_, va) in C
            va.val = va.f()
        end
    end
    return nothing
end

@inline function _singleton_key(g::Observation, lvs::Vector{LatentVaraible})::UInt64
    h = hash(objectid(g), UInt64(0x9e3779b97f4a7c15))
    @inbounds for i in eachindex(lvs)
        lv = lvs[i]
        v = lv_isKnown(lv) ? lv_v(lv) : 0
        h = hash(v, h)
    end
    return h
end

@inline function _subproblem_key(G::Vector{Observation}, domain_map::Dict{Observation, Vector{LatentVaraible}})::UInt64
    if length(G) == 1
        g = G[1]
        return _singleton_key(g, domain_map[g])
    end
    h = UInt64(0x9e3779b97f4a7c15)
    @inbounds for x in G
        h = hash(objectid(x), h)
        lvs = domain_map[x]
        for i in eachindex(lvs)
            lv = lvs[i]
            v = lv_isKnown(lv) ? lv_v(lv) : 0
            h = hash(v, h)
        end
    end
    return h
end

@inline function _prod_key(mult_indices::Vector{Int})::UInt64
    h = UInt64(0x517cc1b727220a95)
    @inbounds for idx in mult_indices
        h = hash(idx, h)
    end
    return h
end

function _select_next_condition(G::Vector{Observation}, all_domains::Vector{Vector{LatentVaraible}})
    if length(G) == 1
        d1 = all_domains[1]
        return isempty(d1) ? nothing : d1[1]
    end
    max_dom_len = 0
    max_dom_idx = 0
    freq_map = Dict{LatentVaraible, Int}()
    max_freq = 0
    best_freq_lv = nothing
    for (i, dom) in enumerate(all_domains)
        len = length(dom)
        if len > max_dom_len
            max_dom_len = len
            max_dom_idx = i
        end
        for lv in dom
            c = get(freq_map, lv, 0) + 1
            freq_map[lv] = c
            if c > max_freq
                max_freq = c
                best_freq_lv = lv
            end
        end
    end
    max_dom_len == 0 && return nothing
    if max_dom_len > max_freq
        return all_domains[max_dom_idx][1]
    else
        return best_freq_lv
    end
end

function _push_typed_leaf!(dag::DensityDAG, leaf::L) where {L<:LeafNode}
    push!(dag.leaf_nodes, leaf)
    tl = dag.typed_leaves
    if isnothing(tl) && length(dag.leaf_nodes) == 1
        dag.typed_leaves = L[leaf]
    elseif tl isa Vector{L}
        push!(tl, leaf)
    else
        dag.typed_leaves = nothing
    end
    return nothing
end

function _compile_leaf!(g::Observation, density::Parsa_Base, dag::DensityDAG, should_eval::Bool)
    ma = rawCallDomain(g.T)
    mm = index_to_parameters(ma, density.parameters)
    leaf_key = hash(objectid(g), mm[:index_key]::UInt64)
    existing = get(dag.leaf_cache, leaf_key, 0)
    if existing != 0
        return existing
    end
    is_const = should_eval && !isnothing(g.X) && !(g in dag.mutable_obs)
    slot = _alloc_slot!(dag, is_const)
    eval_fn = density.evaluate
    leaf = LeafNode(slot, g, mm, eval_fn, is_const)
    _push_typed_leaf!(dag, leaf)
    if is_const
        dag.values[slot] = _eval_leaf(leaf)
    end
    dag.leaf_cache[leaf_key] = slot
    return slot
end

function _eval_singleton_idx(
    g::Observation,
    density::Parsa_Base,
    domain_map::Dict{Observation, Vector{LatentVaraible}},
    dag::DensityDAG;
    should_eval::Bool = false,
)::Int
    dom_g = domain_map[g]
    sub_key = _singleton_key(g, dom_g)
    cached_idx = get(dag.subproblem_cache, sub_key, 0)
    if cached_idx != 0
        return cached_idx
    end

    next_condition = _first_unknown_variable(g, dom_g)
    if !isnothing(next_condition)
        K = next_condition.value_ isa Unknown ? (1:next_condition.Z.K) : (lv_v(next_condition):lv_v(next_condition))
        is_unknown = next_condition.value_ isa Unknown
        child_indices = Vector{Int}(undef, length(K))
        k_indices = Vector{Int}(undef, length(K))
        for (i_k, k) in enumerate(K)
            lv_set(next_condition, k)
            child_indices[i_k] = _eval_singleton_idx(g, density, domain_map, dag; should_eval = should_eval)
            k_indices[i_k] = is_unknown ? k : 0
            lv_set(next_condition, 0)
        end
        is_const = should_eval && all(c -> dag.is_const_node[c], child_indices)
        g_slot = _alloc_slot!(dag, is_const)
        node = OpNode(g_slot, true, next_condition.Z, child_indices, k_indices, is_const)
        push!(dag.op_nodes, node)
        if is_const
            _eval_op_node!(dag.values, node)
        end
        dag.subproblem_cache[sub_key] = g_slot
        return g_slot
    else
        g_slot = _compile_leaf!(g, density, dag, should_eval)
        dag.subproblem_cache[sub_key] = g_slot
        return g_slot
    end
end

function initialize_density_evaluation_idx(
    X::Vector{Observation},
    conditioned_domains::Vector,
    density::Parsa_Base,
    domain_map::Dict{Observation, Vector{LatentVaraible}},
    dag::DensityDAG,
    independent_map::Dict;
    should_eval::Bool = false,
    check_independence::Bool = false,
)::Int
    n_x = length(X)
    if n_x == 1 && !check_independence
        return _eval_singleton_idx(X[1], density, domain_map, dag; should_eval = should_eval)
    end

    if !check_independence && !_may_have_shared_unknown(X, domain_map)
        mult_indices = Vector{Int}(undef, n_x)
        @inbounds for i in 1:n_x
            mult_indices[i] = _eval_singleton_idx(X[i], density, domain_map, dag; should_eval = should_eval)
        end
        prod_key = _prod_key(mult_indices)
        existing_prod = get(dag.subproblem_cache, prod_key, 0)
        if existing_prod != 0
            return existing_prod
        end
        is_const = should_eval && all(c -> dag.is_const_node[c], mult_indices)
        out_slot = _alloc_slot!(dag, is_const)
        node = OpNode(out_slot, false, nothing, mult_indices, Int[], is_const)
        push!(dag.op_nodes, node)
        if is_const
            _eval_op_node!(dag.values, node)
        end
        dag.subproblem_cache[prod_key] = out_slot
        return out_slot
    end

    independent_sets = getIndependentSets(X, conditioned_domains, independent_map, domain_map)
    if check_independence
        exhaustiveIndependenceCheck(independent_sets)
    end
    mult_indices = Vector{Int}()
    sizehint!(mult_indices, length(independent_sets))

    for G in independent_sets
        if length(G) == 1 && !check_independence
            push!(mult_indices, _eval_singleton_idx(G[1], density, domain_map, dag; should_eval = should_eval))
            continue
        end
        sub_key = _subproblem_key(G, domain_map)
        cached_idx = get(dag.subproblem_cache, sub_key, 0)
        if cached_idx != 0
            push!(mult_indices, cached_idx)
            continue
        end

        all_domains = Vector{Vector{LatentVaraible}}(undef, length(G))
        for (i, x) in enumerate(G)
            all_domains[i] = _get_unknown_dependent_variables_fast(x, domain_map[x])
        end

        next_condition = _select_next_condition(G, all_domains)
        if !isnothing(next_condition)
            K = next_condition.value_ isa Unknown ? (1:next_condition.Z.K) : (lv_v(next_condition):lv_v(next_condition))
            is_unknown = next_condition.value_ isa Unknown
            child_indices = Vector{Int}(undef, length(K))
            k_indices = Vector{Int}(undef, length(K))
            new_conditions = check_independence ? [conditioned_domains; next_condition] : conditioned_domains
            for (i_k, k) in enumerate(K)
                lv_set(next_condition, k)
                child_indices[i_k] = initialize_density_evaluation_idx(
                    G,
                    new_conditions,
                    density,
                    domain_map,
                    dag,
                    independent_map;
                    should_eval = should_eval,
                    check_independence = check_independence,
                )
                k_indices[i_k] = is_unknown ? k : 0
                lv_set(next_condition, 0)
            end
            is_const = should_eval && all(c -> dag.is_const_node[c], child_indices)
            g_slot = _alloc_slot!(dag, is_const)
            node = OpNode(g_slot, true, next_condition.Z, child_indices, k_indices, is_const)
            push!(dag.op_nodes, node)
            if is_const
                _eval_op_node!(dag.values, node)
            end
            dag.subproblem_cache[sub_key] = g_slot
            push!(mult_indices, g_slot)
        else
            if length(G) == 1
                g_slot = _compile_leaf!(G[1], density, dag, should_eval)
                dag.subproblem_cache[sub_key] = g_slot
                push!(mult_indices, g_slot)
            else
                leaf_slots = Int[_compile_leaf!(g, density, dag, should_eval) for g in G]
                is_const = should_eval && all(c -> dag.is_const_node[c], leaf_slots)
                g_slot = _alloc_slot!(dag, is_const)
                node = OpNode(g_slot, false, nothing, leaf_slots, Int[], is_const)
                push!(dag.op_nodes, node)
                if is_const
                    _eval_op_node!(dag.values, node)
                end
                dag.subproblem_cache[sub_key] = g_slot
                push!(mult_indices, g_slot)
            end
        end
    end

    if length(mult_indices) == 1
        return mult_indices[1]
    else
        prod_key = _prod_key(mult_indices)
        existing_prod = get(dag.subproblem_cache, prod_key, 0)
        if existing_prod != 0
            return existing_prod
        end
        is_const = should_eval && all(c -> dag.is_const_node[c], mult_indices)
        out_slot = _alloc_slot!(dag, is_const)
        node = OpNode(out_slot, false, nothing, mult_indices, Int[], is_const)
        push!(dag.op_nodes, node)
        if is_const
            _eval_op_node!(dag.values, node)
        end
        dag.subproblem_cache[prod_key] = out_slot
        return out_slot
    end
end

function initialize_density_evaluation(
    X::Vector{Observation},
    conditioned_domains::Vector,
    density::Parsa_Base,
    domain_map::Dict,
    map_collector,
    independent_map::Dict;
    should_eval::Bool = false,
    check_independence::Bool = false,
)
    dag = _get_dag(map_collector)
    typed_domain_map = if domain_map isa Dict{Observation, Vector{LatentVaraible}}
        domain_map
    else
        Dict{Observation, Vector{LatentVaraible}}(k => v for (k, v) in domain_map)
    end
    idx = initialize_density_evaluation_idx(
        X,
        conditioned_domains,
        density,
        typed_domain_map,
        dag,
        independent_map;
        should_eval = should_eval,
        check_independence = check_independence,
    )
    return DAGNodeRef(dag, idx)
end

function prime_X(X::Observation)
    domains = GetDependentVariable(X)
    for lv in domains
        push!(lv.dependent_X, X)
    end
    return nothing
end

# ============================================================================
# E-Step & M-Step Structures
# ============================================================================

struct EStepBranch
    dag_idx::Int
    pi_z::Vector{CategoricalZ}
    pi_k::Vector{Int}
end

struct TauEvaluator
    dag::DensityDAG
    branches::Vector{EStepBranch}
    out_buf::Vector{Float64}
end

function (te::TauEvaluator)()
    vals = te.dag.values
    branches = te.branches
    out = te.out_buf
    m = length(branches)
    max_l = -Inf
    @inbounds for i in 1:m
        br = branches[i]
        lp = vals[br.dag_idx]
        pi_z = br.pi_z
        pi_k = br.pi_k
        for j in eachindex(pi_z)
            lp += safe_log(pi_z[j].Pi[pi_k[j]])
        end
        out[i] = lp
        if lp > max_l
            max_l = lp
        end
    end
    s = 0.0
    @inbounds for i in 1:m
        e = exp(out[i] - max_l)
        out[i] = e
        s += e
    end
    inv_s = 1.0 / s
    @inbounds for i in 1:m
        out[i] *= inv_s
    end
    return out
end

struct TauInitEvaluator
    probs::Union{Nothing, Vector{Float64}}
end

@inline (tie::TauInitEvaluator)() = tie.probs

function wild_tau(tau)
    n = length(tau)
    tau_new = zeros(Float64, n)
    u = rand()
    cum = 0.0
    chosen = n
    @inbounds for k in 1:n
        cum += Float64(tau[k])
        if u <= cum
            chosen = k
            break
        end
    end
    @inbounds tau_new[chosen] = 1.0
    return tau_new
end

function wild_tau!(tau_new::Vector{Float64}, tau::Vector{Float64})
    n = length(tau)
    fill!(tau_new, 0.0)
    u = rand()
    cum = 0.0
    chosen = n
    @inbounds for k in 1:n
        cum += tau[k]
        if u <= cum
            chosen = k
            break
        end
    end
    @inbounds tau_new[chosen] = 1.0
    return tau_new
end

function normalize_log_post(L)
    n = length(L)
    po = Vector{Float64}(undef, n)
    max_l = maximum(L)
    s = 0.0
    @inbounds for i in 1:n
        e = exp(Float64(L[i]) - max_l)
        po[i] = e
        s += e
    end
    inv_s = 1.0 / s
    @inbounds for i in 1:n
        po[i] *= inv_s
    end
    return po
end

function E_step_initalize(
    X::Vector{Observation},
    density::Parsa_Base,
    all_domains::Dict{Observation, Vector{LatentVaraible}},
    map_collector,
    independent_map::Dict,
    verbose::Bool,
)
    n = length(X)
    dag = _get_dag(map_collector)
    tau = Vector{TauEvaluator}(undef, n)
    tau_init = Vector{TauInitEvaluator}(undef, n)
    parameters_used = Vector{Vector{Any}}(undef, n)
    pi_parameters_used = Vector{Vector{Any}}(undef, n)
    all_dependent_observations = getDependentOnX(X, all_domains)

    for i in 1:n
        dep_obs = collect(all_dependent_observations[X[i]])
        if length(dep_obs) > 1
            sort!(dep_obs, by = objectid)
        end
        tau_init_i = E_step_i_initalize_initzial_values(X[i], dep_obs, density, LatentVaraible[], LatentVaraible[])
        (tau_eval_i, parameters_used_i, pi_parameters_used_i) =
            E_step_i_initalize(X[i], dep_obs, density, LatentVaraible[], LatentVaraible[], all_domains, dag, independent_map)

        init_probs = if !isempty(tau_init_i)
            s = sum(tau_init_i)
            Float64[v / s for v in tau_init_i]
        else
            nothing
        end

        tau[i] = tau_eval_i
        tau_init[i] = TauInitEvaluator(init_probs)
        parameters_used[i] = parameters_used_i
        pi_parameters_used[i] = pi_parameters_used_i
    end
    return (tau, parameters_used, pi_parameters_used, tau_init)
end

function get_pre_set_factor(condition::Vector{LatentVaraible}, k_list)
    return sum([d.v() for d in condition] .== k_list) != 0
end

function E_step_i_initalize(
    X_i::Observation,
    X::Vector{Observation},
    density::Parsa_Base,
    used_conditions::Vector,
    domains::Vector,
    domain_map::Dict{Observation, Vector{LatentVaraible}},
    map_collector,
    independent_map::Dict,
)
    dag = _get_dag(map_collector)
    branches = Vector{EStepBranch}()
    params_used = Vector{Any}()
    Pi_used = Vector{Any}()
    _e_step_i_compile!(X_i, X, density, LatentVaraible[], domain_map, dag, independent_map, branches, params_used, Pi_used)
    out_buf = Vector{Float64}(undef, length(branches))
    return (TauEvaluator(dag, branches, out_buf), params_used, Pi_used)
end

function _e_step_i_compile!(
    X_i::Observation,
    X::Vector{Observation},
    density::Parsa_Base,
    used_conditions::Vector{LatentVaraible},
    domain_map::Dict{Observation, Vector{LatentVaraible}},
    dag::DensityDAG,
    independent_map::Dict,
    branches::Vector{EStepBranch},
    params_used::Vector{Any},
    Pi_used::Vector{Any},
)
    condition = callDomain(X_i)
    if isnothing(condition)
        return nothing
    end
    K = condition.value_ isa Unknown ? (1:condition.Z.K) : (lv_v(condition):lv_v(condition))
    next_used = LatentVaraible[used_conditions; condition]
    for k in K
        lv_set(condition, k)
        if !isnothing(callDomain(X_i))
            _e_step_i_compile!(X_i, X, density, next_used, domain_map, dag, independent_map, branches, params_used, Pi_used)
        else
            push!(Pi_used, Tuple((d.Z, lv_v(d)) for d in next_used))
            params = index_to_parameters(rawCallDomain(X_i.T), density.parameters)
            dag_idx = initialize_density_evaluation_idx(X, next_used, density, domain_map, dag, independent_map)
            pi_z = CategoricalZ[]
            pi_k = Int[]
            if condition.value_ isa Unknown
                for d in next_used
                    push!(pi_z, d.Z)
                    push!(pi_k, lv_v(d))
                end
            end
            push!(branches, EStepBranch(dag_idx, pi_z, pi_k))
            push!(params_used, params)
        end
        lv_set(condition, 0)
    end
    return nothing
end

function E_step_i_initalize_initzial_values(
    X_i::Observation,
    X::Vector{Observation},
    density::Parsa_Base,
    used_conditions::Vector,
    domains::Vector,
)
    tau_chain_init = Vector{Int}()
    E_step_i_initalize_initzial_values!(X_i, X, density, used_conditions, domains, tau_chain_init)
    return tau_chain_init
end

function E_step_i_initalize_initzial_values!(
    X_i::Observation,
    X::Vector{Observation},
    density::Parsa_Base,
    used_conditions::Vector,
    domains::Vector,
    tau_chain_init::Vector{Int},
)
    condition = callDomain(X_i)
    if isnothing(condition)
        return tau_chain_init
    end
    K = condition.value_ isa Unknown ? (1:condition.Z.K) : (lv_v(condition):lv_v(condition))
    pre_set_val = condition.init_value
    used_conditions = [used_conditions; condition]
    for k in K
        init_correction = isnothing(pre_set_val) ? 1 : (pre_set_val == k ? 1 : 0)
        lv_set(condition, k)
        if !isnothing(callDomain(X_i))
            E_step_i_initalize_initzial_values!(X_i, X, density, used_conditions, domains, tau_chain_init)
        else
            push!(tau_chain_init, init_correction)
        end
        lv_set(condition, 0)
    end
    return tau_chain_init
end

# ============================================================================
# Parameter & Mixing Weight Updaters / Resetters
# ============================================================================

function get_parameter_package(
    params::Parameter,
    X::Vector{Observation},
    tau::AbstractVector{<:AbstractVector{<:Real}},
    parameter_map::Vector{Vector{Any}},
)
    relavent = getRelaventTaus(params, X, tau, parameter_map)
    x = Observation[y[1] for y in relavent]
    pr = Float64[y[2] for y in relavent]
    map = [y[3] for y in relavent]
    return (x, pr, map)
end

function zip_package(
    mapping::Vector{Tuple{Int, Int}},
    X::Vector{Observation},
    tau::AbstractVector{<:AbstractVector{<:Real}},
    parameter_map::Vector{Vector{Any}},
)
    n = length(mapping)
    x = Vector{Observation}(undef, n)
    pr = Vector{Float64}(undef, n)
    map = Vector{Any}(undef, n)
    @inbounds for i in 1:n
        (x_i, p_i) = mapping[i]
        x[i] = X[x_i]
        pr[i] = Float64(tau[x_i][p_i])
        map[i] = parameter_map[x_i][p_i]
    end
    return (x, pr, map)
end

struct ParamResetter
    params::Vector{Parameter}
end

function (pr::ParamResetter)()
    for param in pr.params
        param.value.value = deepcopy(param.value_original)
    end
    return nothing
end

function set_original_parameters(base::Parsa_Base)
    params = Parameter[]
    for ke in base.parameter_order
        for (_, param) in base.parameters[ke].parameter_map
            param.value_original = deepcopy(param.value.value)
            push!(params, param)
        end
    end
    return ParamResetter(params)
end

struct ParamSaver
    params::Vector{Parameter}
    saved_values::Vector{Any}
end

function (ps::ParamSaver)()
    @inbounds for i in eachindex(ps.params)
        ps.params[i].value.value = deepcopy(ps.saved_values[i])
    end
    return nothing
end

function save_parameters(base::Parsa_Base)
    params = Parameter[]
    saved_values = Any[]
    for ke in base.parameter_order
        for (_, param) in base.parameters[ke].parameter_map
            if !param.is_const
                push!(params, param)
                push!(saved_values, deepcopy(param.value.value))
            end
        end
    end
    return ParamSaver(params, saved_values)
end

struct MStepEntry{R, M}
    index_map::Vector{Tuple{Int, Int}}
    x_vec::Vector{Observation}
    pr_vec::Vector{Float64}
    map_vec::Vector{M}
    run_update::R
end

struct MStepRunner
    entries::Vector{MStepEntry}
end

function _run_m_step_entry!(entry::MStepEntry, tau, log_pdf)
    idx_map = entry.index_map
    pr_vec = entry.pr_vec
    @inbounds for k in eachindex(idx_map)
        (x_i, p_i) = idx_map[k]
        pr_vec[k] = Float64(tau[x_i][p_i])
    end
    entry.run_update(entry.x_vec, pr_vec, entry.map_vec, log_pdf)
    return nothing
end

function (ms::MStepRunner)(X, tau, parameter_map, base::Parsa_Base)
    log_pdf = base.log_pdf
    for entry in ms.entries
        _run_m_step_entry!(entry, tau, log_pdf)
    end
    return nothing
end

function _make_mstep_entry(
    idx_map::Vector{Tuple{Int, Int}},
    X::Vector{Observation},
    parameter_map::Vector{Vector{Any}},
    param::Parameter{T, U, R},
    ::Type{M},
) where {T, U, R, M}
    m = length(idx_map)
    x_vec = Observation[X[x_i] for (x_i, _) in idx_map]
    pr_vec = zeros(Float64, m)
    map_vec = M[parameter_map[x_i][p_i] for (x_i, p_i) in idx_map]
    return MStepEntry{R, M}(idx_map, x_vec, pr_vec, map_vec, param.run_update)
end

function M_step_init(
    X::Vector{Observation},
    tau::AbstractVector{<:AbstractVector{<:Real}},
    parameter_map::Vector{Vector{Any}},
    base::Parsa_Base,
)
    entries = Vector{MStepEntry}()
    for ke in base.parameter_order
        for (_, param) in base.parameters[ke].parameter_map
            if !param.is_const
                idx_map = getRelaventTausIndex(param, X, tau, parameter_map)
                if !isempty(idx_map)
                    (x1, p1) = idx_map[1]
                    M_type = typeof(parameter_map[x1][p1])
                    push!(entries, _make_mstep_entry(idx_map, X, parameter_map, param, M_type))
                end
            end
        end
    end
    return MStepRunner(entries)
end

function post_process_params(base::Parsa_Base)
    for ke in base.parameter_order
        for (_, param) in base.parameters[ke].parameter_map
            if !isnothing(param.post_processing)
                param.value.value = param.post_processing(param.value.value)
            end
        end
    end
    return nothing
end

function _collect_all_Z(X::Vector{Observation}, domain_map::Union{Nothing, Dict{Observation, Vector{LatentVaraible}}} = nothing)
    all_Z = Vector{CategoricalZ}()
    seen = Set{UInt64}()
    for x in X
        lvs = isnothing(domain_map) ? GetDependentVariable(x) : domain_map[x]
        for lv in lvs
            z = lv.Z
            zid = objectid(z)
            if !(zid in seen)
                push!(seen, zid)
                push!(all_Z, z)
            end
        end
    end
    return all_Z
end

struct PiSaver
    all_Z::Vector{CategoricalZ}
    vals::Vector{Vector{Float64}}
end

function (ps::PiSaver)()
    @inbounds for i in eachindex(ps.all_Z)
        copyto!(ps.all_Z[i].Pi, ps.vals[i])
    end
    return nothing
end

function save_Pi(X::Vector{Observation}, domain_map::Union{Nothing, Dict{Observation, Vector{LatentVaraible}}} = nothing)
    all_Z = _collect_all_Z(X, domain_map)
    vals = Vector{Float64}[copy(Z.Pi) for Z in all_Z]
    return PiSaver(all_Z, vals)
end

struct PiResetter
    all_Z::Vector{CategoricalZ}
end

function (pr::PiResetter)()
    for Z in pr.all_Z
        if !Z.constant
            k = length(Z.Pi)
            fill!(Z.Pi, 1.0 / k)
        end
    end
    return nothing
end

function reset_Pi(X::Vector{Observation}, domain_map::Union{Nothing, Dict{Observation, Vector{LatentVaraible}}} = nothing)
    all_Z = _collect_all_Z(X, domain_map)
    return PiResetter(all_Z)
end

struct PiUpdater
    all_Z::Vector{CategoricalZ}
    mappings::Vector{Vector{Vector{Tuple{Int, Int}}}}
end

function (pu::PiUpdater)(tau)
    for (i, Z_cat) in enumerate(pu.all_Z)
        Z_cat.constant && continue
        pi_vec = Z_cat.Pi
        map_i = pu.mappings[i]
        total = 0.0
        @inbounds for k in 1:Z_cat.K
            s = 0.0
            for (i_1, i_2) in map_i[k]
                s += Float64(tau[i_1][i_2])
            end
            pi_vec[k] = s
            total += s
        end
        if total > 0.0
            inv_total = 1.0 / total
            @inbounds for k in 1:Z_cat.K
                pi_vec[k] *= inv_total
            end
        end
    end
    return nothing
end

function update_pi_from_dag!(pu::PiUpdater, dag::DensityDAG, root_idx::Int, adj::Vector{Float64})
    for z_cat in pu.all_Z
        if !z_cat.constant
            fill!(z_cat.pi_acc, 0.0)
        end
    end
    fill!(adj, 0.0)
    @inbounds adj[root_idx] = 1.0
    vals = dag.values
    op_nodes = dag.op_nodes
    @inbounds for idx in length(op_nodes):-1:1
        node = op_nodes[idx]
        w = adj[node.out_idx]
        w == 0.0 && continue
        if node.is_sum
            z_cat = node.Z::CategoricalZ
            Z_pi = z_cat.Pi
            z_acc = z_cat.pi_acc
            is_var = !z_cat.constant
            p_val = vals[node.out_idx]
            c_idxs = node.child_indices
            k_idxs = node.k_indices
            for j in eachindex(c_idxs)
                k = k_idxs[j]
                lp = k > 0 ? safe_log(Z_pi[k]) : 0.0
                prob = exp(lp + vals[c_idxs[j]] - p_val)
                wp = w * prob
                adj[c_idxs[j]] += wp
                if is_var && k > 0
                    z_acc[k] += wp
                end
            end
        else
            for c_idx in node.child_indices
                adj[c_idx] += w
            end
        end
    end
    for z_cat in pu.all_Z
        z_cat.constant && continue
        z_acc = z_cat.pi_acc
        s = 0.0
        @inbounds for k in 1:z_cat.K
            s += z_acc[k]
        end
        if s > 0.0
            inv_s = 1.0 / s
            @inbounds for k in 1:z_cat.K
                z_cat.Pi[k] = z_acc[k] * inv_s
            end
        end
    end
    return nothing
end

function Pi_init(
    X::Vector{Observation},
    tau::AbstractVector{<:AbstractVector{<:Real}},
    pi_parameters_used::Vector{Vector{Any}},
    domain_map::Union{Nothing, Dict{Observation, Vector{LatentVaraible}}} = nothing,
)
    all_Z = _collect_all_Z(X, domain_map)
    mappings = Vector{Vector{Vector{Tuple{Int, Int}}}}(undef, length(all_Z))
    for (i, Z_cat) in enumerate(all_Z)
        map_z = Vector{Vector{Tuple{Int, Int}}}(undef, Z_cat.K)
        for k in 1:Z_cat.K
            map_z[k] = getRelaventTausIndex((Z_cat, k), X, tau, pi_parameters_used)
        end
        mappings[i] = map_z
    end
    return PiUpdater(all_Z, mappings)
end

# ============================================================================
# EM Algorithm (LMEM) & Visualization
# ============================================================================

function LMEM(
    X_set::Set{Observation},
    base::Parsa_Base;
    eps = 10^-6,
    init_eps = 10^-4,
    n_init = 1,
    verbose = true,
    max_steps = 1000,
    catch_init_error = false,
    allow_desc_likelihood = false,
    n_M_steps = 1,
)
    X = collect(X_set)
    n = length(X)
    for x in X
        prime_X(x)
    end
    base.eval_catch = Dict{Any, Any}()

    domain_map = Dict{Observation, Vector{LatentVaraible}}(x => unique(GetDependentVariable(x)) for x in X)
    dag = DensityDAG()
    independent_map = Dict{Any, Any}()

    (tau_chain, parameter_map, pi_parameters_used, tau_init) =
        E_step_initalize(X, base, domain_map, dag, independent_map, verbose)
    likelihood = initialize_density_evaluation(X, LatentVaraible[], base, domain_map, dag, independent_map)
    dag_adj = zeros(Float64, length(dag.values))

    init_probs = Vector{Float64}[ta() for ta in tau_init]
    tau_work = Vector{Float64}[zeros(Float64, length(p)) for p in init_probs]

    M = M_step_init(X, tau_work, parameter_map, base)
    param_reset = set_original_parameters(base)
    pi_reset = reset_Pi(X, domain_map)
    Pi = Pi_init(X, tau_work, pi_parameters_used, domain_map)

    init_likelihoods = Vector{Float64}[Float64[] for _ in 1:n_init]
    best_likelihood = -Inf
    best_tau = nothing
    best_prams = nothing
    best_Pi = nothing

    last_plot_ns = UInt64(0)
    plot_interval_ns = UInt64(50_000_000) # 20 FPS max during interactive runs

    for i_init in 1:n_init
        param_reset()
        pi_reset()
        for i in 1:n
            wild_tau!(tau_work[i], init_probs[i])
        end
        Pi(tau_work)
        M(X, tau_work, parameter_map, base)
        call_collection(dag)
        lik_old = -Inf
        lik_new = likelihood()
        try
            for i_wild in 1:max_steps
                for i in 1:n
                    tau_chain[i]()
                end
                tau_work_from_chain!(tau_work, tau_chain)
                update_pi_from_dag!(Pi, dag, likelihood.idx, dag_adj)
                for _ in 1:n_M_steps
                    M(X, tau_work, parameter_map, base)
                end
                call_collection(dag)
                lik_old = lik_new
                lik_new = likelihood()
                if (lik_new < lik_old && i_wild != 1) && !allow_desc_likelihood
                    @warn "init error... decreasing likelihood"
                end
                push!(init_likelihoods[i_init], lik_new)
                if verbose && (stdout isa Base.TTY)
                    now_ns = time_ns()
                    if now_ns - last_plot_ns >= plot_interval_ns
                        plotit(init_likelihoods, Float64[])
                        last_plot_ns = now_ns
                    end
                end
                if abs(lik_new - lik_old) / abs(lik_old) < init_eps
                    break
                end
            end
        catch e
            if !catch_init_error
                rethrow(e)
            else
                if !verbose
                    @warn e
                end
                continue
            end
        end
        if lik_new > best_likelihood || isnothing(best_prams)
            best_likelihood = lik_new
            best_tau = Vector{Float64}[copy(t) for t in tau_work]
            best_prams = save_parameters(base)
            best_Pi = save_Pi(X, domain_map)
        end
    end

    best_Pi()
    best_prams()
    for i in 1:n
        copyto!(tau_work[i], best_tau[i])
    end
    call_collection(dag)
    lik_old = -Inf
    lik_new = likelihood()

    all_likelihoods = Float64[lik_new]
    i = 1
    neg_lik_flag = false
    while (abs(lik_new - lik_old) / abs(lik_new)) > eps && max_steps > 0 && i <= max_steps
        if (lik_new < lik_old) && !allow_desc_likelihood && i > 3
            @warn ("non-increasing likelihood... got rel. differnce of " * string(abs(lik_new - lik_old) / abs(lik_new)))
            break
        end
        for j in 1:n
            tau_chain[j]()
        end
        tau_work_from_chain!(tau_work, tau_chain)
        update_pi_from_dag!(Pi, dag, likelihood.idx, dag_adj)
        for _ in 1:n_M_steps
            M(X, tau_work, parameter_map, base)
        end
        call_collection(dag)
        lik_old = lik_new
        lik_new = likelihood()
        push!(all_likelihoods, Float64(lik_new))
        if verbose && (stdout isa Base.TTY)
            now_ns = time_ns()
            if now_ns - last_plot_ns >= plot_interval_ns
                plotit(init_likelihoods, all_likelihoods)
                last_plot_ns = now_ns
            end
        end
        i += 1
    end

    if verbose && (stdout isa Base.TTY)
        plotit(init_likelihoods, all_likelihoods)
    end

    post_process_params(base)
    return (log_likelihood = likelihood, n = n, neg_lik_flag = neg_lik_flag)
end

@inline function tau_work_from_chain!(tau_work::Vector{Vector{Float64}}, tau_chain::Vector{TauEvaluator})
    @inbounds for i in eachindex(tau_work)
        copyto!(tau_work[i], tau_chain[i].out_buf)
    end
    return tau_work
end

function plotit(lines, final_lines)
    println("\33[H")
    print("\33c\e[3J")
    terminal = displaysize(stdout)
    lll = Float64[]
    for l in lines
        for v in l
            v != 0 && push!(lll, Float64(v))
        end
    end
    for v in final_lines
        v != 0 && push!(lll, Float64(v))
    end
    isempty(lll) && return nothing
    ymax = maximum(lll)
    ymin = minimum(lll)
    if ymin == ymax
        ymin -= 1.0
        ymax += 1.0
    end
    xmin = 0
    xmax_init = maximum(length(l) for l in lines; init = 1)
    xmax = max(xmax_init + length(final_lines), 1)
    plt = UnicodePlots.lineplot(
        [0],
        [0];
        ylim = (ymin, ymax),
        xlim = (xmin, xmax),
        xlabel = "steps",
        ylabel = "log-likelihood",
        height = max(Int(round(terminal[1] / 2)), 5),
        width = max(Int(round(terminal[2] / 2)), 10),
        color = :red,
    )
    for n_l in lines
        if !isempty(n_l)
            UnicodePlots.lineplot!(plt, 1:length(n_l), n_l, color = :blue)
        end
    end
    if !isempty(final_lines)
        UnicodePlots.lineplot!(plt, xmax_init:(length(final_lines) + xmax_init - 1), final_lines)
    end
    display(plt)
    return nothing
end

# ============================================================================
# Posterior Probability Evaluation
# ============================================================================

struct PosteriorEvaluator
    dag::DensityDAG
    branches::Vector{EStepBranch}
    pi_labels::Vector{Vector{Int}}
    vv::Vector{Float64}
end

function (pe::PosteriorEvaluator)()
    call_collection(pe.dag)
    vals = pe.dag.values
    branches = pe.branches
    vv = pe.vv
    m = length(branches)
    max_l = -Inf
    @inbounds for i in 1:m
        br = branches[i]
        lp = vals[br.dag_idx]
        pi_z = br.pi_z
        pi_k = br.pi_k
        for j in eachindex(pi_z)
            lp += safe_log(pi_z[j].Pi[pi_k[j]])
        end
        vv[i] = lp
        if lp > max_l
            max_l = lp
        end
    end
    s = 0.0
    @inbounds for i in 1:m
        e = exp(vv[i] - max_l)
        vv[i] = e
        s += e
    end
    inv_s = 1.0 / s
    PV = Dict{Vector{Int}, Float64}()
    top_prob = -1.0
    best_cond = pe.pi_labels[1]
    @inbounds for i in 1:m
        prob = vv[i] * inv_s
        label = pe.pi_labels[i]
        PV[label] = prob
        if prob > top_prob
            top_prob = prob
            best_cond = label
        end
    end
    return (max = best_cond, probability = PV)
end

function posterior_initalize(conditions, X::Vector{Observation}, density::Parsa_Base)
    for x in X
        prime_X(x)
    end
    domains = conditions
    all_domains = [GetDependentVariable(x) for x in X]
    X_sub_pairs = [(x, xf) for (x, xf) in zip(X, all_domains) if !isdisjoint(domains, xf)]
    X_full = Vector{Observation}()
    for (_, xf) in X_sub_pairs
        for lv in xf
            if !lv_isKnown(lv)
                append!(X_full, collect(lv.dependent_X))
            end
        end
    end
    X_sub = unique(X_full)
    if isempty(X_sub)
        X_sub = Observation[X[1]]
    elseif length(X_sub) > 1
        sort!(X_sub, by = objectid)
    end
    domain_map = Dict{Observation, Vector{LatentVaraible}}(x => unique(GetDependentVariable(x)) for x in X_sub)
    dag = DensityDAG(mutable_obs = Set{Observation}(X))
    independent_map = Dict{Any, Any}()
    branches = Vector{EStepBranch}()
    pi_labels = Vector{Vector{Int}}()
    _posterior_compile!(domains, X_sub, density, LatentVaraible[], domain_map, branches, pi_labels, dag, independent_map)
    vv = Vector{Float64}(undef, length(branches))
    return PosteriorEvaluator(dag, branches, pi_labels, vv)
end

function _posterior_compile!(
    domains,
    X::Vector{Observation},
    density::Parsa_Base,
    used_conditions::Vector{LatentVaraible},
    domain_map::Dict{Observation, Vector{LatentVaraible}},
    branches::Vector{EStepBranch},
    pi_labels::Vector{Vector{Int}},
    dag::DensityDAG,
    independent_map::Dict,
)
    domains_left = setdiff(domains, used_conditions)
    condition = domains_left[1]
    K = condition.value_ isa Unknown ? (1:condition.Z.K) : (lv_v(condition):lv_v(condition))
    next_used = LatentVaraible[used_conditions; condition]
    for k in K
        lv_set(condition, k)
        if length(domains_left) > 1
            _posterior_compile!(domains, X, density, next_used, domain_map, branches, pi_labels, dag, independent_map)
        else
            dag_idx = initialize_density_evaluation_idx(
                X,
                next_used,
                density,
                domain_map,
                dag,
                independent_map;
                should_eval = true,
            )
            push!(pi_labels, Int[lv_v(d) for d in domains])
            pi_z = CategoricalZ[d.Z for d in next_used]
            pi_k = Int[lv_v(d) for d in next_used]
            push!(branches, EStepBranch(dag_idx, pi_z, pi_k))
        end
        lv_set(condition, 0)
    end
    return nothing
end

function max_posterior(post)
    top_prob = -1.0
    cond = nothing
    for (k, v) in post
        if v > top_prob
            top_prob = v
            cond = k
        end
    end
    return cond
end

function max_posterior_val(post)
    top_prob = -1.0
    for (_, v) in post
        if v > top_prob
            top_prob = v
        end
    end
    return top_prob
end

function load_iris_dataset()
    path = joinpath(@__DIR__, "..", "examples/datasets")
    file = joinpath(path, "Iris.csv")
    iris = CSV.read(file, DataFrame)
    iris_matrix = Matrix{Float64}(iris[:, 2:5])
    class_string = vec(iris[:, 6])
    mapping = Dict(val => i for (i, val) in enumerate(unique(class_string)))
    class = Int[mapping[val] for val in class_string]
    return iris_matrix, class
end
