# High-level DSL notation and user-facing API for ParsaModel.jl

macro |>(mod, code)
    Base.eval(mod, code)
end

categorical(K::Int; name::String = "LV") = CategoricalZ(K = K, name = name)

function categorical(V::Vector{<:Pair}; name::String = "LV")
    set = CategoricalZset()
    for (ke, va) in V
        set.set[ke] = categorical(va, name = name)
    end
    return set
end

function categorical(V::Vector{Float64}; name::String = "LV")
    lv = CategoricalZ(K = length(V), Pi = V, name = name, constant = true)
    return lv
end

function categorical(V::Vector{Real}; name::String = "LV")
    lv = CategoricalZ(K = length(V), Pi = Float64.(V), name = name)
    return lv
end

function (PB::Parsa_Base)(x...)
    return (PB, collect(x))
end

function (PB::Parsa_Base)(x::Vector{<:Pair})
    return (PB, x)
end

function Base.:~(X::Observation, map::Any)
    base = map[1]
    X.base = base
    X.T = map[2]
    push!(base.X, X)
    domains = GetDependentVariable(X)
    for lv in domains
        if !isnothing(lv.Z.base) && lv.Z.base != base
            error("Z already associated with a base distribution")
        end
        lv.Z.base = base
    end
    return nothing
end

function <--(lv::LV_wrap, v::Int)
    lv_set_init(resolve_lv(lv), v)
    return nothing
end

function <--(P::Parameter, v::Any)
    P.value.value = v
    return nothing
end

function Base.setindex!(PG::ParameterGenerator, v, indx)
    param = PG[indx]
    param.value.value = v
    param.is_const = true
    return nothing
end

function Base.setindex!(Z::CategoricalZ, v, indx)
    resolve_lv(Z[indx]).value_ = v
    return nothing
end

function Base.setindex!(Z::LV_wrap, v, indx)
    resolve_lv(resolve_lv(Z)[indx]).value_ = v
    return nothing
end

struct ObservationLikelihoodEvaluator
    dag::DensityDAG
    ref::DAGNodeRef
end

function (ole::ObservationLikelihoodEvaluator)()
    call_collection(ole.dag)
    return ole.ref()
end

function f(X::Observation...)
    X_vec = collect(Observation, X)
    for x in X_vec
        prime_X(x)
    end
    dag = DensityDAG(mutable_obs = Set{Observation}(X_vec))
    independent_map = Dict{Any, Any}()
    domain_map = Dict{Observation, Vector{LatentVaraible}}(x => unique(GetDependentVariable(x)) for x in X_vec)
    ref = initialize_density_evaluation(X_vec, LatentVaraible[], X_vec[1].base, domain_map, dag, independent_map; should_eval = true)
    return ObservationLikelihoodEvaluator(dag, ref)
end

function f(LV::LV_wrap...)
    LVs = LatentVaraible[resolve_lv(lv) for lv in LV]
    for x in LVs[1].Z.base.X
        prime_X(x)
    end
    all_X = Vector{Observation}()
    for lv in LVs
        for x in lv.dependent_X
            if !(x in all_X)
                push!(all_X, x)
            end
        end
    end
    return posterior_initalize(LVs, all_X, all_X[1].base)
end

function EM!(PB::Parsa_Base; args...)
    l1, l2, flag = LMEM(PB.X, PB; args...)
    PB.full_likelihood = l1
    PB.n = l2
    if flag
        return (descresing_likelihood = true,)
    else
        return nothing
    end
end

function Base.getindex(PB::Parsa_Base, sym::Symbol)
    return PB.parameters[sym]
end

function n_params(model::Parsa_Base)
    M = 0
    for (_, gen) in model.parameters
        for (_, par) in gen.parameter_map
            if !par.is_const
                M += Int(par.value.n_parameters)
            end
        end
    end
    X_vec = collect(model.X)
    all_Z = _collect_all_Z(X_vec)
    for Z in all_Z
        if !Z.constant
            M += length(Z.Pi) - 1
        end
    end
    return M
end

function BIC(model::Parsa_Base)
    log_lik = model.full_likelihood()
    M = n_params(model)
    return Float64(M * log(model.n) - 2 * log_lik)
end

