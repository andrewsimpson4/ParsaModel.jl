

macro |>(mod, code)
    Base.eval(mod, code)
end

categorical(K::Int; name="LV") = CategoricalZ(K = K, name=name)

function categorical(V::Vector{<:Pair}; name ="LV")
    set = CategoricalZset()
    for (ke, va) in V
        set.set[ke] = categorical(va, name=name)
    end
    return set
end

function categorical(V::Vector{Float64}; name ="LV")
    LV = CategoricalZ(K = length(V), name=name)
    LV.Pi = V
    LV.constant = true
    return LV
end
function categorical(V::Vector{Real}; name ="LV")
    LV = CategoricalZ(K = length(V), name=name)
    LV.Pi = V
    return LV
end

function (PB::Parsa_Base)(x...)
	return (PB, collect(x))
end

function (PB::Parsa_Base)(x::Vector{<:Pair})
	return (PB, x)
end

function Base.:~(X::Observation, map::Any)
    X.base = map[1]
    X.T = map[2]
    push!(map[1].X, X)
    domains = GetDependentVariable(X)
    for LV in domains
        if !isnothing(LV.Z.base)
            if LV.Z.base != map[1]
                error("Z already associated with a base distribution")
            end
        end
        LV.Z.base = map[1]
    end
    return nothing
end

function <--(LV::LV_wrap, val::Int)
    lv_set_init(LV.LV(), val)
    return nothing
end

# function <|(LV::LV_wrap, val::Int)
#     LV.LV().value_ = val
# end

function <--(P::Parameter, val::Any)
   P.value.value = val
   return nothing
end

# function <|(P::Parameter, val::Any)
#     P.value.value = val
#     P.is_const = true
#     return nothing
# end

function Base.setindex!(PG::ParameterGenerator, val, indx)
    PG[indx].value.value = val
    PG[indx].is_const = true
    return nothing
end

function Base.setindex!(Z::CategoricalZ, val, indx)
    Z[indx].LV().value_ = val
end

function Base.setindex!(Z::LV_wrap, val, indx)
    Z.LV()[indx].LV().value_ = val
end

function f(X::Observation...)
    X = collect(X)
    map_collector = OrderedDict()
    independent_map = Dict()
    domain_map = Dict([x => unique([LV for LV in GetDependentVariable(x) if !lv_isKnown(LV)]) for x in X])
    tt = initialize_density_evaluation(X, Vector{}(), X[1].base, domain_map, map_collector, independent_map; should_eval=true)
    function()
        call_collection(map_collector)
        tt()
    end
end

function f(LV::LV_wrap...)
    LVs = [lv.LV() for lv in LV]
    prime_X.(LVs[1].Z.base.X)
    all_X = unique(reduce(vcat, [collect(LV.dependent_X) for LV in LVs]))
    posterior_initalize(LVs, all_X, all_X[1].base)
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
    local to_return = Dict()
    for (key, val) in PB.parameters[sym].parameter_map
        to_return[key] = val
    end
    return to_return
end

function val(P::ParameterGenerator)
    local to_return = Dict()
    for (key, val) in P.parameter_map
        to_return[key] = val.value.value
    end
    return to_return
end
val(P::Parameter) = P.value.value
val(P::CategoricalZ) = P.Pi
function val(P::CategoricalZset)
    local to_return = Dict()
    for (key, val_) in P.set
        to_return[key] = val(val_)
    end
    return to_return
end
function val(X::Observation)
	X.X
end

function n_params(model)
    M::Int = 0
    for (_, gen) in model.parameters
        for (_, par) in gen.parameter_map
            if !par.is_const
                M = M + par.value.n_parameters
            end
        end
    end
    return M
end

function BIC(model)
    log_lik = model.full_likelihood()
    M = n_params(model)
    Float64(M * log(model.n) - 2 * log_lik)
end


# function Base.getindex(PG::ParameterGenerator, indx...)
#     PG.parameter_map[indx...].value.value
# end

# function BIC(model)
#     log_lik = model.fit_model.log_likelihood()
#     local M = 0
#     for (_, gen) in model.base_model.parameters
#         for (_, par) in gen.parameter_map
#             if !par.is_const
#                 M = M + par.value.n_parameters
#             end
#         end
#     end
#     Float64(M * log(model.fit_model.n) - 2 * log_lik)
# end