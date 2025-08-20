

categorical(K::Int) = CategoricalZ(K = K, name="LV")

function categorical(V::Vector{<:Pair})
    set = CategoricalZset()
    for (ke, va) in V
        set.set[ke] = categorical(va)
    end
    return set
end

function categorical(V::Vector{Real})
    LV = CategoricalZ(K = length(V), name="LV")
    LV.Pi = V
    return LV
end

function (PB::Parsa_Base)(x...)
	return (PB, collect(x))
end

function ~(X::Observation, map::Any)
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
    initialize_density_evaluation(X, Vector{}(), X[1].base, Dict(), Dict())
end

function f(LV::LV_wrap...)
    LVs = [lv.LV() for lv in LV]
    prime_X.(LVs[1].Z.base.X)
    all_X = collect(union(reduce(vcat, [LV.dependent_X for LV in LVs])))
    posterior_initalize(LVs, all_X, all_X[1].base)
end

function EM!(PB::Parsa_Base; args...)
    LMEM(PB.X, PB; args...)
    return nothing
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

# function Base.getindex(PG::ParameterGenerator, indx...)
#     PG.parameter_map[indx...].value.value
# end

