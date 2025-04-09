

mutable struct Unknown
    value::Int
end
Unknown() = Unknown(0)

mutable struct Parsa_index
    value :: Any
end

Base.@kwdef mutable struct CategoricalZset
    set = Dict()
end

Base.@kwdef mutable struct CategoricalZ
    K::Int
    LV = Dict{Any, LatentVaraible}()
    Pi::Vector{Real} = ones(K) ./ K
end


Base.@kwdef mutable struct LatentVaraible
    value_::Union{Int,Unknown}
    Z::CategoricalZ
    dependent_X = Dict{Any, Observation}()
end

Base.@kwdef mutable struct CategoricalZVec
    set::Vector{Union{CategoricalZ, LatentVaraible}}
end


lv_v(LV::LatentVaraible) = typeof(LV.value_) == Unknown ? LV.value_.value : LV.value_
lv_set(LV::LatentVaraible, x) = typeof(LV.value_) == Unknown ? LV.value_.value = x : nothing
lv_isKnown(LV::LatentVaraible) = typeof(LV.value_) == Unknown ? (lv_v(LV) == 0 ? false : true) : true

function get_possible_indexes(V::Vector{Any}, i::Int)

    L = length(V)
    next = V[i]
    if typeof(next) == Parsa_index
        next = [next]
    end
    ve = Vector{Any}(undef, 0)
    if i == L
        ve = [[n] for n in next]
    else
        for n in next
            ind_sets = get_possible_indexes(V, i+1)
            ve = [ve; [[n; ind] for ind in ind_sets]]
        end
    end

    return ve
end

function Base.getindex(PG::CategoricalZVec, indx...)
    indx = collect(indx)
    [typeof(P) == CategoricalZ ? P[indx...] : P for P in PG.set]
end

function Base.getindex(PG::CategoricalZ, indx...)
    indx = collect(indx)
    indx_new = []
    for ix in indx
        # indx_new = [indx_new; ix]
        if typeof(ix) == Vector{Int} || typeof(ix) == Int
            push!(indx_new, ix...)
        else
            push!(indx_new, ix)
        end
    end
    indx = indx_new
    V = Vector{Any}(undef, length(indx))
    for (i, ix) in enumerate(indx)
        if typeof(ix) == LatentVaraible

            if lv_v(ix) == 0
                V[i] = 1:ix.Z.K
            else
                V[i] = lv_v(ix)
            end
        elseif typeof(ix) == Vector{LatentVaraible}
            ma = -Inf
            for ix2 in ix
                mv = 0
                if lv_v(ix2) == 0
                    mv = ix2.Z.K
                else
                    mv = lv_v(ix2)
                end
                if mv > ma
                    ma = mv
                end
            end
            V[i] = 1:ma
        else
            V[i] = ix
        end
    end
    all_indx = get_possible_indexes(V, 1)
    LV = Vector{LatentVaraible}(undef, length(all_indx))
    for (i, v) in enumerate(all_indx)
        if haskey(PG.LV, v)
            LV[i] = PG.LV[v]
        else
            PG.LV[v] = LatentVaraible(PG)
            LV[i] = PG.LV[v]
        end
    end
    if length(LV) == 1
        return LV[1]
    else
        return LV
    end
end

function Base.getindex(PG::CategoricalZset, indx...)
    indx = collect(indx)
    V = Vector{Any}(undef, length(indx))
    extra_LV = Vector{LatentVaraible}()
    for (i, ix) in enumerate(indx)
        if typeof(ix) == LatentVaraible
            if lv_v(ix) == 0
                V[i] = 1:ix.Z.K
                extra_LV = [extra_LV; ix]
            else
                V[i] = lv_v(ix)
            end
        else
            V[i] = indx
        end
    end
    all_indx = get_possible_indexes(V, 1)
    # println(all_indx)
    LV = Vector{CategoricalZ}(undef, length(all_indx))
    for (i, v) in enumerate(all_indx)
        LV[i] = PG.set[v]
    end
    # println(length(LV))
    LV = [LV; extra_LV]
    # println(length(LV))
    if length(LV) == 1
        return LV[1]
    else
        return CategoricalZVec(LV)
    end
end



function Base.setindex!(Z::CategoricalZ, LV::LatentVaraible, i::Int64...)
    Z.LV[collect(i)] = LV
end

LatentVaraible(Z::CategoricalZ) = LatentVaraible(value_ = Unknown(), Z= Z)
LatentVaraible(Z::CategoricalZ, value::Int) = LatentVaraible(value_ = value, Z= Z)

function sampleZ(Z::CategoricalZ; value=nothing)
    if isnothing(value)
        LatentVaraible(Z)
    else
        LatentVaraible(Z, value)
    end
end

Base.@kwdef mutable struct ParameterValue
    value::Any
    n_parameters = length(value)
end

function compress_package(X, taus, parameter_maps)
    # map_new = [Dict([key => vv.value.value for (key,vv) in params]) for params in parameter_maps]
    map_new = Vector{Dict}(undef, length(parameter_maps))
    for i in eachindex(parameter_maps)
        map_new[i] = copy(parameter_maps[i])
        for (k, v) in map_new[i]
            map_new[i][k] = v.value.value
        end
    end
    x_new = [x.X for x in X]
    zip(x_new, taus, map_new)
end

Base.@kwdef mutable struct Parameter
    value::ParameterValue
    update::Function
    run_update::Function = (X, taus, parameter_maps, log_pdf) -> value.value = update(value.value, compress_package(X, taus, parameter_maps), log_pdf)
    is_const::Bool = false
    post_processing = nothing
end

Parsa_Parameter(initial_value, update_function::Function) = Parameter(value = ParameterValue(value = initial_value), update = update_function)
Parsa_Parameter(initial_value, number_of_parameters, update_function::Function) = Parameter(value = ParameterValue(value = initial_value, n_parameters = number_of_parameters), update = update_function)

Parsa_Parameter(initial_value, update_function::Function, post_processing::Function) = Parameter(value = ParameterValue(value = initial_value), update = update_function, post_processing=post_processing)
Parsa_Parameter(initial_value, number_of_parameters, update_function::Function, post_processing::Function) = Parameter(value = ParameterValue(value = initial_value, n_parameters = number_of_parameters), update = update_function,post_processing=post_processing)


Base.@kwdef mutable struct ParameterGenerator
    parameter_base :: Parameter
    parameter_map :: Dict = Dict()
end

function Base.getindex(PG::ParameterGenerator, indx...)
    indx = collect(indx)
    if size(indx) == ()
        indx = indx[1]
    end
    if typeof(indx) == Vector{Array{Int64, 0}}
        indx = [indx[1][1]]
    end
    if haskey(PG.parameter_map, indx)
        return PG.parameter_map[indx]
    else
        PG.parameter_map[indx] = deepcopy(PG.parameter_base)
        return PG.parameter_map[indx]
    end
end

function index_to_parameter_values(p, parameters)
    new = Dict()
    for (key, indx) in p
        V = indx
        if typeof(V) == LatentVaraible
            V = lv_v(indx)
        else
            V = [typeof(v) == LatentVaraible ? lv_v(v) : v for v in indx]
        end
        new[key] = parameters[key][V].value.value
    end
    return new
end

function index_to_parameters(p, parameters)
    new = Dict()
    for (key, indx) in p
        V = indx
        if typeof(V) == LatentVaraible
            V = lv_v(indx)
        else
            V = [typeof(v) == LatentVaraible ? lv_v(v) : v for v in indx]
        end
        new[key] = parameters[key][V]
    end
    return new
end

function parameter_map_to_values(p)
    new = Dict()
    for (key, val) in p
        new[key] = val.value.value
    end
    return new
end

Base.@kwdef mutable struct Parsa_Base
    pdf::Function
    log_pdf::Function
    parameters::Dict
    parameter_order :: Vector
    is_valid_input :: Function
    evaluate::Function = (X, p) -> pdf(X.X, index_to_parameter_values(p, parameters))
    # evaluate::Function = (X, p) -> pdf(X.X, p)
    eval_catch = Dict()
end


Parsa_density(pdf, log_pdf, is_valid_input, params...) = Parsa_Base(pdf = pdf, log_pdf = log_pdf, is_valid_input = is_valid_input,
                                                   parameter_order = [k for (k,_) in collect(params)],
                                                   parameters = Dict([key => ParameterGenerator(parameter_base = vv) for (key, vv) in Dict(params...)]))


struct T
    domain::Vector{Function}
    map::Function
end

mutable struct Observation
    X::Any
    T::T
end
