# Core data structures and type definitions for ParsaModel.jl

mutable struct Unknown
    value::Int
end
Unknown() = Unknown(0)

mutable struct Parsa_index
    value::Any
end

mutable struct Observation
    X::Any
    T::Any
    base::Any
end

(obs::Observation)(x) = (obs.X = x)

function Observation(x::AbstractVector{<:Real})
    return Observation(x isa Vector{Float64} ? x : collect(Float64, x), nothing, nothing)
end
Observation(x) = Observation(x, nothing, nothing)
Observation() = Observation(nothing, nothing, nothing)

Base.@kwdef mutable struct CategoricalZset
    set::Dict{Any, Any} = Dict{Any, Any}()
end

mutable struct CategoricalZ
    K::Int
    LV::Dict{Any, Any}
    Pi::Vector{Float64}
    pi_acc::Vector{Float64}
    name::String
    constant::Bool
    base::Any
end

function CategoricalZ(;
    K::Int,
    LV::Dict{Any, Any} = Dict{Any, Any}(),
    Pi::AbstractVector{<:Real} = fill(1.0 / K, K),
    name::String = "LV",
    constant::Bool = false,
    base::Any = nothing,
)
    pi_vec = Pi isa Vector{Float64} ? Pi : collect(Float64, Pi)
    return CategoricalZ(K, LV, pi_vec, zeros(Float64, K), name, constant, base)
end

Base.@kwdef mutable struct LatentVaraible
    value_::Union{Int, Unknown}
    init_value::Union{Nothing, Int} = nothing
    Z::CategoricalZ
    dependent_X::Set{Observation} = Set{Observation}()
    index::Any
    active::Bool = false
    is_index::Bool = false
end

const LatentVariable = LatentVaraible

LatentVaraible(Z::CategoricalZ, index) = LatentVaraible(value_ = Unknown(), Z = Z, index = index)
LatentVaraible(Z::CategoricalZ, value::Int, index) = LatentVaraible(value_ = value, Z = Z, index = index)

Base.@kwdef mutable struct CategoricalZVec
    inside::Any
    outside::Any
end

@inline lv_v(lv::LatentVaraible)::Int = lv.value_ isa Unknown ? lv.value_.value : lv.value_

@inline function lv_set(lv::LatentVaraible, x::Int)
    if lv.value_ isa Unknown
        lv.value_.value = x
    end
    lv.active = (x != 0)
    return nothing
end

@inline function lv_set_init(lv::LatentVaraible, x::Int)
    lv.init_value = x
    return nothing
end

@inline function lv_isKnown(lv::LatentVaraible)::Bool
    return lv.value_ isa Unknown ? (lv.value_.value != 0) : true
end

function get_possible_indexes(V::Vector{Any}, i::Int)
    L = length(V)
    next = V[i]
    if next isa Parsa_index
        next = [next]
    end
    if i == L
        return [[n] for n in next]
    end
    ve = Vector{Any}()
    ind_sets = get_possible_indexes(V, i + 1)
    for n in next
        for ind in ind_sets
            push!(ve, [n; ind])
        end
    end
    return ve
end

function Base.display(lv::LatentVaraible)
    println(lv.Z.name * "[" * string(lv.index) * "]")
end

function Base.display(Z::CategoricalZ)
    println(Z.name)
end

"""
    LV_wrap{T, I<:Tuple}

Lightweight, allocation-free lazy index expression over a categorical distribution
(`CategoricalZ`, `CategoricalZset`, or nested `LV_wrap`).
"""
struct LV_wrap{T, I<:Tuple}
    target::T
    indices::I
end

LV_wrap(fn::Function) = LV_wrap(fn, ())

@inline function Base.getproperty(w::LV_wrap, s::Symbol)
    if s === :LV
        return w
    end
    return getfield(w, s)
end

@inline (w::LV_wrap)() = resolve_lv(w)

@inline resolve_lv(w::LV_wrap{<:Function, Tuple{}}) = w.target()

@inline function _extract_single_index(ix)
    if ix isa LV_wrap
        ix = resolve_lv(ix)
    end
    if ix isa LatentVaraible
        ix.is_index = true
        if !lv_isKnown(ix)
            return ix
        else
            return lv_v(ix)
        end
    elseif ix isa Int
        return ix
    elseif length(ix) == 1
        return Int(first(ix))
    else
        return Int(last(ix))
    end
end

function _lookup_categorical_z(PG::CategoricalZ, ind)
    lv = get(PG.LV, ind, nothing)
    if isnothing(lv)
        lv = LatentVaraible(PG, ind)
        PG.LV[ind] = lv
    end
    return lv::LatentVaraible
end

function _resolve_target(PG::CategoricalZ, indices::Tuple)
    n = length(indices)
    if n == 1
        v = _extract_single_index(indices[1])
        v isa LatentVaraible && return v
        return _lookup_categorical_z(PG, v::Int)
    elseif n == 2
        v1 = _extract_single_index(indices[1])
        v1 isa LatentVaraible && return v1
        v2 = _extract_single_index(indices[2])
        v2 isa LatentVaraible && return v2
        tkey = (v1::Int, v2::Int)
        lv = get(PG.LV, tkey, nothing)
        if !isnothing(lv)
            return lv::LatentVaraible
        end
        vkey = Int[v1::Int, v2::Int]
        lv = _lookup_categorical_z(PG, vkey)
        PG.LV[tkey] = lv
        return lv
    else
        for i in 1:n
            v = _extract_single_index(indices[i])
            v isa LatentVaraible && return v
        end
        indx_new = Vector{Int}(undef, n)
        @inbounds for i in 1:n
            indx_new[i] = _extract_single_index(indices[i])::Int
        end
        return _lookup_categorical_z(PG, indx_new)
    end
end

function _resolve_target(PG::CategoricalZset, indices::Tuple)
    n = length(indices)
    if n == 1
        v = _extract_single_index(indices[1])
        v isa LatentVaraible && return v
        return PG.set[v::Int]
    else
        for i in 1:n
            v = _extract_single_index(indices[i])
            v isa LatentVaraible && return v
        end
        indx_new = Vector{Int}(undef, n)
        @inbounds for i in 1:n
            indx_new[i] = _extract_single_index(indices[i])::Int
        end
        return PG.set[indx_new]
    end
end

@inline _resolve_target(lv::LatentVaraible, ::Tuple) = lv

function resolve_lv(w::LV_wrap)
    t = w.target
    if t isa LV_wrap
        inner = resolve_lv(t)
        inner isa LatentVaraible && return inner
        res = _resolve_target(inner, w.indices)
        return res isa LV_wrap ? resolve_lv(res) : res
    else
        res = _resolve_target(t, w.indices)
        return res isa LV_wrap ? resolve_lv(res) : res
    end
end

Base.getindex(x::LV_wrap, y::LV_wrap...) = LV_wrap(x, y)
Base.getindex(x::LV_wrap, y::Int...) = LV_wrap(x, y)
Base.getindex(PG::CategoricalZ, indx...) = LV_wrap(PG, indx)
Base.getindex(lv::LatentVaraible, indx...) = lv
Base.getindex(PG::CategoricalZset, indx...) = LV_wrap(PG, indx)

function Base.setindex!(Z::CategoricalZ, lv::LatentVaraible, i::Int64...)
    Z.LV[collect(i)] = lv
end

_normalize_param_val(x::Int) = Float64(x)
_normalize_param_val(x::AbstractVector{<:Real}) = x isa Vector{Float64} ? x : collect(Float64, x)
_normalize_param_val(x::AbstractMatrix{<:Real}) = x isa Matrix{Float64} ? x : Matrix{Float64}(x)
_normalize_param_val(nt::NamedTuple{(:inv, :det)}) = (inv = _normalize_param_val(nt.inv), det = Float64(nt.det))
_normalize_param_val(x) = x

mutable struct ParameterValue{T}
    value::T
    n_parameters::Int
end

function ParameterValue(; value, n_parameters::Real = length(value))
    norm_val = _normalize_param_val(value)
    return ParameterValue{typeof(norm_val)}(norm_val, Int(n_parameters))
end

function compress_package(X, taus, parameter_maps)
    map_new = Vector{Dict{Symbol, Any}}(undef, length(parameter_maps))
    for i in eachindex(parameter_maps)
        d = Dict{Symbol, Any}()
        for (k, v) in pairs(parameter_maps[i])
            if k !== :index_key
                d[k] = v.value.value
            end
        end
        map_new[i] = d
    end
    x_new = [x.X for x in X]
    return zip(x_new, taus, map_new)
end

struct DefaultRunUpdate{T, U<:Function} <: Function
    value::ParameterValue{T}
    update::U
end

@inline function (r::DefaultRunUpdate{T, U})(X, taus, parameter_maps, log_pdf) where {T, U}
    r.value.value = r.update(r.value.value, zip(X, taus, parameter_maps), log_pdf)
    return r.value.value
end

mutable struct Parameter{T, U<:Function, R<:Function}
    value::ParameterValue{T}
    update::U
    run_update::R
    is_const::Bool
    post_processing::Any
    value_original::Any
end

function Parameter(;
    value::ParameterValue{T},
    update::U,
    run_update::R = DefaultRunUpdate(value, update),
    is_const::Bool = false,
    post_processing::Any = nothing,
    value_original::Any = nothing,
) where {T, U<:Function, R<:Function}
    return Parameter{T, U, R}(value, update, run_update, is_const, post_processing, value_original)
end

Parameter(initial_value, update_function::Function) =
    Parameter(value = ParameterValue(value = initial_value), update = update_function)

Parameter(initial_value, number_of_parameters, update_function::Function) =
    Parameter(value = ParameterValue(value = initial_value, n_parameters = Int(number_of_parameters)), update = update_function)

Parameter(initial_value, update_function::Function, post_processing::Function) =
    Parameter(value = ParameterValue(value = initial_value), update = update_function, post_processing = post_processing)

Parameter(initial_value, number_of_parameters, update_function::Function, post_processing::Function) =
    Parameter(value = ParameterValue(value = initial_value, n_parameters = Int(number_of_parameters)), update = update_function, post_processing = post_processing)

Base.@kwdef mutable struct ParameterGenerator
    parameter_base::Parameter
    parameter_map::Dict{Any, Parameter} = Dict{Any, Parameter}()
end

function Base.getindex(PG::ParameterGenerator, indx...)
    key = length(indx) == 1 ? indx[1] : collect(indx)
    p = get(PG.parameter_map, key, nothing)
    if isnothing(p)
        p = deepcopy(PG.parameter_base)
        PG.parameter_map[key] = p
    end
    return p
end

function index_to_parameters(p, parameters::Dict{Symbol, ParameterGenerator})
    ks = sort!(collect(Symbol, keys(p)))
    n = length(ks)
    vals = Vector{Any}(undef, n + 1)
    h = UInt64(0x9e3779b97f4a7c15)
    for (i, key) in enumerate(ks)
        indx = p[key]
        if indx isa Int
            V = indx
        elseif indx isa LatentVaraible
            V = lv_v(indx)
        elseif indx isa AbstractVector
            if length(indx) == 1
                v1 = indx[1]
                V = v1 isa LatentVaraible ? lv_v(v1) : v1
            else
                V = Int[v isa LatentVaraible ? lv_v(v) : Int(v) for v in indx]
            end
        else
            V = length(indx) == 1 ? indx[1] : indx
        end
        vals[i] = parameters[key][V]
        h = hash(key, hash(V, h))
    end
    vals[n + 1] = h
    names = (ks..., :index_key)
    return NamedTuple{names}(Tuple(vals))
end

function parameter_map_to_values(p)
    new = Dict{Symbol, Any}()
    for (key, v) in pairs(p)
        if key !== :index_key
            new[key] = v.value.value
        end
    end
    return new
end

struct DefaultEvaluate{F<:Function} <: Function
    pdf::F
end

@inline (e::DefaultEvaluate{F})(X::Observation, p) where {F} = e.pdf(X.X, p)

Base.@kwdef mutable struct Parsa_Base
    pdf::Function
    log_pdf::Function
    parameters::Dict{Symbol, ParameterGenerator}
    parameter_order::Vector{Symbol}
    is_valid_input::Function
    X::Set{Observation} = Set{Observation}()
    evaluate::Function = DefaultEvaluate(pdf)
    eval_catch::Dict{Any, Any} = Dict{Any, Any}()
    full_likelihood::Any = nothing
    n::Int = 0
end

function ParsaDensity(pdf, log_pdf, is_valid_input, params...)
    param_order = Symbol[k for (k, _) in params]
    param_dict = Dict{Symbol, ParameterGenerator}(k => ParameterGenerator(parameter_base = v) for (k, v) in params)
    return Parsa_Base(
        pdf = pdf,
        log_pdf = log_pdf,
        is_valid_input = is_valid_input,
        parameter_order = param_order,
        parameters = param_dict,
    )
end

@inline p_v(p::Parameter) = p.value.value
@inline x_v(X::Observation) = X.X

@inline val(P::Parameter) = P.value.value
@inline val(P::CategoricalZ) = P.Pi
@inline val(X::Observation) = X.X

function val(P::ParameterGenerator)
    to_return = Dict{Any, Any}()
    for (key, v) in P.parameter_map
        to_return[key] = v.value.value
    end
    return to_return
end

function val(P::CategoricalZset)
    to_return = Dict{Any, Any}()
    for (key, v) in P.set
        to_return[key] = val(v)
    end
    return to_return
end

const SummationPackage = Base.Iterators.Zip{<:Tuple{AbstractVector{Observation}, AbstractVector{<:Real}, AbstractVector}}
