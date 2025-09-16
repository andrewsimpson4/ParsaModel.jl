

mutable struct Unknown
	value::Int
end
Unknown() = Unknown(0)

mutable struct Parsa_index
	value::Any
end

Base.@kwdef mutable struct CategoricalZset
	set = Dict()
end

Base.@kwdef mutable struct CategoricalZ
	K::Int
	LV = Dict{Any, LatentVaraible}()
	Pi::Vector{Real} = ones(K) ./ K
    name::String
	constant = false
	base = nothing
end


Base.@kwdef mutable struct LatentVaraible
	value_::Union{Int, Unknown}
	init_value = nothing
	Z::CategoricalZ
	dependent_X = Set{Observation}()
    index::Any
	active=false
end

Base.@kwdef mutable struct CategoricalZVec
	inside::Any
	outside::Any
end


lv_v(LV::LatentVaraible) = typeof(LV.value_) == Unknown ? LV.value_.value : LV.value_
# lv_set(LV::LatentVaraible, x) = typeof(LV.value_) == Unknown ? LV.value_.value = x : nothing
function lv_set(LV::LatentVaraible, x)
	if typeof(LV.value_) == Unknown
		LV.value_.value = x
	end
	if x == 0
		LV.active=false
	else
		LV.active=true
	end
end
lv_set_init(LV::LatentVaraible, x) = LV.init_value = x
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

function Base.display(LV::LatentVaraible)
    println(LV.Z.name * "[" * string(LV.index) * "]")
end

function Base.display(Z::CategoricalZ)
    println(Z.name)
end

struct LV_wrap
	LV::Any
end

function Base.getindex(x::LV_wrap, y::LV_wrap...)
	FF = function ()
		ne = x.LV()[[yy.LV() for yy in y]...]
		if isa(ne, LV_wrap)
			return ne.LV()
		else
			return ne
		end
	end
	return LV_wrap(FF)
end

function Base.getindex(x::LV_wrap, y::Int...)
	FF = function ()
		ne = x.LV()[y...]
		if isa(ne, LV_wrap)
			return ne.LV()
		else
			return ne
		end
	end
	LV_wrap(FF)
end

function Base.getindex(PG::CategoricalZ, indx...)
	LV_wrap(function ()
		indx_new = Vector{Int}(undef, length(indx))
		for (i, ix) in enumerate(indx)
			if isa(ix, LV_wrap)
				ix = ix.LV()
			end
			if typeof(ix) == LatentVaraible
				if !lv_isKnown(ix)
					return ix
				else
					indx_new[i] = lv_v(ix)
				end
			else
				if length(ix) == 1
					indx_new[i] = ix
				else
					for xx in ix
						indx_new[i] = xx
					end
				end
			end
		end
		ind = length(indx_new) == 1 ? indx_new[1] : indx_new
		LV = get(PG.LV, ind, nothing)
		if isnothing(LV)
			PG.LV[ind] = LatentVaraible(PG, ind)
			LV = PG.LV[ind]
		end
		return LV
	end)
end

Base.getindex(LV:: LatentVaraible, indx...) = LV


function Base.getindex(PG::CategoricalZset, indx...)
	LV_wrap(function()
		indx_new = Vector{Int}(undef, length(indx))
		for (i, ix) in enumerate(indx)
			if isa(ix, LV_wrap)
				ix = ix.LV()
			end
			if typeof(ix) == LatentVaraible
				if !lv_isKnown(ix)
					return ix
				else
					indx_new[i] = lv_v(ix)
				end
			else
				indx_new[i] = ix
			end
		end
		ind = length(indx_new) == 1 ? indx_new[1] : indx_new
		return PG.set[ind]
	end)
end



function Base.setindex!(Z::CategoricalZ, LV::LatentVaraible, i::Int64...)
	Z.LV[collect(i)] = LV
end

LatentVaraible(Z::CategoricalZ, index) = LatentVaraible(value_ = Unknown(), Z = Z, index=index)
LatentVaraible(Z::CategoricalZ, value::Int, index) = LatentVaraible(value_ = value, Z = Z,index=index)

Base.@kwdef mutable struct ParameterValue
	value::Any
	n_parameters = length(value)
end

function compress_package(X, taus, parameter_maps)
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
	run_update::Function = (X, taus, parameter_maps, log_pdf) -> value.value = update(value.value, zip(X, taus, parameter_maps), log_pdf)
	is_const::Bool = false
	post_processing = nothing
end

Parameter(initial_value, update_function::Function) = Parameter(value = ParameterValue(value = initial_value), update = update_function)
Parameter(initial_value, number_of_parameters, update_function::Function) = Parameter(value = ParameterValue(value = initial_value, n_parameters = number_of_parameters), update = update_function)

Parameter(initial_value, update_function::Function, post_processing::Function) = Parameter(value = ParameterValue(value = initial_value), update = update_function, post_processing = post_processing)
Parameter(initial_value, number_of_parameters, update_function::Function, post_processing::Function) =
	Parameter(value = ParameterValue(value = initial_value, n_parameters = number_of_parameters), update = update_function, post_processing = post_processing)


Base.@kwdef mutable struct ParameterGenerator
	parameter_base::Parameter
	parameter_map::Dict = Dict()
end

function Base.getindex(PG::ParameterGenerator, indx...)
	indx = collect(indx)
	# if size(indx) == ()
	# 	indx = indx[1]
	# end
	# if typeof(indx) == Vector{Array{Int64, 0}}
	# 	indx = [indx[1][1]]
	# end
	indx = length(indx) == 1 ? indx[1] : indx
	if haskey(PG.parameter_map, indx)
		return PG.parameter_map[indx]
	else
		PG.parameter_map[indx] = deepcopy(PG.parameter_base)
		return PG.parameter_map[indx]
	end
end

function index_to_parameters(p, parameters)
	new = Dict()
	for (key, indx) in p
		V = indx
		if !isa(indx, AbstractVector) && !isa(indx, Int)
			V = typeof(V) == LatentVaraible ? lv_v(V) : V

		elseif !isa(indx, Int)
			V = indx
			V = [typeof(v) == LatentVaraible ? lv_v(v) : v for v in V]
		end
		V = length(V) == 1 ? V[1] : V
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
	parameter_order::Vector
	is_valid_input::Function
	X = Set{Observation}()
	evaluate::Function = (X, p) -> pdf(X.X, p)
	eval_catch = Dict()
	full_likelihood = nothing
	n = 0
end


ParsaDensity(pdf, log_pdf, is_valid_input, params...) = Parsa_Base(pdf = pdf, log_pdf = log_pdf, is_valid_input = is_valid_input,
	parameter_order = [k for (k, _) in collect(params)],
	parameters = Dict([key => ParameterGenerator(parameter_base = vv) for (key, vv) in Dict(params...)]))

mutable struct Observation
	X::Any
	T::Any
	base::Any
end

Observation(x) = Observation(x, nothing, nothing)

function p_v(p::Parameter)
	p.value.value
end

function x_v(X::Observation)
	X.X
end


SummationPackage = Base.Iterators.Zip{Tuple{Vector{Observation}, Vector{Real}, Vector{Dict{Any, Any}}}}
