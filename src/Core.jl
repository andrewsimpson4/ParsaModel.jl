using UnicodePlots
using Distributions
using OrderedCollections
using Base: IdSet

function getRelaventTaus(parameter::Parameter, X::Vector{Observation}, tau::Vector{Vector{Real}}, parameter_map::Vector{Vector{Any}})
	reduce(vcat, [[(x, pr, param) for (pr, param) in zip(tau_x, params_x) if parameter in values(param)] for (x, tau_x, params_x) in zip(X, tau, parameter_map)])
end

function getRelaventTausIndex(parameter::Parameter, X::Vector{Observation}, tau::Vector{Vector{Real}}, parameter_map::Vector{Vector{Any}})
	indx = Vector{Tuple}()
	i_1 = 1
	for params_x in parameter_map
		i_2 = 1
		for params in params_x
			if parameter in values(params)
				push!(indx, (i_1, i_2))
			end
			i_2 += 1
		end
		i_1 += 1
	end
	return indx
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

# function getIndependentSets(X::Vector{Observation}, Z, domain_map)
# 	domain_map = Dict()
# 	for x in X
# 			domain_map[x] = GetDependentVariable(x)
# 	end
# 	n = length(X)
# 	# if length(domain_map) == 0
# 	# 	for x in X
# 	# 		domain_map[x] = GetDependentVariable(x)
# 	# 	end
# 	# end
# 	indo_sets = Vector{Any}(undef, n)
# 	for i in 1:n
# 		sets = unique([X[i]; conditional_dependent_search(domain_map[X[i]], domain_map, Z, X)])
# 		indo_sets[i] = sets[sortperm([objectid(x) for x in sets])]
# 	end
# 	return unique(indo_sets)
# end

# function conditional_dependent_search(D, domain_map, Z, X)
# 	x_depo = Vector{Observation}()
# 	new_d = setdiff(D, Z)
# 	for d in new_d
# 		if !lv_isKnown(d)
# 			d_depo = collect(values(d.dependent_X))
# 			new_Z = setdiff(reduce(vcat, [domain_map[x] for x in d_depo if x in X]), [D; Z])
# 			if length(new_Z) > 0
# 				d_depo = [d_depo; conditional_dependent_search(new_Z, domain_map, [Z; D], X)]
# 			end
# 			for x in d_depo
# 				if x in X
# 					x_depo = [x_depo; x]
# 				end
# 			end
# 		end
# 	end
# 	return x_depo
# end

# function getIndependentSets(X::Vector{Observation})
# 	domains = Dict([x => [lv for lv in GetDependentVariable(x) if !lv_isKnown(lv)] for x in X])
# 	indo_sets = Dict([x => IdSet() for x in X])
# 	indo_sets_obs = Vector{Vector{Observation}}(undef, length(X))
# 	for x in X
# 		push!(indo_sets[x], x)
# 		for condition in domains[x]
# 				for va in condition.dependent_X
# 					if !(va in indo_sets[x]) && va in X && condition in domains[va]
# 						push!(indo_sets[x], va)
# 					end
# 				end
# 		end
# 	end

# 	for (i, x) in enumerate(X)
# 		for y in indo_sets[x]
# 			conditional_dependent_search!(y, indo_sets, indo_sets[x], X)
# 		end
# 		collected_x = collect(indo_sets[x])
# 		indo_sets_obs[i] = collected_x[sortperm([objectid(x) for x in collected_x])]
# 	end
# 	return unique(indo_sets_obs)
# end

# function conditional_dependent_search!(x, indo_sets, set, X)
# 	for y in indo_sets[x]
# 		if !(y in set) && y in X
# 			push!(set, y)
# 			conditional_dependent_search!(y, indo_sets, set, X)
# 		end
# 	end
# end


function getIndependentSets(X::Vector{Observation}, conditions, independent_map, domain_map)
	magic_key = (X, conditions)
	if haskey(independent_map, magic_key)
		return independent_map[magic_key]
	end
	# domains = Dict([x => Set([lv for lv in setdiff(domain_map[x], conditions) if !lv_isKnown(lv)]) for x in X])
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
	independent_map[magic_key] = independent_sets
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

function conditional_dependent_search!(x, indo_sets, set, X)
	for y in indo_sets[x]
		if !(y in set) #&& y in X
			push!(set, y)
			conditional_dependent_search!(y, indo_sets, set, X)
		end
	end
end


function getDependentOnX(X::Vector{Observation})
	domains = Dict([x => [lv for lv in GetDependentVariable(x) if !lv_isKnown(lv)] for x in X])
	indo_sets = Dict([x => Set{Observation}() for x in X])
	for x in X
		push!(indo_sets[x], x)
		for condition in domains[x]
			for va in condition.dependent_X
				if !(va in indo_sets[x])
					push!(indo_sets[x], va)
				end
			end
		end
	end
	# return indo_sets
	for (i, x) in enumerate(X)
		for y in indo_sets[x]
			conditional_dependent_search!(y, indo_sets, indo_sets[x], X)
		end
	end
	return indo_sets
end


# function getDependentOnX(X::Vector{Observation})
# 	indo_sets = getIndependentSets(X)
# 	println("#####")
# 	println(length(indo_sets))
# 	println("#####")
# 	dependent_vec = Vector{}()
# 	for (i, x) in enumerate(X)
# 		for set in indo_sets
# 			if x in set
# 				push!(dependent_vec, set)
# 				break
# 			end
# 		end
# 	end
# 	return dependent_vec
# end

# function initialize_density_evaluation(X::Vector{Observation}, density::Parsa_Base, independent_by)
# 	if length(independent_by) == 0
# 		D = Vector{Any}()
# 		return (initialize_density_evaluation(X, D, density))
# 	else
# 		D = Vector{Any}()
# 		return (initialize_density_evaluation_ind(X, D, density, independent_by))
# 	end
# end

# function initialize_density_evaluation(X::Vector{Observation}, conditioned_domains::Vector, density::Parsa_Base, independent_by)
# 	if length(independent_by) == 0
# 		D = Vector{Any}()
# 		return (initialize_density_evaluation(X, conditioned_domains, density))
# 	else
# 		D = Vector{Any}()
# 		return (initialize_density_evaluation_ind(X, conditioned_domains, density, independent_by))
# 	end
# end

struct EVAL{F}
	f::F
	is_eval::Bool
end
(f::EVAL)() = f.f()
prod_foldl(G) = foldl((a, g) -> a * g(), G; init=1)
sum_foldl(G) = foldl((a, g) -> a + g(), G; init=0)

Base.@kwdef mutable struct HOLDER
	f::EVAL
	is_eval::Bool
	val::BigFloat = 0.0
end

function initialize_density_evaluation(X::Vector{Observation}, conditioned_domains::Vector, density::Parsa_Base, domain_map::Dict, map_collector::OrderedDict, independent_map::Dict; should_eval=false)
	# println(length(conditioned_domains))
	# relavent_conditions = intersect(relavent_conditions, conditioned_domains)
	relavent_conditions = intersect(reduce(vcat, [domain_map[x] for x in X]), conditioned_domains)
	relavent_conditions_id = [objectid(r) for r in relavent_conditions]
	conditions_key = [(LV, lv_v(LV)) for LV in relavent_conditions[sortperm(relavent_conditions_id)]]
	# conditions_key = [(LV, lv_v(LV)) for LV in relavent_conditions]
	magic_key = (X, conditions_key)
	if haskey(map_collector, magic_key)
		# println("here")
		return EVAL(() -> map_collector[magic_key].val, map_collector[magic_key].is_eval)
		# return map_collector[(X, all_domains_key)]
	end
	independent_sets = getIndependentSets(X, relavent_conditions, independent_map, domain_map)
	# println(length(independent_sets))
	# println(length(X))
	# println("----")
	# sleep(0.1)
	mult_list = Vector{EVAL}()
	for G in independent_sets
		all_domains = Vector{Vector{LatentVaraible}}(undef, length(G))
		for (i, x) in enumerate(G)
			# all_domains[i] = setdiff(domain_map[x], conditioned_domains) #GetDependentVariable(x)
			all_domains[i] = GetDependentVariable(x) #GetDependentVariable(x)
		end
		# all_domains = [GetDependentVariable(x) for x in G] #[unique([lv for lv in GetDependentVariable(x)]) for x in G]
		domain_lengths = [length(d) for d in all_domains]
		domains = [LV for LV in (reduce(vcat, all_domains))]
		lv_freq_map = countmap(domains)
		if maximum(domain_lengths; init=0) <= maximum(values(lv_freq_map); init=0)
			next_conditions = domains #setdiff(domains, conditioned_domains)
			lv_freq_map = filter(x -> x[1] in next_conditions, lv_freq_map)
			top_order = sortperm(collect(values(lv_freq_map)); rev=true)
			next_conditions = collect(keys(lv_freq_map))[top_order]
		else
			next_conditions = all_domains[argmax(domain_lengths)]
		end
		if length(next_conditions) != 0
			next_condition = next_conditions[1]
			K = typeof(next_condition.value_) == Unknown ? (1:next_condition.Z.K) : lv_v(next_condition)
			sum_list = Vector{EVAL}(undef, length(K))
			for (i_k, k) in enumerate(K)
				lv_set(next_condition, k)
				new_conditions = [conditioned_domains; next_condition]
				lik_new = initialize_density_evaluation(G, new_conditions, density, domain_map, map_collector, independent_map; should_eval = should_eval)
				pi_c = () -> (typeof(next_condition.value_) == Unknown ? next_condition.Z.Pi[k] : 1)
				sum_list[i_k] = EVAL(() -> (pi_c() * lik_new()), false)
				# if lik_new.is_eval && should_eval
				# 	val = (pi_c() * lik_new())
				# 	sum_list[i_k] = EVAL(() -> val, true)
				# else
				# 	sum_list[i_k] = EVAL(() -> (pi_c() * lik_new()), false)
				# end
				lv_set(next_condition, 0)
			end
			sum_list = Tuple(sum_list)
			if should_eval && sum([m.is_eval for m in sum_list]) == length(sum_list)
				val = sum_foldl(sum_list)
				push!(mult_list, EVAL(() -> val, true))
			else
				push!(mult_list, EVAL(() -> sum_foldl(sum_list), false))
			end
		else
			for g in G
				ma = rawCallDomain(g.T)
				mm = index_to_parameters(ma, density.parameters)
				if should_eval && !isnothing(g.X)
					val = BigFloat(density.evaluate(g,  mm))
					push!(mult_list, EVAL(() -> val, true))
				else
					push!(mult_list, EVAL(() -> BigFloat(density.evaluate(g,  mm)), false))
					# push!(mult_list, EVAL(() -> (println("do eval"); BigFloat(density.evaluate(g,  mm))), false))
				end
			end

		end

	end
	mult_list = Tuple(mult_list)
	if should_eval && sum([m.is_eval for m in mult_list]) == length(mult_list)
		val = prod_foldl(mult_list)
		map_collector[magic_key] = HOLDER(f = EVAL(() -> val, true), is_eval=true)
		return EVAL(() -> map_collector[magic_key].val, true)
	else
		map_collector[magic_key] = HOLDER(f = EVAL(() -> prod_foldl(mult_list), false), is_eval=false)
		return EVAL(() -> map_collector[magic_key].val, false)
	end
end


# function initialize_density_evaluation(X::Vector{Observation}, conditioned_domains::Vector, density::Parsa_Base, domain_map::Dict, map_collector::Dict)
# 	independent_sets = getIndependentSets(X)
# 	mult_list = Vector{Function}()
# 	for G in independent_sets
# 		all_domains = [unique([lv for lv in GetDependentVariable(x)]) for x in G]
# 		domain_lengths = [length(d) for d in all_domains]
# 		domains = [LV for LV in (reduce(vcat, all_domains))]
# 		lv_freq_map = countmap(domains)
# 		if maximum(domain_lengths; init=0) <= maximum(values(lv_freq_map); init=0)
# 			next_conditions = domains #setdiff(domains, conditioned_domains)
# 			lv_freq_map = filter(x -> x[1] in next_conditions, lv_freq_map)
# 			top_order = sortperm(collect(values(lv_freq_map)); rev=true)
# 			next_conditions = collect(keys(lv_freq_map))[top_order]
# 		else
# 			next_conditions = all_domains[argmax(domain_lengths)]
# 		end
# 		if length(next_conditions) != 0
# 			next_condition = next_conditions[1]
# 			K = typeof(next_condition.value_) == Unknown ? (1:next_condition.Z.K) : lv_v(next_condition)
# 			sum_list = Vector{Function}(undef, length(K))
# 			for (i_k, k) in enumerate(K)
# 				lv_set(next_condition, k)
# 				new_conditions = [conditioned_domains; next_condition]
# 				lik_new = initialize_density_evaluation(G, new_conditions, density, domain_map, map_collector)
# 				pi_c = () -> (typeof(next_condition.value_) == Unknown ? next_condition.Z.Pi[k] : 1)
# 				sum_list[i_k] = () -> (pi_c() * lik_new())
# 				lv_set(next_condition, 0)
# 			end
# 			ff = function()
# 				pp::BigFloat = 0.0
# 				for t in sum_list
# 					pp += t()
# 				end
# 				return pp
# 			end
# 			push!(mult_list, ff)
# 		else
# 			for g in G
# 				ma = rawCallDomain(g.T)
# 				mm = index_to_parameters(ma, density.parameters)
# 				push!(mult_list, () -> BigFloat(density.evaluate(g,  mm)))
# 			end

# 		end

# 	end
# 	FF = function()
# 		pp2::BigFloat = 1.0
# 		for t in mult_list
# 			pp2 *= t()
# 		end
# 		return pp2
# 	end
# 	return FF
# end

function prime_X(X::Observation)
    domains = GetDependentVariable(X)
    for LV in domains
        push!(LV.dependent_X, X)
    end
end

function call_collection(C::OrderedDict)
	for (ke,va) in C
		va.val = va.f()
	end
end

function LMEM(X::Set{Observation}, base::Parsa_Base;
	eps = 10^-6,
	init_eps = 10^-4,
	n_init = 1,
	verbose = true,
    max_steps=1000,
    catch_init_error = false,
    allow_desc_likelihood = false)
	##########
	X = collect(X)
	prime_X.(X)
	base.eval_catch = Dict()
	global pbar = ProgressBar(total = length(X) + 1)
	set_description(pbar, "Compiling")
	# domain_map = Dict([x => [LV for LV in unique(flattenConditionalDomain(x.T.domain)) if !lv_isKnown(LV)] for x in X])
	domain_map = Dict([x => unique([LV for LV in GetDependentVariable(x)]) for x in X])
	map_collector = OrderedDict()
	independent_map = Dict()
	# map_collector_lik = OrderedDict()
	(tau_chain, parameter_map, pi_parameters_used, tau_init) = E_step_initalize(X, base, domain_map, map_collector, independent_map, verbose)
	# return (0, 0, true)
	likelihood_ = initialize_density_evaluation(X, Vector{}(), base, domain_map, map_collector, independent_map)
	likelihood = () -> Float64(log(likelihood_()))
	# likelihood = () -> 1
	# if verbose
	# 	likelihood_ = initialize_density_evaluation(X, Vector{}(), base, Dict(), map_collector)
	# 	likelihood = () -> log(likelihood_())
	# end
	verbose ? update(pbar) : nothing
	tau_wild = [wild_tau(ta()) for ta in tau_init]
	M = M_step_init(X, tau_wild, parameter_map, base)
	param_reset = set_original_parameters(base)
	pi_reset = reset_Pi(X)
	Pi = Pi_init(X, tau_wild, pi_parameters_used)
	#init_likelihoods = zeros(n_init, n_wild)
	init_likelihoods = [[] for _ in 1:n_init]
	best_likelihood = -Inf
	best_tau = nothing
	best_prams = nothing
	best_Pi = nothing
	domain_post_catch = Vector{Vector}(undef, n_init)
	tau_wild = [wild_tau(ta()) for ta in tau_init]
	Pi(tau_wild)
	M(X, tau_wild, parameter_map, base)
	Pi(tau_wild)
	M(X, tau_wild, parameter_map, base)
	for i_init in 1:n_init
		param_reset()
		pi_reset()
		call_collection(map_collector)
		tau_wild::Vector{Vector{Real}} = [wild_tau(ta()) for ta in tau_init]
		Pi(tau_wild)
		M(X, tau_wild, parameter_map, base)
		lik_old = -Inf
		lik_new = likelihood()
		try
			for i_wild in 1:max_steps
				call_collection(map_collector)
				tau_wild = [(ta()) for ta in tau_chain]
				Pi(tau_wild)
				M(X, tau_wild, parameter_map, base)
				# println(lik_new)
				lik_old = lik_new
				lik_new = likelihood()
				if lik_new < lik_old && i_wild != 1 && !allow_desc_likelihood
					# init_likelihoods[i_init, i_wild:end] .= minimum(init_likelihoods[i_init, 1:(i_wild-1)])
					error("init error... decreasing likelihood")
				end
				# init_likelihoods[i_init, i_wild] = likelihood()
				if i_wild > 2
					push!(init_likelihoods[i_init], Float64(lik_new))
					verbose ? plotit(init_likelihoods, Vector{}()) : nothing
				end
				if abs(lik_new - lik_old) / abs(lik_old) < init_eps
					break
				end
			end
		catch e
			if !catch_init_error
				rethrow(e)
			else
				if !verbose @warn e end
				continue
			end
		end
		# lik_new = likelihood()
		if lik_new > best_likelihood
			best_likelihood = lik_new
			best_tau = tau_wild
			best_prams = save_parameters(base)
			best_Pi = save_Pi(X)
		end
	end
	best_Pi()
	best_prams()
	##########

	lik_old = -Inf
	# call_collection(map_collector)
	lik_new = best_likelihood #likelihood()

	all_likelihoods::Vector{Real} = [Float64(lik_new)]
	all_steps::Vector{Real} = [1]
	i = 2
	neg_lik_flag = false
	while ((abs(lik_new - lik_old) / abs(lik_new)) > eps && max_steps > 0) #|| i <= 5
		if (lik_new < lik_old)
			# println("non-increasing likelihood... stopping here")
			# neg_lik_flag = true
			# break
			error("non-increasing likelihood...")
		end
		call_collection(map_collector)
		tau::Vector{Vector{Real}} = [(ta()) for ta in tau_chain]
		Pi(tau)
		M(X, tau, parameter_map, base)
		lik_old = lik_new
        lik_new = ((likelihood()))
		all_likelihoods = [all_likelihoods; Float64(lik_new)]
		all_steps = [all_steps; i]
		verbose ? plotit(init_likelihoods, all_likelihoods) : nothing
		i = i + 1
	end
    post_process_params(base)
	return (log_likelihood = likelihood, n = length(X), neg_lik_flag = neg_lik_flag)
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
	plt = UnicodePlots.lineplot([0], [0]; ylim = (ymin, ymax), xlim = (xmin, xmax), xlabel = "steps", ylabel = "log-likelihood", height = Int(round(terminal[1] / 2)), width = Int(round(terminal[2] / 2)), color = :red)
	for i in 1:size(lines)[1]
		n_l = lines[i] #[l for l in lines[i, :] if l != 0]
		UnicodePlots.lineplot!(plt, 1:length(n_l), n_l, color = :blue)
	end
	if length(final_lines) > 0
		n_l = final_lines
		UnicodePlots.lineplot!(plt, xmax_init:(length(n_l)+xmax_init-1), n_l)
	end
	display(plt)
	sleep(0.001)
end

function get_parameter_package(params::Parameter, X::Vector{Observation}, tau::Vector{Vector{Real}}, parameter_map::Vector{Vector{Any}})
	relavent = getRelaventTaus(params, X, tau, parameter_map)
	x = [y[1] for y in relavent]
	pr::Vector{Real} = [y[2] for y in relavent]
	map = [y[3] for y in relavent]
	return (x, pr, map)
end

function zip_package(mapping, X::Vector{Observation}, tau::Vector{Vector{Real}}, parameter_map::Vector{Vector{Any}})
	x = [X[x_i] for (x_i, _) in mapping]
	pr::Vector{Real} = [tau[x_i][p_i] for (x_i, p_i) in mapping]
	map = [parameter_map[x_i][p_i] for (x_i, p_i) in mapping]
	return (x, pr, map)
end

function set_original_parameters(base::Parsa_Base)
	for ke in base.parameter_order
		for (ke2, param) in base.parameters[ke].parameter_map
			param.value_original = deepcopy(param.value.value)
		end
	end

	return function()
		for ke in base.parameter_order
			for (ke2, param) in base.parameters[ke].parameter_map
				param.value.value = param.value_original
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

function M_step_init(X::Vector{Observation}, tau::Vector{Vector{Real}}, parameter_map::Vector{Vector{Any}}, base::Parsa_Base)
	mappings = Vector{}()
	for ke in base.parameter_order
		for (ke2, params) in base.parameters[ke].parameter_map
			if !params.is_const
				mappings = [mappings; (getRelaventTausIndex(params, X, tau, parameter_map), params)]
			end
		end
	end
	return function (X, tau, parameter_map, base)
		for (index_map, param) in mappings
			(x, pr, map) = zip_package(index_map, X, tau, parameter_map)
			param.run_update(x, pr, map, base.log_pdf)
		end
	end
end

function post_process_params(base)
    for ke in base.parameter_order
        for (_, param) in base.parameters[ke].parameter_map
            if !isnothing(param.post_processing)
               param.value.value = param.post_processing(param.value.value)
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

function E_step_initalize(X::Vector{Observation}, density::Parsa_Base, all_domains, map_collector, independent_map, verbose)
	n = length(X)
	tau = Vector{Function}(undef, n)
	Q = Vector{Function}(undef, n)
	tau_pre_set = Vector{Function}(undef, n)
	tau_init = Vector{Function}(undef, n)
	parameters_used = Vector{Vector{Any}}(undef, n)
	pi_parameters_used = Vector{Vector{Any}}(undef, n)
    all_dependent_observations = getDependentOnX(X)
   for i in 1:n
		dependent_observations = unique(collect(all_dependent_observations[X[i]]))
		tau_init_i = E_step_i_initalize_initzial_values(X[i], dependent_observations, density, Vector{}(), Vector{}())
		# tau_init_i = E_step_i_initalize_initzial_values(X[i], dependent_observations, density, Vector{}(), (all_domains[X[i]]))
        # (tau_i, parameters_used_i, pi_parameters_used_i) = E_step_i_initalize(X[i], dependent_observations, density, Vector{}(), (all_domains[X[i]]), all_domains, map_collector)
		(tau_i, parameters_used_i, pi_parameters_used_i) = E_step_i_initalize(X[i], dependent_observations, density, Vector{}(), Vector{}(), all_domains, map_collector, independent_map)
        tau_i_func = () -> (tau_eval = tau_i(); Float64.(tau_eval / sum(tau_eval)))
		tau_i_init_func = () -> (L = tau_init_i; length(L) != 0 ? L ./ sum(L) : nothing)

		tau[i] = tau_i_func
		tau_init[i] = tau_i_init_func
		parameters_used[i] = parameters_used_i
		pi_parameters_used[i] = pi_parameters_used_i
		if verbose
			update(pbar)
		end
	end
	return (tau, parameters_used, pi_parameters_used, tau_init, tau_pre_set, Q)
end

function get_pre_set_factor(condition::Vector{LatentVaraible}, k_list)
	if sum([d.v() for d in condition] .== k_list) == 0
		return false
	else
		return true
	end
end

function E_step_i_initalize(X_i::Observation, X::Vector{Observation}, density::Parsa_Base, used_conditions::Vector, domains::Vector, domain_map::Dict, map_collector::OrderedDict, independent_map::Dict)
	tau_chain = Vector{Function}()
	params_used = Vector{Any}()
	Pi_used = Vector{Any}()
	E_step_i_initalize(X_i, X, density, used_conditions, domain_map, map_collector, independent_map, tau_chain, params_used, Pi_used)
	es = Vector{BigFloat}(undef, length(tau_chain))
	F = function()
		for (i,t) in enumerate(tau_chain)
			es[i] = t()
		end
		return es
	end
	return (F, params_used, Pi_used)
end

function E_step_i_initalize(X_i::Observation, X::Vector{Observation}, density::Parsa_Base, used_conditions::Vector, domain_map::Dict, map_collector::OrderedDict, independent_map::Dict, tau_chain::Vector{Function}, params_used::Vector{Any}, Pi_used::Vector{Any})
	condition = callDomain(X_i) #domains_left[1]
	if isnothing(condition)
		return nothing
	end
	K = typeof(condition.value_) == Unknown ? (1:condition.Z.K) : lv_v(condition)
	used_conditions = [used_conditions; condition]
	for k in K
		lv_set(condition, k)
		if !isnothing(callDomain(X_i))
			E_step_i_initalize(X_i, X, density, used_conditions, domain_map, map_collector, independent_map, tau_chain, params_used, Pi_used)
		else
			push!(Pi_used, Tuple([(d.Z, lv_v(d)) for d in used_conditions]))
			params = index_to_parameters(rawCallDomain(X_i.T), density.parameters)
			tau = initialize_density_evaluation(X, used_conditions, density, domain_map, map_collector, independent_map)
			current_ks = [(lv_v(d), typeof(condition.value_) == Unknown ) for d in used_conditions]
			Pi_val = () -> prod([current_ks[i][2] ? d.Z.Pi[current_ks[i][1]] : d.Z.Pi[current_ks[i][1]] for (i, d) in enumerate(used_conditions)])
			push!(tau_chain, () -> Pi_val() * tau())
			push!(params_used, params)
		end
		lv_set(condition, 0)
	end
end

function E_step_i_initalize_initzial_values(X_i::Observation, X::Vector{Observation}, density::Parsa_Base, used_conditions::Vector, domains::Vector)
	tau_chain_init = Vector{Int}()
	E_step_i_initalize_initzial_values!(X_i, X, density, used_conditions, domains, tau_chain_init)
	return tau_chain_init
end
function E_step_i_initalize_initzial_values!(X_i::Observation, X::Vector{Observation}, density::Parsa_Base, used_conditions::Vector, domains::Vector, tau_chain_init::Vector)
	condition = callDomain(X_i) #domains_left[1]
	if isnothing(condition)
		return tau_chain_init
	end
	K = typeof(condition.value_) == Unknown ? (1:condition.Z.K) : lv_v(condition)
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
end

function getRelaventTaus(parameter::Tuple{CategoricalZ, Int}, X::Vector{Observation}, tau::Vector{Vector{Real}}, parameter_map::Vector{Vector{Any}})
	reduce(vcat, [[pr for (pr, param) in zip(tau_x, params_x) if parameter in (param)] for (x, tau_x, params_x) in zip(X, tau, parameter_map)])
end

function getRelaventTausIndex(parameter::Tuple{CategoricalZ, Int}, X::Vector{Observation}, tau::Vector{Vector{Real}}, parameter_map::Vector{Vector{Any}})
	indx = Vector{Tuple}()
	for (i_1, params_x) in enumerate(parameter_map)
		for (i_2, params) in enumerate(params_x)
			if parameter in params
				push!(indx, (i_1, i_2))
			end
		end
	end
	return indx
end

function save_Pi(X::Vector{Observation})
	domains = unique(reduce(vcat, [GetDependentVariable(x) for x in X]))
	all_Z = unique([LV.Z for LV in domains])
	vals = Vector{}()
	for Z in all_Z
		push!(vals, deepcopy(Z.Pi))
	end
	return function()
		for (Z, v) in zip(all_Z, vals)
			Z.Pi = v
		end
	end
end

function reset_Pi(X::Vector{Observation})
	domains = unique(reduce(vcat, [GetDependentVariable(x) for x in X]))
	all_Z = unique([LV.Z for LV in domains])
	return function()
		for Z in all_Z
			k = length(Z.Pi)
			if !Z.constant
				Z.Pi = zeros(k) .+ 1/k
			end
		end
	end
end

function Pi_init(X::Vector{Observation}, tau::Vector{Vector{Real}}, pi_parameters_used::Vector{Vector{Any}})
	# domains = unique(reduce(vcat, [flattenConditionalDomain(x.T.domain) for x in X]))
	domains = unique(reduce(vcat, [GetDependentVariable(x) for x in X]))
	all_Z = unique([LV.Z for LV in domains])
	mappings = [[[] for _ in 1:Z_cat.K] for Z_cat in all_Z]
	for (i, Z_cat) in enumerate(all_Z)
		for k in 1:Z_cat.K
			mappings[i][k] = getRelaventTausIndex((Z_cat, k), X, tau, pi_parameters_used)
		end
	end
	return function (tau)
		for (i, Z_cat) in enumerate(all_Z)
			Pi_new = zeros(Z_cat.K)
			for k in 1:Z_cat.K
				relavent = [tau[i_1][i_2] for (i_1, i_2) in mappings[i][k]]
				Pi_new[k] = sum(relavent; init = 0)
			end
			Pi_new = Pi_new ./ sum(Pi_new)
			if !Z_cat.constant
				Z_cat.Pi = Pi_new
			end
		end
	end
end

function posterior_initalize(conditions, X::Vector{Observation}, density::Parsa_Base)
	prime_X.(X)
	domains = conditions
    all_domains = [GetDependentVariable(x) for x in X]
	X_sub = [(x, xf) for (x,xf) in zip(X, all_domains) if !isdisjoint(domains, xf)]
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
		X_sub=[X[1]]
	end
	domain_map = Dict([x => unique(GetDependentVariable(x)) for x in X_sub])
	map_collector = OrderedDict()
	independent_map = Dict()
	tau = Vector{}()
	Pi = Vector{}()
	posterior_initalize!(domains, X_sub, density, Vector{}(), domain_map, tau, Pi, map_collector, independent_map)
	tau_l = length(tau)
	vv = Vector{BigFloat}(undef, tau_l)
	function ()
		call_collection(map_collector)
		for i in 1:tau_l
			vv[i] = tau[i]()
		end
		tau_vec = (vv ./ sum(vv))
		Pi_val = [[y for (_, y) in pi_i] for pi_i in Pi]
		PV = Dict([pp => vv for (pp, vv) in zip(Pi_val, tau_vec)])
		return (max = max_posterior(PV), probability = PV)
	end
end

function posterior_initalize!(domains, X::Vector{Observation}, density::Parsa_Base, used_conditions::Vector, domain_map::Dict, tau_chain::Vector, Pi_used::Vector, map_collector::OrderedDict, independent_map::Dict)
	domains_left = setdiff(domains, used_conditions)
	condition = domains_left[1]
	# display(condition)
	K = typeof(condition.value_) == Unknown ? (1:condition.Z.K) : lv_v(condition)
	used_conditions = [used_conditions; condition]
	for k in K
		lv_set(condition, k)
		if length(domains_left) > 1
			posterior_initalize!(domains, X, density, used_conditions, domain_map, tau_chain::Vector, Pi_used::Vector, map_collector, independent_map)
		else
			tau = initialize_density_evaluation(X, used_conditions, density, domain_map, map_collector, independent_map; should_eval=true)
			push!(Pi_used, Tuple([(d.Z, lv_v(d)) for d in domains]))
			current_ks = [lv_v(d) for d in used_conditions]
			Pi_val = () -> prod([d.Z.Pi[current_ks[i]] for (i, d) in enumerate(used_conditions)])
			push!(tau_chain, () -> Pi_val() * tau())
		end
		lv_set(condition, 0)

	end
	return (tau_chain, Pi_used)
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
	return (cond)
end

function max_posterior_val(post)
	top_prob = 0
	cond = Nothing
	for p in post
		if p[2] > top_prob
			top_prob = p[2]
			cond = p[1]
		end
	end
	return (top_prob)
end

