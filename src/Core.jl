using UnicodePlots
using Distributions

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
	return unique(LV)
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

function getIndependentSets(X::Vector{Observation}, Z, domain_map)
	domain_map = Dict()
	for x in X
			domain_map[x] = GetDependentVariable(x)
	end
	n = length(X)
	# if length(domain_map) == 0
	# 	for x in X
	# 		domain_map[x] = GetDependentVariable(x)
	# 	end
	# end
	indo_sets = Vector{Any}(undef, n)
	for i in 1:n
		sets = unique([X[i]; conditional_dependent_search(domain_map[X[i]], domain_map, Z, X)])
		indo_sets[i] = sets[sortperm([objectid(x) for x in sets])]
	end
	return unique(indo_sets)
end

function conditional_dependent_search(D, domain_map, Z, X)
	x_depo = Vector{Observation}()
	new_d = setdiff(D, Z)
	for d in new_d
		if !lv_isKnown(d)
			d_depo = collect(values(d.dependent_X))
			new_Z = setdiff(reduce(vcat, [domain_map[x] for x in d_depo if x in X]), [D; Z])
			if length(new_Z) > 0
				d_depo = [d_depo; conditional_dependent_search(new_Z, domain_map, [Z; D], X)]
			end
			for x in d_depo
				if x in X
					x_depo = [x_depo; x]
				end
			end
		end
	end
	return x_depo
end

function getIndependentSets(X::Vector{Observation})
	domains = Dict([x => [lv for lv in GetDependentVariable(x) if !lv_isKnown(lv)] for x in X])
	indo_sets = Dict([x => Set{Observation}() for x in X])
	indo_sets_obs = Vector{Vector{Observation}}(undef, length(X))
	for x in X
		push!(indo_sets[x], x)
		for condition in domains[x]
				for va in condition.dependent_X
					if !(va in indo_sets[x]) && va in X && condition in domains[va]
						push!(indo_sets[x], va)
					end
				end
		end
	end

	for (i, x) in enumerate(X)
		for y in indo_sets[x]
			conditional_dependent_search!(y, indo_sets, indo_sets[x], X)
		end
		collected_x = collect(indo_sets[x])
		indo_sets_obs[i] = collected_x[sortperm([objectid(x) for x in collected_x])]
	end
	return unique(indo_sets_obs)
end

function conditional_dependent_search!(x, indo_sets, set, X)
	for y in indo_sets[x]
		if !(y in set) && y in X
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

function initialize_density_evaluation(X::Vector{Observation}, density::Parsa_Base, independent_by)
	if length(independent_by) == 0
		D = Vector{Any}()
		return (initialize_density_evaluation(X, D, density))
	else
		D = Vector{Any}()
		return (initialize_density_evaluation_ind(X, D, density, independent_by))
	end
end

function initialize_density_evaluation(X::Vector{Observation}, conditioned_domains::Vector, density::Parsa_Base, independent_by)
	if length(independent_by) == 0
		D = Vector{Any}()
		return (initialize_density_evaluation(X, conditioned_domains, density))
	else
		D = Vector{Any}()
		return (initialize_density_evaluation_ind(X, conditioned_domains, density, independent_by))
	end
end

function initialize_density_evaluation(X::Vector{Observation}, conditioned_domains::Vector, density::Parsa_Base, domain_map::Dict, map_collector::Dict)
	independent_sets = getIndependentSets(X)
	# println(length(independent_sets))
	# println("---")
	# sleep(0.1)
	# println(length(independent_sets))
	# println("--")
	# sleep(0.1)
	# println(length(independent_sets))
	# println(length(independent_sets))
	mult_list = Vector{Function}()
	for G in independent_sets

		all_domains = [unique([lv for lv in GetDependentVariable(x)]) for x in G]
		domain_lengths = [length(d) for d in all_domains]
		# println(countmap(domain_lengths))
		# if (sum(argmax(domain_lengths) .== domain_lengths) == length(domain_lengths))
		domains = [LV for LV in (reduce(vcat, all_domains))]
		lv_freq_map = countmap(domains)
		# println(length(domain_lengths))
		# println((values(lv_freq_map)))
		# if maximum(domain_lengths; init=0) <= maximum(values(lv_freq_map); init=0)
			# println(values(lv_freq_map))
			next_conditions = domains #setdiff(domains, conditioned_domains)
			lv_freq_map = filter(x -> x[1] in next_conditions, lv_freq_map)
			top_order = sortperm(collect(values(lv_freq_map)); rev=true)
			next_conditions = collect(keys(lv_freq_map))[top_order]
		# else
		# 	next_conditions = all_domains[argmax(domain_lengths)]
		# end
		# domains = unique([LV for LV in (reduce(vcat, all_domains))])
		# best = [(lv_set(lv, 1); L = length(getIndependentSets(G)); lv_set(lv, 0); L) for lv in domains]
		# next_conditions = length(best) == 0 ? [] : [domains[argmax(best)]]
		if length(next_conditions) != 0
			next_condition = next_conditions[1]
			# display(next_condition)
			# sleep(0.1)
			K = typeof(next_condition.value_) == Unknown ? (1:next_condition.Z.K) : lv_v(next_condition)
			sum_list = Vector{Function}(undef, length(K))
			for (i_k, k) in enumerate(K)
				lv_set(next_condition, k)
				new_conditions = [conditioned_domains; next_condition]
				lik_new = initialize_density_evaluation(G, new_conditions, density, domain_map, map_collector)
				sum_list[i_k] = () -> (next_condition.Z.Pi[k] * lik_new())
				lv_set(next_condition, 0)
			end
			ff = function()
				pp::BigFloat = 0.0
				for t in sum_list
					pp += t()
				end
				return pp
			end
			push!(mult_list, ff)
		else
			for g in G
				ma = rawCallDomain(g.T)
				mm = index_to_parameters(ma, density.parameters)
				push!(mult_list, () -> BigFloat(density.evaluate(g,  mm)[1]))
			end

		end

	end
	FF =  function()
		pp2::BigFloat = 1.0
		for t in mult_list
			pp2 *= t()
		end
		return pp2
	end
	return FF
end

function prime_X(X::Observation)
    domains = GetDependentVariable(X)
    for LV in domains
        push!(LV.dependent_X, X)
    end
end

function LMEM(X::Set{Observation}, base::Parsa_Base;
	eps = 10^-6,
	n_init = 1,
	n_wild = 5,
	approx = false,
	verbose = true,
	should_initialize = true,
	Q_func = false,
    max_steps=1000,
    independent_by = Vector{CategoricalZ}())
	##########
	X = collect(X)
	prime_X.(X)
	base.eval_catch = Dict()
	global pbar = ProgressBar(total = length(X) + 1)
	set_description(pbar, "Compiling")
	# domain_map = Dict([x => [LV for LV in unique(flattenConditionalDomain(x.T.domain)) if !lv_isKnown(LV)] for x in X])
	# domain_map = Dict([x => [LV for LV in GetDependentVariable(x) if !lv_isKnown(LV)] for x in X])
	map_collector = Dict()
	(tau_chain, parameter_map, pi_parameters_used, tau_init) = E_step_initalize(X, base, Dict(), map_collector, verbose)
	likelihood = () -> 1
	if verbose
		likelihood_ = initialize_density_evaluation(X, Vector{}(), base, Dict(), map_collector)
		likelihood = () -> log(likelihood_())
	end
	verbose ? update(pbar) : nothing
	tau_wild = [wild_tau(ta()) for ta in tau_init]
	M = M_step_init(X, tau_wild, parameter_map, base)
	Pi = Pi_init(X, tau_wild, pi_parameters_used)
	init_likelihoods = zeros(n_init, n_wild)
	if should_initialize
		best_likelihood = -Inf
		best_tau = nothing
		domain_post_catch = Vector{Vector}(undef, n_init)
		tau_wild = [wild_tau(ta()) for ta in tau_init]
		Pi(tau_wild)
		M(X, tau_wild, parameter_map, base)
		Pi(tau_wild)
		M(X, tau_wild, parameter_map, base)
		for i_init in 1:n_init
			tau_wild::Vector{Vector{Real}} = [wild_tau(ta()) for ta in tau_init]
			Pi(tau_wild)
			M(X, tau_wild, parameter_map, base)
			for i_wild in 1:n_wild
				tau_wild = [(ta()) for ta in tau_chain]
				Pi(tau_wild)
				M(X, tau_wild, parameter_map, base)
				init_likelihoods[i_init, i_wild] = likelihood()
				verbose ? plotit(init_likelihoods, Vector{}()) : nothing
			end
			lik_new = likelihood()
			if lik_new > best_likelihood
				best_likelihood = lik_new
				best_tau = tau_wild
			end
		end
		tau_start::Vector{Vector{Real}} = best_tau
		Pi(tau_start)
		M(X, tau_start, parameter_map, base)
		Pi(tau_start)
		M(X, tau_start, parameter_map, base)
	else
		tau_start = [(ta()) for ta in tau_init]
		Pi(tau_start)
		M(X, tau_start, parameter_map, base)
		Pi(tau_start)
		M(X, tau_start, parameter_map, base)
	end

	##########

	lik_old = -Inf
	lik_new = ((likelihood()))
	all_likelihoods::Vector{Real} = [Float64(lik_new)]
	all_steps::Vector{Real} = [1]
	i = 2
	while ((abs(lik_new - lik_old) / abs(lik_new)) > eps && max_steps > 0) || i <= 5
		if (lik_new < lik_old)
			println("error")
		end
        lik_old = lik_new
        lik_new = ((likelihood()))
		tau::Vector{Vector{Real}} = [(ta()) for ta in tau_chain]
		Pi(tau)
		M(X, tau, parameter_map, base)
		all_likelihoods = [all_likelihoods; Float64(lik_new)]
		all_steps = [all_steps; i]
		verbose ? plotit(init_likelihoods, all_likelihoods) : nothing
		i = i + 1
	end
    post_process_params(base)
	return (log_likelihood = likelihood, n = length(X))
end

function plotit(lines, final_lines)
	println("\33[H")
	print("\33c\e[3J")
	terminal = displaysize(stdout)
	lll = [l for l in [vec(lines); final_lines] if l != 0]
	ymax = maximum([l for l in lll])
	ymin = minimum([l for l in lll])
	xmin = 0
	xmax = length(findall(lines[1, :] .!= 0)) + length(final_lines)
	plt = UnicodePlots.lineplot([0], [0]; ylim = (ymin, ymax), xlim = (xmin, xmax), xlabel = "steps", ylabel = "log-likelihood", height = Int(round(terminal[1] / 2)), width = Int(round(terminal[2] / 2)), color = :red)
	for i in 1:size(lines)[1]
		n_l = [l for l in lines[i, :] if l != 0]
		UnicodePlots.lineplot!(plt, 1:length(n_l), n_l, color = :blue)
	end
	if length(final_lines) > 0
		n_l = final_lines
		UnicodePlots.lineplot!(plt, size(lines)[2]:(length(n_l)+size(lines)[2]-1), n_l)
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
	tau_new = zeros(length(tau))
	tau_new[g] = 1
	tau_new /= sum(tau_new)
	return tau_new
end

function E_step_initalize(X::Vector{Observation}, density::Parsa_Base, all_domains, map_collector, verbose)
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
		(tau_i, parameters_used_i, pi_parameters_used_i) = E_step_i_initalize(X[i], dependent_observations, density, Vector{}(), Vector{}(), all_domains, map_collector)
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

function E_step_i_initalize(X_i::Observation, X::Vector{Observation}, density::Parsa_Base, used_conditions::Vector, domains::Vector, domain_map::Dict, map_collector::Dict)
	tau_chain = Vector{Function}()
	params_used = Vector{Any}()
	Pi_used = Vector{Any}()
	E_step_i_initalize(X_i, X, density, used_conditions, domain_map, map_collector, tau_chain, params_used, Pi_used)
	es = Vector{BigFloat}(undef, length(tau_chain))
	F = function()
		for (i,t) in enumerate(tau_chain)
			es[i] = t()
		end
		return es
	end
	return (F, params_used, Pi_used)
end

function E_step_i_initalize(X_i::Observation, X::Vector{Observation}, density::Parsa_Base, used_conditions::Vector, domain_map::Dict, map_collector::Dict, tau_chain::Vector{Function}, params_used::Vector{Any}, Pi_used::Vector{Any})
	condition = callDomain(X_i) #domains_left[1]
	if isnothing(condition)
		return nothing
	end
	K = typeof(condition.value_) == Unknown ? (1:condition.Z.K) : lv_v(condition)
	used_conditions = [used_conditions; condition]
	for k in K
		lv_set(condition, k)
		if !isnothing(callDomain(X_i))
			E_step_i_initalize(X_i, X, density, used_conditions, domain_map, map_collector, tau_chain, params_used, Pi_used)
		else
			push!(Pi_used, Tuple([(d.Z, lv_v(d)) for d in used_conditions]))
			params = index_to_parameters(rawCallDomain(X_i.T), density.parameters)
			tau = initialize_density_evaluation(X, used_conditions, density, domain_map, map_collector)
			current_ks = [lv_v(d) for d in used_conditions]
			Pi_val = () -> prod([d.Z.Pi[current_ks[i]] for (i, d) in enumerate(used_conditions)])
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
	domain_map = Dict([x => GetDependentVariable(x) for x in X_sub])
	tau = Vector{}()
	Pi = Vector{}()
	# println(length(X_sub))
	# println("HERE")
	posterior_initalize!(domains, X_sub, density, Vector{}(), domain_map, tau, Pi)
	tau_l = length(tau)
	vv = Vector{BigFloat}(undef, tau_l)
	function ()
		for i in 1:tau_l
			vv[i] = tau[i]()
		end
		tau_vec = (vv ./ sum(vv))
		Pi_val = [[y for (_, y) in pi_i] for pi_i in Pi]
		PV = Dict([pp => vv for (pp, vv) in zip(Pi_val, tau_vec)])
		return (max = max_posterior(PV), probability = PV)
	end
end

function posterior_initalize!(domains, X::Vector{Observation}, density::Parsa_Base, used_conditions::Vector, domain_map::Dict, tau_chain::Vector, Pi_used::Vector)
	domains_left = setdiff(domains, used_conditions)
	condition = domains_left[1]
	# display(condition)
	K = typeof(condition.value_) == Unknown ? (1:condition.Z.K) : lv_v(condition)
	used_conditions = [used_conditions; condition]
	for k in K
		lv_set(condition, k)
		if length(domains_left) > 1
			posterior_initalize!(domains, X, density, used_conditions, domain_map, tau_chain::Vector, Pi_used::Vector)
		else
			tau = initialize_density_evaluation(X, domains, density, domain_map, Dict())
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

