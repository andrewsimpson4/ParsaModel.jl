
using UnicodePlots


function getRelaventTaus(parameter::Parameter, X::Vector{Observation}, tau::Vector{Vector{Real}}, parameter_map::Vector{Vector{Any}})
	reduce(vcat, [[(x, pr, param) for (pr, param) in zip(tau_x, params_x) if parameter in values(param)] for (x, tau_x, params_x) in zip(X, tau, parameter_map)])
end

function getRelaventTausIndex(parameter::Parameter, X::Vector{Observation}, tau::Vector{Vector{Real}}, parameter_map::Vector{Vector{Any}})
	reduce(vcat, [[(x_i, p_i) for (p_i, (pr, param)) in enumerate(zip(tau_x, params_x)) if parameter in values(param)] for (x_i, (x, tau_x, params_x)) in enumerate(zip(X, tau, parameter_map))])
end

joint_CatSet(D) = typeof(D) == CategoricalZVec ? [D.inside; D.outside] : D
function flattenConditionalDomain(Domain)
	dd = [unique(reduce(vcat, ([joint_CatSet(v) for v in reduce(vcat, values(d()))]))) for d in Domain]
	R = (reduce(vcat, dd))
	G = reduce(vcat, [r for r in R if typeof(r) != Int64])
	typeof(G) == LatentVaraible ? [G] : G
end

function flattenConditionalDomainSimple(Domain)
	dd = [d() for d in Domain]
	R = (reduce(vcat, dd))
	return R
end

function flattenConditionalDomainNested(Domain)
	dd = [unique((values(d()))) for d in Domain]
	R = (reduce(vcat, dd))
	[r for r in R if typeof(r) != Int64]
end

function getDependentObservations(X::Observation, Y::Vector{Observation})::Vector{Observation}
	f1 = flattenConditionalDomain(X.T.domain)
	overlap = [length(intersect(f1, flattenConditionalDomain(y.T.domain))) > 0 for y in Y]
	Y[overlap.==1]
end

function getDependentObservations(f1::Vector{Any}, f2::Vector{Vector{Any}}, Y::Vector{Observation})::Vector{Observation}
	overlap = [length(intersect(f1, f)) > 0 for f in f2]
	Y[overlap.==1]
end

function getDependentObservations(f1::Vector{LatentVaraible}, f2::Vector{Vector{LatentVaraible}}, Y::Vector{Observation})::Vector{Observation}
	overlap = [length(intersect(f1, f)) > 0 for f in f2]
	Y[overlap.==1]
end

function getIndependentSets(X::Vector{Observation}, Z)
	n = length(X)
	domain_map = Dict([x => flattenConditionalDomain(x.T.domain) for x in X])
	indo_sets = Vector{Any}(undef, n)
	for i in 1:n
		sets = unique([X[i]; conditional_dependent_search(domain_map[X[i]], domain_map, Z)])
		indo_sets[i] = sets[sortperm([objectid(x) for x in sets])]
	end
	return unique(indo_sets)
end

function conditional_dependent_search(D, domain_map, Z)
	x_depo::Vector{Observation} = []
	new_d = setdiff(D, Z)
	for d in new_d
		if !lv_isKnown(d)
			d_depo = collect(values(d.dependent_X))
			new_Z = setdiff(reduce(vcat, [domain_map[x] for x in d_depo]), [D; Z])
			if length(new_Z) > 0
				d_depo = [d_depo; conditional_dependent_search(new_Z, domain_map, [Z; D])]
			end
			x_depo = [x_depo; d_depo]
		end
	end
	return x_depo
end


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

function initialize_density_evaluation(X::Vector{Observation}, conditioned_domains::Vector, density::Parsa_Base)
	# current_condition = [(LV, lv_v(LV)) for LV in conditioned_domains if typeof(LV.value_) == Unknown]
	# if haskey(density.eval_catch, (X, current_condition))
	#     # println("found??")
	#     return () -> density.eval_catch[(X, current_condition)]
	# end
	independent_sets = getIndependentSets(X, conditioned_domains)
	mult_list = Vector{}()
	for G in independent_sets
		domains = (flattenConditionalDomain(reduce(vcat, [x.T.domain for x in G])))
		lv_freq_map = countmap(domains)
		next_conditions = setdiff(domains, conditioned_domains)
		lv_freq_map = filter(x -> x[1] in next_conditions, lv_freq_map)
		top_order = sortperm(collect(values(lv_freq_map)); rev=true)
		next_conditions = collect(keys(lv_freq_map))[top_order]
		if length(next_conditions) != 0
			next_condition = next_conditions[1]
			K = typeof(next_condition.value_) == Unknown ? (1:next_condition.Z.K) : lv_v(next_condition)
			sum_list = Vector{Function}(undef, length(K))
			for (i_k, k) in enumerate(K)
				lv_set(next_condition, k)
				new_conditions = [conditioned_domains; next_condition]
				lik_new = initialize_density_evaluation(G, new_conditions, density)
				sum_list[i_k] = () -> (lv_set(next_condition, k); eval = next_condition.Z.Pi[k] * lik_new(); lv_set(next_condition, 0); eval)
				lv_set(next_condition, 0)
			end
			mult_list = [mult_list; () -> sum([t() for t in sum_list])]
		else
			for g in G
				mm = g.T.map()
				mult_list = [mult_list; () -> BigFloat(density.evaluate(g,  mm)[1])]
			end

		end

	end
	# density.eval_catch[(X, current_condition)] = 1
	# return () -> (ll = prod([t() for t in mult_list]); density.eval_catch[(X, current_condition)] = ll; return ll)
	return () -> prod([t() for t in mult_list])
end


function initialize_density_evaluation_ind(X::Vector{Observation}, density::Parsa_Base, independent_by)
	D = Vector{Any}()
	return (initialize_density_evaluation_ind(X, D, density, independent_by))
end


function initialize_density_evaluation_ind(X::Vector{Observation}, conditioned_domains::Vector, density::Parsa_Base, independent_by)
	current_condition = [(LV, LV.v()) for LV in conditioned_domains if typeof(LV.value_) == Unknown]
	if haskey(density.eval_catch, (X, current_condition))
	    return () -> density.eval_catch[(X, current_condition)]
	end
	# independent_sets, _ = independent_sets_from_independent_by(X, independent_by)
    independent_sets = [collect(values(LV.dependent_X)) for (_, LV) in independent_by[1].LV]
	mult_list = Vector{}()
	for G in independent_sets
		domains = (flattenConditionalDomain(reduce(vcat, [x.T.domain for x in G])))
		next_conditions = setdiff(domains, conditioned_domains)
		if length(next_conditions) != 0
			next_condition = next_conditions[1]
			K = typeof(next_condition.value_) == Unknown ? (1:next_condition.Z.K) : lv_v(next_condition)
			sum_list = Vector{}()
			for k in K
				lv_set(next_condition, k)
				new_conditions = [conditioned_domains; next_condition]
				lik_new = initialize_density_evaluation(G, new_conditions, density)
				sum_list = [sum_list; () -> (lv_set(next_condition, k); eval = next_condition.Z.Pi[k] * lik_new(); lv_set(next_condition, 0); eval)]
				lv_set(next_condition, 0)
			end
			mult_list = [mult_list; () -> sum([t() for t in sum_list])]
		else
			mm = G.T.map()
			mult_list = [mult_list; () -> BigFloat(density.evaluate(G, mm)[1])]
		end

	end
	density.eval_catch[(X, current_condition)] = 1
	return () -> (ll = prod([t() for t in mult_list]); density.eval_catch[(X, current_condition)] = ll; return ll)
	# return () -> prod([t() for t in mult_list])
end

function LMEM(X::Vector{Observation}, base::Parsa_Base;
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
	base.eval_catch = Dict()
	global pbar = ProgressBar(total = length(X) + 1)
	set_description(pbar, "Compiling")
	(tau_chain, parameter_map, pi_parameters_used, tau_init, tau_pre_set, Q) = E_step_initalize(X, base, independent_by)
	likelihood_ = initialize_density_evaluation(X, base, independent_by)
	likelihood = () -> log(likelihood_())
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
	println("---")
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
		for (_, params) in base.parameters[ke].parameter_map
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
	G = Distributions.Categorical(tau)
	g = rand(G)
	tau_new = zeros(length(tau))
	tau_new[g] = 1
	tau_new /= sum(tau_new)
	return tau_new
end

function E_step_initalize(X::Vector{Observation}, density::Parsa_Base, independent_by)
	n = length(X)
	tau = Vector{Function}(undef, n)
	Q = Vector{Function}(undef, n)
	tau_pre_set = Vector{Function}(undef, n)
	tau_init = Vector{Function}(undef, n)
	parameters_used = Vector{Vector{Any}}(undef, n)
	pi_parameters_used = Vector{Vector{Any}}(undef, n)
	all_domains = [flattenConditionalDomain(x.T.domain) for x in X]
   for i in 1:n
		dependent_observations = unique(reduce(vcat, [collect(values(LV.dependent_X)) for LV in all_domains[i]]))
		tau_init_i = E_step_i_initalize_initzial_values(X[i], dependent_observations, density, Vector{}())
        (tau_i, parameters_used_i, pi_parameters_used_i, tau_i_pre_set, pi_chain_i) = E_step_i_initalize(X[i], dependent_observations, density, Vector{}())
        tau_i_func = () -> (tau_eval = tau_i([]); Float64.(tau_eval / sum(tau_eval)))
		Q_i = () -> (tau_eval = tau_i([]); pp = pi_chain_i([]); probs = (tau_eval / sum(tau_eval)); sum(probs .* log.(tau_eval)))
		tau_i_pre_set_func = () -> (tau_eval = tau_i_pre_set([]); (tau_eval / sum(tau_eval)))
		tau_i_init_func = () -> (L = tau_init_i([]); L ./ sum(L))

		tau[i] = tau_i_func
		Q[i] = Q_i
		tau_init[i] = tau_i_init_func
		tau_pre_set[i] = tau_i_pre_set_func
		parameters_used[i] = parameters_used_i
		pi_parameters_used[i] = pi_parameters_used_i
		update(pbar)

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

function E_step_i_initalize(X_i::Observation, X::Vector{Observation}, density::Parsa_Base, used_conditions::Vector)
	domains = flattenConditionalDomain(X_i.T.domain)
	domains_left = setdiff(domains, used_conditions)
	condition = domains_left[1]

	tau_chain = (V) -> V
	tau_chain_pre_set = (V) -> V
	pi_chain = (V) -> V
	params_used = Vector{Any}()
	K = typeof(condition.value_) == Unknown ? (1:condition.Z.K) : lv_v(condition)
	Pi_used = Vector{Any}()
	for k in K
		known_correction = () -> typeof(condition.value_) != Unknown ? (lv_v(condition) == k ? 1 : 0) : 1
		lv_set(condition, k)
		if length(domains_left) > 1
			(tau, params, pi_params, tau_pre_set, pi_c) = E_step_i_initalize(X_i, X, density, [used_conditions; condition])
			tau_chain = tau_chain ∘ (V) -> (lv_set(condition, k); eval = [tau([]); V]; lv_set(condition, 0); eval)
			pi_chain = pi_chain ∘ (V) -> (lv_set(condition, k); eval = [pi_c([]); V]; lv_set(condition, 0); eval)
			tau_chain_pre_set = tau_chain_pre_set ∘ (V) -> ([tau_pre_set([]); V])
			params_used = [params_used; params]
			Pi_used = [Pi_used; pi_params]
		else
			domains = [used_conditions; condition]
			params = index_to_parameters(X_i.T.map(), density.parameters)
			tau = initialize_density_evaluation(X, domains, density)
			Pi_used = [Pi_used; Tuple([(d.Z, lv_v(d)) for d in domains])]
			Pi_val = () -> prod([typeof(d.value_) == Unknown ? d.Z.Pi[lv_v(d)] : 1 for d in domains])
			tau_chain = ((V) -> (lv_set(condition, k); eval = [V; Pi_val() * tau()]; lv_set(condition, 0); eval)) ∘ tau_chain
			pi_chain = ((V) -> (lv_set(condition, k); eval = [V; Pi_val()]; lv_set(condition, 0); eval)) ∘ pi_chain
			tau_chain_pre_set = ((V) -> ([V; lv_v(condition) == k ? 1 : 0])) ∘ tau_chain_pre_set
			params_used = [params_used; params]

		end
		lv_set(condition, 0)
	end
	return (tau_chain, params_used, Pi_used, tau_chain_pre_set, pi_chain)
end


function E_step_i_initalize_initzial_values(X_i::Observation, X::Vector{Observation}, density::Parsa_Base, used_conditions::Vector)
	domains = flattenConditionalDomain(X_i.T.domain)
	domains_left = setdiff(domains, used_conditions)
	condition = domains_left[1]
	tau_chain_init = (V) -> V
	K = typeof(condition.value_) == Unknown ? (1:condition.Z.K) : lv_v(condition)
	pre_set_val = lv_v(condition)
	for k in K
		init_correction = pre_set_val == 0 ? 1 : (pre_set_val == k ? 1 : 0)
		lv_set(condition, k)
		if length(domains_left) > 1
			tau_init = E_step_i_initalize_initzial_values(X_i, X, density, [used_conditions; condition])
			tau_chain_init = tau_chain_init ∘ (V) -> ([init_correction * tau_init([]); V])
		else
			tau_chain_init = ((V) -> [V; init_correction]) ∘ tau_chain_init

		end
		lv_set(condition, 0)
	end
	return tau_chain_init
end

function getRelaventTaus(parameter::Tuple{CategoricalZ, Int}, X::Vector{Observation}, tau::Vector{Vector{Real}}, parameter_map::Vector{Vector{Any}})
	reduce(vcat, [[pr for (pr, param) in zip(tau_x, params_x) if parameter in (param)] for (x, tau_x, params_x) in zip(X, tau, parameter_map)])
end

function getRelaventTausIndex(parameter::Tuple{CategoricalZ, Int}, X::Vector{Observation}, tau::Vector{Vector{Real}}, parameter_map::Vector{Vector{Any}})
	reduce(vcat, [[(i_1, i_2) for (i_2, (pr, param)) in enumerate(zip(tau_x, params_x)) if parameter in param] for (i_1, (x, tau_x, params_x)) in enumerate(zip(X, tau, parameter_map))])
end


function Pi_init(X::Vector{Observation}, tau::Vector{Vector{Real}}, pi_parameters_used::Vector{Vector{Any}})
	domains = reduce(vcat, flattenConditionalDomain(reduce(vcat, [x.T.domain for x in X])))
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
				Pi_new[k] = sum(relavent)
			end
			Pi_new = Pi_new ./ sum(Pi_new)
			Z_cat.Pi = Pi_new
		end
	end
end

function posterior_probability(conditions::LatentVaraible, X::Vector{Observation}, base::Parsa_Base)
	posterior_probability([conditions], X, base)
end

function posterior_probability(conditions::Function, X::Vector{Observation}, base::Parsa_Base)
	domains = flattenConditionalDomainSimple([conditions])
	X_sub = unique(reduce(vcat, [collect(values(LV.dependent_X)) for LV in domains]))
	(tau, Pi) = posterior_probability_sub(X_sub, conditions, base, 1)
	tau = Float64.(tau ./ sum(tau))
	Pi = [[y for (_, y) in pi_i] for pi_i in Pi]
	return collect(zip(Pi, tau))
end

function posterior_probability_sub(X::Vector{Observation}, domains::Function, density::Parsa_Base, j::Int)
	conditions = flattenConditionalDomainSimple([domains])
	condition = conditions[j]
	taus = Vector{Real}()
	Pi_used = Vector{Any}()

	K = typeof(condition.value_) == Unknown ? (1:condition.Z.K) : lv_v(condition)
	for k in K
		lv_set(condition, k)
		if j < length(conditions)
			(tau, pi_params) = posterior_probability_sub(X, domains, density, j + 1)
			taus = [taus; tau]
			Pi_used = [Pi_used; pi_params]
		else
			tau = prod([d.Z.Pi[lv_v(d)] for d in conditions]) * initialize_density_evaluation(X, conditions, density)()

			taus = [taus; tau]
			Pi_used = [Pi_used; Tuple([(d.Z, lv_v(d)) for d in conditions])]

		end
		lv_set(condition, 0)

	end
	return (taus, Pi_used)
end

function posterior_initalize(conditions, X::Vector{Observation}, density::Parsa_Base)
	domains = flattenConditionalDomainSimple([conditions])
    all_domains = [flattenConditionalDomain(x.T.domain) for x in X]
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
	(tau, Pi) = posterior_initalize(domains, X_sub, density, Vector{}())
	function ()
		tt = tau([])
		tau_vec = Float64.(tt ./ sum(tt))
		Pi_val = [[y for (_, y) in pi_i] for pi_i in Pi]
		return collect(zip(Pi_val, tau_vec))
	end
end

function posterior_initalize(domains, X::Vector{Observation}, density::Parsa_Base, used_conditions::Vector)
	domains_left = setdiff(domains, used_conditions)
	condition = domains_left[1]
	tau_chain = (V) -> V
	K = typeof(condition.value_) == Unknown ? (1:condition.Z.K) : lv_v(condition)
	Pi_used = Vector{Any}()
	for k in K
		lv_set(condition, k)
		if length(domains_left) > 1
			(tau, pi_params) = posterior_initalize(X_i, X, density, [used_conditions; condition])
			tau_chain = tau_chain ∘ (V) -> (lv_set(condition, k); eval = [tau([]); V]; lv_set(condition, 0); eval)
			Pi_used = [Pi_used; pi_params]
		else
			tau = initialize_density_evaluation(X, domains, density)
			Pi_used = [Pi_used; Tuple([(d.Z, lv_v(d)) for d in domains])]
			Pi_val = () -> prod([typeof(d.value_) == Unknown ? d.Z.Pi[lv_v(d)] : 1 for d in domains])
			tau_chain = ((V) -> (lv_set(condition, k); eval = [V; Pi_val() * tau()]; lv_set(condition, 0); eval)) ∘ tau_chain

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

