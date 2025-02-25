include("./Types.jl")
include("./Models.jl")
include("./Core.jl")


function ParsaModel(base_model)
    space = Module()
    Base.eval(space, quote
        base_model = $base_model
        fit_model = nothing
        X_val = Dict()
    end)
    return space
end

macro Categorical(model, name, K)
    mod = esc(model)
    K_val = esc(K)
    na = (name)
    quote
        $mod.$na = Main.CategoricalZ(K = $K_val)
    end
end

macro Categorical_Set(model, name, K, indx)
    mod = esc(model)
    na = QuoteNode(name)
    K_val = esc(K)
    ind = esc(indx)
    quote
        Base.eval($mod, quote
            $$na = Main.CategoricalZset()
            for (ind, k) in zip($$ind, $$K_val)
                $$na.set[[ind]] = Main.CategoricalZ(K = k)
            end
        end)
    end
end

macro Observation(model, main_obj, index_set)
    mod = esc(model)
    obs_name = string(main_obj.args[1].args[1])
    X_name = main_obj.args[2].args[1].args[1]
    X_loaded = esc(X_name)
    map = QuoteNode(main_obj.args[2].args[2])
    indx = QuoteNode(index_set.args[1])
    set = esc(index_set.args[2])

    quote
        Base.eval($mod, quote
            for j in $$set
                $$indx = j
                va = ($$X_loaded[j])
                local tt = Main.T([() -> Dict((() -> $$map)())], () -> Dict((() -> $$map)()))
                local ob = Main.Observation(va, tt)
                global X_val[($$obs_name, j)] = ob
            end
        end)
    end
end


macro Known(model, eq, index_set)
    mod = esc(model)
    name = eq.args[1]
    vals = eq.args[2]
    loaded_vals = esc(vals.args[1])
    vals_indx = QuoteNode(vals.args[2])
    indx = QuoteNode(index_set.args[1])
    indx_2 = QuoteNode(name.args[2])
    set = esc(index_set.args[2])
    Z = QuoteNode(name.args[1])
    quote
        Base.eval($mod, quote
            for j in $$set
                $$indx = j
                z = $$Z
                va = $$loaded_vals[$$vals_indx]
                z[$$indx_2] = Main.LatentVaraible(z, va)
            end
        end)
    end
end

macro Initialize(model, eq, index_set)
    mod = esc(model)
    name = eq.args[1]
    vals = eq.args[2]
    loaded_vals = esc(vals.args[1])
    indx = QuoteNode(index_set.args[1])
    indx_2 = QuoteNode(name.args[2])
    set = esc(index_set.args[2])
    Z = QuoteNode(name.args[1])
    quote
        Base.eval($mod, quote
            for j in $$set
                $$indx = j
                z = $$Z
                va = $$loaded_vals[j]
                Main.lv_set(z[$$indx_2], va)
            end
        end)
    end
end

macro Constant(model, main_obj, index_set)
    mod = esc(model)
    param_name = QuoteNode(main_obj.args[1].args[1])
    param_indx = QuoteNode(main_obj.args[1].args[2])
    vals_loaded = esc(main_obj.args[2].args[1])
    vals_loaded_indx = QuoteNode(main_obj.args[2].args[2])
    indx = QuoteNode(index_set.args[1])
    set = esc(index_set.args[2])
    quote
        Base.eval($mod, quote
            for j in $$set
                $$indx = j
                base_model.parameters[$$param_name][$$param_indx].is_const = true
                base_model.parameters[$$param_name][$$param_indx].value.value = $$vals_loaded[$$vals_loaded_indx]
            end
        end)
    end

end


macro Parameter(model, main_obj, index_set)
    mod = esc(model)
    param_name = QuoteNode(main_obj.args[1])
    param_indx = QuoteNode(main_obj.args[2])
    indx = QuoteNode(index_set.args[1])
    set = esc(index_set.args[2])
    quote
        Base.eval($mod, quote
            local to_return = Dict()
            for j in $$set
                $$indx = j
                to_return[j] = base_model.parameters[$$param_name][$$param_indx].value.value
            end
            return to_return
        end)
    end

end

macro Parameter(model, main_obj)
    mod = esc(model)
    param_name = QuoteNode(main_obj)
    quote
        Base.eval($mod, quote
            if typeof($$param_name) == Symbol
                local to_return = Dict()
                for (key, val) in base_model.parameters[$$param_name].parameter_map
                    to_return[key] = val.value.value
                end
                return to_return
            elseif typeof($$param_name) == Main.CategoricalZ
                return $$param_name.Pi
            else
                local to_return = Dict()
                for (key, val) in $$param_name.set
                    to_return[key] = val.Pi
                end
                return to_return
            end
        end)
    end

end


function EM!(model; args...)
    space = eval(:($model))
    Base.eval(space, quote
       local X::Vector{Main.Observation} = collect(values(X_val))
       fit_model = Main.LMEM(X, base_model; $args...)
       return nothing
    end)
end

macro posterior_probability(model, conditions, index_set)
    mod = esc(model)
    indx = QuoteNode(index_set.args[1])
    set = esc(index_set.args[2])
    cond = QuoteNode(conditions)
    quote
        Base.eval($mod, quote
            local post = Vector{}(undef, length($$set))
            for (i, j) in enumerate($$set)
                $$indx = j
                local X::Vector{Main.Observation} = collect(values(X_val))
                post[i] = Main.posterior_probability(() -> $$cond, X, base_model)
            end
            return post
        end)
    end
end

macro max_posterior(model, conditions, index_set)
    mod = esc(model)
    indx = QuoteNode(index_set.args[1])
    set = esc(index_set.args[2])
    cond = QuoteNode(conditions)
    quote
        Base.eval($mod, quote
            local post = Vector{}(undef, length($$set))
            for (i, j) in enumerate($$set)
                $$indx = j
                local X::Vector{Main.Observation} = collect(values(X_val))
                post[i] = Main.max_posterior(Main.posterior_probability(() -> $$cond, X, base_model))
                if length(post[i]) == 1
                    post[i] = post[i][1]
                end
            end
            return (post)
        end)
    end
end


macro BIC(model)
    mod = esc(model)
    quote
        Base.eval($mod, quote
          log_lik = fit_model.log_likelihood()
          local M = 0
          for (_, gen) in base_model.parameters
            for (_, par) in gen.parameter_map
                if !par.is_const
                    M = M + par.value.n_parameters
                end
            end
          end
          Float64(M * log(fit_model.n) - 2 * log_lik)
    end)
    end
end


macro likelihood(model, conditions, index_set)
    mod = esc(model)
    indx = QuoteNode(index_set.args[1])
    set = esc(index_set.args[2])
    obs = string(conditions.args[1])
    quote
        Base.eval($mod, quote
            local post = Vector{Real}()
            for j in $$set
                $$indx = j
                local X = X_val[($$obs, $$indx)]
                post = [post; [Main.initialize_density_evaluation([X], base_model, false)()]]
            end
            return post
        end)
    end
end

macro likelihood(model)
    mod = esc(model)
    quote
        Base.eval($mod, quote
          log_lik = fit_model.log_likelihood()
          return Float64(log_lik)
    end)
    end
end

