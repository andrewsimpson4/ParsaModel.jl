


function Parsa_Model(base_model::Parsa_Base)
    space = Module()
    Base.eval(space, quote
        base_model = $base_model
        fit_model = nothing
        X_val = Dict()
        is_parsa_model = true
        independent_by = Vector{$CategoricalZ}()
    end)
    return space
end

macro Categorical(model, name, K)
    mod = esc(model)
    K_val = esc(K)
    na = (name)
    quote
        if !(typeof($mod) == Module)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !isdefined($mod, :is_parsa_model)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !(typeof($K_val) == Int)
            error("Third parameter must be an integer.")
        end
        $mod.$na = CategoricalZ(K = $K_val)
    end
end

macro Categorical_Set(model, name, K, indx)
    mod = esc(model)
    na = QuoteNode(name)
    K_val = esc(K)
    ind = esc(indx)
    quote
        if !(typeof($mod) == Module)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !isdefined($mod, :is_parsa_model)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !(typeof($K_val) == Vector{Int})
            error("Third parameter must be a vector of integers.")
        end
        # if !(typeof($K_val) == Vector{Vector})
        #     error("Third parameter must be a vector of vectors")
        # end
        Base.eval($mod, quote
            $$na = $$CategoricalZset()
            for (ind, k) in zip($$ind, $$K_val)
                $$na.set[[ind]] = $$CategoricalZ(K = k)
            end
        end)
    end
end

macro Observation(model, main_obj, index_set)
    mod = esc(model)
    s1 = string(main_obj.args[1])
    s2 = string(main_obj.args[2].args[1])
    obs_name = string(main_obj.args[1].args[1])
    obs_indx = QuoteNode(main_obj.args[1].args[2])
    X_name = main_obj.args[2].args[1].args[1]
    X_loaded = esc(X_name)
    map = QuoteNode(main_obj.args[2].args[2])
    indx = QuoteNode(index_set.args[1])
    set = esc(index_set.args[2])

    quote
        if !(typeof($mod) == Module)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !isdefined($mod, :is_parsa_model)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        re = r"^[a-zA-Z][a-zA-Z0-9_]*\[[a-zA-Z]*\]$"
        re2 =r"^[a-zA-Z][a-zA-Z0-9_]*\[([a-zA-Z]*)\]$"
        re3 =r"^[a-zA-Z]*$"
        if match(re, $s1) == nothing || match(re, $s2) == nothing
            error("invalid notation")
        end
        if match(re3, string($indx)) == nothing
            error("Invalid index variable")
        end
        if string($indx) != match(re2, $s1)[1]
            error("unknown " * match(re2, $s1)[1])
        end
        if string($indx) != match(re2, $s2)[1]
            error("unknown " * match(re2, $s2)[1])
        end
        Base.eval($mod, quote
            for j in $$set
                $$indx = j
                va = ($$X_loaded[j])
                if !($$mod.base_model.is_valid_input(va))
                    error("Invalid observation type for density")
                end
                local tt = $$T([() -> Dict((() -> $$map)())], () -> Dict((() -> $$map)()))
                local ob = $$Observation(va, tt)
                global X_val[($$obs_name, j)] = ob
            end
        end)
    end
end

# macro UpdateObservation(model, main_obj, index_set)
#     mod = esc(model)
#     s1 = string(main_obj.args[1])
#     s2 = string(main_obj.args[2])
#     obs_name = string(main_obj.args[1].args[1])
#     obs_indx = QuoteNode(main_obj.args[1].args[2])
#     X_name = main_obj.args[2].args[1]
#     X_loaded = esc(X_name)
#     indx = QuoteNode(index_set.args[1])
#     set = esc(index_set.args[2])

#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use Parsa_Model function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use Parsa_Model function")
#         end
#         re = r"^[a-zA-Z][a-zA-Z0-9_]*\[[a-zA-Z]*\]$"
#         re2 =r"^[a-zA-Z][a-zA-Z0-9_]*\[([a-zA-Z]*)\]$"
#         re3 =r"^[a-zA-Z]*$"
#         if match(re, $s1) == nothing || match(re, $s2) == nothing
#             error("invalid notation")
#         end
#         if match(re3, string($indx)) == nothing
#             error("Invalid index variable")
#         end
#         if string($indx) != match(re2, $s1)[1]
#             error("unknown " * match(re2, $s1)[1])
#         end
#         if string($indx) != match(re2, $s2)[1]
#             error("unknown " * match(re2, $s2)[1])
#         end
#         Base.eval($mod, quote
#             for j in $$set
#                 $$indx = j
#                 va = ($$X_loaded[j])
#                 # if !($$mod.base_model.is_valid_input(va))
#                 #     error("Invalid observation type for density")
#                 # end
#                 # local tt = $$T([() -> Dict((() -> $$map)())], () -> Dict((() -> $$map)()))
#                 # local ob = $$Observation(va, tt)
#                 X_val[($$obs_name, j)].X = va
#             end
#         end)
#     end
# end

macro Known(model, eq, index_set)
    mod = esc(model)
    name = eq.args[1]
    vals = eq.args[2]
    s1 = string(name)
    s2 = string(vals)
    loaded_vals = esc(vals.args[1])
    vals_indx = QuoteNode(vals.args[2])
    indx = QuoteNode(index_set.args[1])
    indx_2 = QuoteNode(name.args[2])
    set = esc(index_set.args[2])
    Z = QuoteNode(name.args[1])
    quote
        if !(typeof($mod) == Module)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !isdefined($mod, :is_parsa_model)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        re = r"^[a-zA-Z][a-zA-Z0-9_]*\[[a-zA-Z]*\]$"
        re2 =r"^[a-zA-Z][a-zA-Z0-9_]*\[([a-zA-Z]*)\]$"
        re3 =r"^[a-zA-Z]*$"
        if match(re, $s1) == nothing || match(re, $s2) == nothing
            error("invalid notation")
        end
        if match(re3, string($indx)) == nothing
            error("Invalid index variable")
        end
        if string($indx) != match(re2, $s1)[1]
            error("unknown " * match(re2, $s1)[1])
        end
        if string($indx) != match(re2, $s2)[1]
            error("unknown " * match(re2, $s2)[1])
        end
        # tp = LatentVaraible
        Base.eval($mod, quote
            for j in $$set
                $$indx = j
                z = $$Z
                va = $$loaded_vals[$$vals_indx]
                z[$$indx_2] = $$LatentVaraible(z, va)
            end
        end)
    end
end

# macro Known(model, eq)
#     mod = esc(model)
#     name = eq.args[1]
#     vals = eq.args[2]
#     loaded_vals = esc(vals)
#     # vals_indx = esc(vals.args[2])
#     # indx = QuoteNode(index_set.args[1])
#     indx_2 = esc(name.args[2])
#     # set = esc(index_set.args[2])
#     Z = QuoteNode(name.args[1])
#     quote
#         # tp = LatentVaraible
#         Base.eval($mod, quote
#             z = $$Z
#             va = $$loaded_vals
#             z[$$indx_2] = $$LatentVaraible(z, va)
#         end)
#     end
# end

macro Initialize(model, eq, index_set)
    mod = esc(model)
    name = eq.args[1]
    vals = eq.args[2]
    s1 = string(name)
    s2 = string(vals)
    loaded_vals = esc(vals.args[1])
    indx = QuoteNode(index_set.args[1])
    indx_2 = QuoteNode(name.args[2])
    set = esc(index_set.args[2])
    Z = QuoteNode(name.args[1])
    quote
        if !(typeof($mod) == Module)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !isdefined($mod, :is_parsa_model)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        re = r"^[a-zA-Z][a-zA-Z0-9_]*\[[a-zA-Z]*\]$"
        re2 =r"^[a-zA-Z][a-zA-Z0-9_]*\[([a-zA-Z]*)\]$"
        re3 =r"^[a-zA-Z]*$"
        if match(re, $s1) == nothing || match(re, $s2) == nothing
            error("invalid notation")
        end
        if match(re3, string($indx)) == nothing
            error("Invalid index variable")
        end
        if string($indx) != match(re2, $s1)[1]
            error("unknown " * match(re2, $s1)[1])
        end
        if string($indx) != match(re2, $s2)[1]
            error("unknown " * match(re2, $s2)[1])
        end
        Base.eval($mod, quote
            for j in $$set
                $$indx = j
                z = $$Z
                va = $$loaded_vals[j]
                $$lv_set(z[$$indx_2], va)
            end
        end)
    end
end

macro Constant(model, main_obj, index_set)
    mod = esc(model)
    s1 = string(main_obj.args[1])
    s2 = string(main_obj.args[2])
    st = main_obj.args[1].args[1]
    param_name = QuoteNode(main_obj.args[1].args[1])
    param_indx = QuoteNode(main_obj.args[1].args[2])
    vals_loaded = esc(main_obj.args[2].args[1])
    vals_loaded_indx = QuoteNode(main_obj.args[2].args[2])
    indx = QuoteNode(index_set.args[1])
    set = esc(index_set.args[2])
    quote
        if !(typeof($mod) == Module)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !isdefined($mod, :is_parsa_model)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        re_sym = r"^.*\[[a-zA-Z]*\]$"
        re = r"^[a-zA-Z][a-zA-Z0-9_]*\[[a-zA-Z]*\]$"
        re2 =r"^[a-zA-Z][a-zA-Z0-9_]*\[([a-zA-Z]*)\]$"
        re2_sym =r"^.*\[([a-zA-Z]*)\]$"
        re3 =r"^[a-zA-Z]*$"
        if (match(re_sym, $s1) == nothing || typeof($st) != Symbol) || match(re, $s2) == nothing
            println(match(re_sym, $s1) == nothing)
            println(typeof($st) == Symbol)
            error("invalid notation")
        end
        if match(re3, string($indx)) == nothing
            error("Invalid index variable")
        end
        if string($indx) != match(re2_sym, $s1)[1]
            error("unknown " * match(re2, $s1)[1])
        end
        if string($indx) != match(re2_sym, $s2)[1]
            error("unknown " * match(re2, $s2)[1])
        end
        Base.eval($mod, quote
            for j in $$set
                $$indx = j
                base_model.parameters[$$param_name][$$param_indx].is_const = true
                base_model.parameters[$$param_name][$$param_indx].value.value = $$vals_loaded[$$vals_loaded_indx]
            end
        end)
    end

end


# macro Parameter(model, main_obj, index_set)
#     mod = esc(model)
#     param_name = QuoteNode(main_obj.args[1])
#     param_indx = QuoteNode(main_obj.args[2])
#     indx = QuoteNode(index_set.args[1])
#     set = esc(index_set.args[2])
#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use Parsa_Model function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use Parsa_Model function")
#         end
#         Base.eval($mod, quote
#             local to_return = Dict()
#             for j in $$set
#                 $$indx = j
#                 to_return[j] = base_model.parameters[$$param_name][$$param_indx].value.value
#             end
#             return to_return
#         end)
#     end

# end

macro Parameter(model, main_obj)
    mod = esc(model)
    param_name = QuoteNode(main_obj)
    # s_tmp = main_obj
    s1 = string(main_obj)
    quote
        if !(typeof($mod) == Module)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !isdefined($mod, :is_parsa_model)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        re = r"^[a-zA-Z][a-zA-Z0-9_]*$"
        if $s1[1] != ':' && match(re, $s1) == nothing
            error("invalid notation")
        end

        Base.eval($mod, quote
            if typeof($$param_name) == Symbol
                local to_return = Dict()
                for (key, val) in base_model.parameters[$$param_name].parameter_map
                    to_return[key] = val.value.value
                end
                return to_return
            elseif typeof($$param_name) == $$CategoricalZ
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

macro IndependentBy(model, name)
    mod = esc(model)
    na = (name)
    quote
        if !(typeof($mod) == Module)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !isdefined($mod, :is_parsa_model)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        $mod.independent_by = [$mod.independent_by ;$mod.$na]
    end
end


function EM!(model; args...)
    space = eval(:($model))
    if !(typeof(space) == Module)
        error("First parameter must be a valid module. Use Parsa_Model function")
    end
    if !isdefined(space, :is_parsa_model)
        error("First parameter must be a valid module. Use Parsa_Model function")
    end
    Base.eval(space, quote
       local X::Vector{$Observation} = collect(values(X_val))
       fit_model = $LMEM(X, base_model; independent_by = independent_by, $args...)
       return nothing
    end)
end

macro posterior_probability(model, conditions, index_set)
    mod = esc(model)
    indx = QuoteNode(index_set.args[1])
    set = esc(index_set.args[2])
    cond = QuoteNode(conditions)
    quote
        if !(typeof($mod) == Module)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !isdefined($mod, :is_parsa_model)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        re3 =r"^[a-zA-Z]*$"
        if match(re3, string($indx)) == nothing
            error("Invalid index variable")
        end
        Base.eval($mod, quote
            local post = Dict()
            for (i, j) in enumerate($$set)
                $$indx = j
                local X::Vector{$$Observation} = collect(values(X_val))
                post[i] = $$posterior_probability(() -> $$cond, X, base_model)
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
        if !(typeof($mod) == Module)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !isdefined($mod, :is_parsa_model)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        re3 =r"^[a-zA-Z]*$"
        if match(re3, string($indx)) == nothing
            error("Invalid index variable")
        end
        Base.eval($mod, quote
            local post = Dict()
            for (i, j) in enumerate($$set)
                $$indx = j
                local X::Vector{$$Observation} = collect(values(X_val))
                post[i] = $$max_posterior($$posterior_probability(() -> $$cond, X, base_model))
                # post[i] = $$max_posterior($$posterior_initalize(() -> $$cond, X, base_model)())
                if length(post[i]) == 1
                    post[i] = post[i][1]
                end
            end
            return (post)
        end)
    end
end

# macro max_posterior_initialize(model, conditions)
#     mod = esc(model)
#     cond = QuoteNode(conditions)
#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use Parsa_Model function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use Parsa_Model function")
#         end
#         Base.eval($mod, quote
#             local X::Vector{$$Observation} = collect(values(X_val))
#             post = $$posterior_initalize(() -> $$cond, X, base_model)
#             return function()
#                 pp = post()
#                 $$max_posterior(pp)
#             end
#         end)
#     end
# end


macro BIC(model)
    mod = esc(model)
    quote
        if !(typeof($mod) == Module)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !isdefined($mod, :is_parsa_model)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
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
        if !(typeof($mod) == Module)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !isdefined($mod, :is_parsa_model)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        Base.eval($mod, quote
            local post = Vector{Real}()
            for j in $$set
                $$indx = j
                local X = X_val[($$obs, $$indx)]
                post = [post; [$$initialize_density_evaluation([X], base_model, false)()]]
            end
            return post
        end)
    end
end

macro likelihood(model)
    mod = esc(model)
    quote
        if !(typeof($mod) == Module)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        if !isdefined($mod, :is_parsa_model)
            error("First parameter must be a valid module. Use Parsa_Model function")
        end
        Base.eval($mod, quote
          log_lik = fit_model.log_likelihood()
          return Float64(log_lik)
    end)
    end
end

