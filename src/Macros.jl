
# notation outline

N = MvtNormal(p)
Z = Categorical(Z)
for i in 1:n
    text_X[i] ~ N(:mu => Z[i], :cov => Z[i])
end
EM!(N)
G = f(Z[1])


# N2 = MvtNormal(p)
# Z = Categorical(Z)
# for i in 1:n
#     text_X[i] ~ N2(:mu => Z[i], :cov => Z[i])
# end
# G = f(Z[1])

# model = ParsaBase(F = MvtNormal(p))
# @| model begin
#     Z = Categorical(3)
#     for i in 1:n
#         text_X[i] ~ @F(:mu => Z[i], :cov => Z[i])
#     end
#     for i in 1:n
#         text_X[i] ~ @F(:mu => Z[i], :cov => Z[i])
#     end
#     for i in 1:3
#         :mu[i] === ones(p)
#         Z[i] == 1
#     end
#     G = @f([Z[i] for i in 1:5]...)
#     Z[1]
#     :mu[1]
# end




# simply wrap in module to make all variables secluded


function ParsaBase(;F::Parsa_Base)
    space = Module()
    Base.include(space, "../src/Types.jl")
    Base.include(space, "../src/Models.jl")
    Base.include(space, "../src/Core.jl")
    Base.include(space, "../src/Macros.jl")
    Base.eval(space, quote
        base_model = $F
        fit_model = nothing
        X_val = Set()
        is_parsa_model = true
    end)
    return space
end

macro >(model, expr)
    mod = eval(model)
    Base.eval(mod, quote
        $expr
    end)
end


# macro F(map...)
#    ma = QuoteNode(map)
#    quote
#     for mm in $map
#         println(typeof(mm))
#     end
#    end
# end



# d

# # ParsaBase(Parsa_Base) = ParsaBase(;F = ParsaBase)

# macro |(model, expr...)
#     result_expr = quote end
#     mod = esc(model)
#     # expr = collect(expr)
#     for (ex_i, ex) in enumerate(expr)
#         if typeof(ex) == QuoteNode || typeof(ex) == Symbol
#             new = quote
#                 if $ex_i == length($expr)
#                     (@Parameter($mod, $ex))
#                 else
#                     display(@Parameter($mod, $ex))
#                 end
#             end
#             push!(result_expr.args, new)
#             continue
#         end
#         if length(ex.args) == 2
#             if string(ex.args[2].args[1]) == "Categorical"
#                 q1 = ex.args[1]
#                 q2 = esc(ex.args[2].args[2])
#                 new = quote
#                     (@Categorical($mod, $q1, $q2))
#                 end
#                 push!(result_expr.args, new)
#             elseif string(ex.head) == "="
#                 Z = (ex.args[1].args[1])
#                 indx = (ex.args[1].args[2].args[1])
#                 indx_set = ex.args[1].args[2].args[2]
#                 new_set = ex.args[2]
#                 t1 = :(($Z)[$indx])
#                 t2 = :($t1 = $new_set)
#                 t3 = :($indx = $indx_set)
#                 if typeof(Z) == Symbol
#                     new = quote
#                         @Initialize($mod, $t2, $t3)
#                     end
#                     push!(result_expr.args, new)
#                 end
#                 if typeof(Z) == QuoteNode
#                     new = quote
#                         @Constant_Init($mod, $t2, $t3)
#                     end
#                     push!(result_expr.args, new)
#                 end
#             end
#         end
#         if length(ex.args) == 3
#             if string(ex.args[3].args[1]) == "F"
#                 X = (ex.args[2].args[1])
#                 indx = (ex.args[2].args[2].args[1])
#                 indx_set = ex.args[2].args[2].args[2]
#                 map = Expr(:vect, ex.args[3].args[2:end]...)
#                 t1 = :(($X)[$indx])
#                 t2 = :($t1 = $t1 = $map)
#                 t3 = :($indx = $indx_set)
#                 new = quote
#                     @Observation($mod, $t2, $t3)
#                 end
#                 push!(result_expr.args, new)
#             elseif string(ex.args[1]) == "=="
#                 Z = (ex.args[2].args[1])
#                 if typeof(Z) == Symbol || typeof(Z) == QuoteNode
#                     indx = (ex.args[2].args[2].args[1])
#                     indx_set = ex.args[2].args[2].args[2]
#                     new_set = (ex.args[3])
#                     t1 = :(($Z)[$indx])
#                     t2 = :($t1 = $new_set)
#                     t3 = :($indx = $indx_set)
#                     if typeof(Z) == Symbol
#                         new = quote
#                             @Known($mod, $t2, $t3)
#                         end
#                         push!(result_expr.args, new)
#                     end
#                     if typeof(Z) == QuoteNode
#                         new = quote
#                             @Constant($mod, $t2, $t3)
#                         end
#                         push!(result_expr.args, new)
#                     end
#                 else
#                     Z = ex.args[2].args[1].args[1]
#                     indx_1 = ex.args[2].args[1].args[2].args[1]
#                     indx_set_1 = ex.args[2].args[1].args[2].args[2]
#                     indx_2 = ex.args[2].args[2].args[1]
#                     indx_set_2 = ex.args[2].args[2].args[2]
#                     new_set = ex.args[3]

#                     t1 = :(($Z)[$indx_1][$indx_2])
#                     t2 = :($t1 = $new_set)
#                     t3 = :($indx_1 = $indx_set_1)
#                     t4 = :($indx_2 = $indx_set_2)

#                     new = quote
#                             @Known($mod, $t2, $t3, $t4)
#                         end
#                     push!(result_expr.args, new)
#                 end
#             end
#         end
#         if string(ex.args[1]) == "f"
#             if length(ex.args[2:end]) > 0
#                 Z = (ex.args[2].args[1])
#                 indx = (ex.args[2].args[2].args[1])
#                 indx_set = (ex.args[2].args[2].args[2])
#                 indx_set_esc = esc(ex.args[2].args[2].args[2])
#                 t1 = (:(($Z)[$indx]))
#                 t2 = (:($indx = $indx_set))
#                 Z = QuoteNode(ex.args[2].args[1])

#                 indx_esc = [esc(e.args[2].args[2]) for e in ex.args[2:end]]
#                 Zs = [e.args[1] for e in ex.args[2:end]]
#                 new = quote
#                     if !isdefined($mod, $Z)
#                         @likelihood($mod, $t1, $t2)
#                     else
#                         t3 = Expr(:vect, [:(($$Z)[$i___]) for i___ in $indx_set_esc]...)
#                         indxes = [[:($z[$i]) for i in r] for (r, z) in zip($(Expr(:vect, indx_esc...)), $Zs)]
#                         indxes = Expr(:vect, reduce(vcat, indxes)...)
#                         Base.eval($mod, quote
#                             local X::Vector{$$Observation} = collect(values(X_val))
#                             post = $$posterior_initalize(() -> $indxes, X, base_model)
#                             return post
#                         end)
#                     end
#                 end
#                 push!(result_expr.args, new)
#             end
#         end
#     end
#     return result_expr
# end

# macro Categorical(model, name, K)
#     mod = esc(model)
#     K_val = esc(K)
#     na = QuoteNode(name)
#     str = string(name)
#     quote
#         if length($K_val) == 1
#             if !(typeof($mod) == Module)
#                 error("First parameter must be a valid module. Use ParsaBase function")
#             end
#             if !isdefined($mod, :is_parsa_model)
#                 error("First parameter must be a valid module. Use ParsaBase function")
#             end
#             if !(typeof($K_val) == Int)
#                 error("Third parameter must be an integer.")
#             end
#             Base.eval($mod, quote
#                 $$na = $$CategoricalZ(K = $$K_val, name=$$str)
#             end)
#         else
#             if isa(($K_val)[1], Pair)
#                 @Categorical_Set($mod, $name, $K_val)
#             else
#                 Base.eval($mod, quote
#                     $$na = $$CategoricalZ(K = length($$K_val), name=$$str)
#                     $$na.Pi = $$K_val
#                     $$na.constant = true
#                 end)
#             end
#         end
#     end
# end

# macro Categorical_Set(model, name, K)
#     mod = esc(model)
#     na = QuoteNode(name)
#     name = string(name)
#     K_val = esc(K)
#     quote
#         # if !(typeof($mod) == Module)
#         #     error("First parameter must be a valid module. Use ParsaBase function")
#         # end
#         # if !isdefined($mod, :is_parsa_model)
#         #     error("First parameter must be a valid module. Use ParsaBase function")
#         # end
#         # if !(typeof($K_val) == Vector{Int})
#         #     error("Third parameter must be a vector of integers.")
#         # end
#         # if !(typeof($K_val) == Vector{Vector})
#         #     error("Third parameter must be a vector of vectors")
#         # end
#         # Base.eval($mod, quote
#         #     $$na = $$CategoricalZset()
#         #     for (ind, k) in zip(1:length($$K_val), $$K_val)
#         #         $$na.set[[ind]] = $$CategoricalZ(K = k)
#         #     end
#         # end)
#         Base.eval($mod, quote
#         $$na = $$CategoricalZset()
#         for (ind, k) in $$K_val
#             if !isa(ind, AbstractVector)
#                 ind = [ind]
#             end
#             if isa(k, Vector)
#                 $$na.set[ind] = $$CategoricalZ(K = length(k), name=$$name*"["*string(ind)*"]")
#                 $$na.set[ind].Pi = k
#                 $$na.set[ind].constant = true
#             else
#                 $$na.set[ind] = $$CategoricalZ(K = k, name=$$name*"["*string(ind)*"]")
#             end
#         end
#     end)
#     end
# end

# macro Observation(model, main_obj, index_set)
#     mod = esc(model)
#     s1 = string(main_obj.args[1])
#     s2 = string(main_obj.args[2].args[1])
#     obs_name = string(main_obj.args[1].args[1])
#     obs_indx = QuoteNode(main_obj.args[1].args[2])
#     X_name = main_obj.args[2].args[1].args[1]
#     X_loaded = esc(esc(X_name))
#     map = QuoteNode(main_obj.args[2].args[2])
#     indx = QuoteNode(index_set.args[1])
#     set = esc(esc(index_set.args[2]))

#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use ParsaBase function")
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
#                 if !($$mod.base_model.is_valid_input(va.X))
#                     error("Invalid observation type for density")
#                 end
#                 local tt = $$T([() -> Dict((() -> $$map)())], () -> Dict((() -> $$map)()))
#                 # local ob = $$Observation(va, tt)
#                 ob = va
#                 ob.T = tt
#                 # $$SetDependentX(ob, Vector{}(), ($$obs_name, j))
#                 domains = $$GetDependentVariable(ob)
#                 for LV in domains
#                     LV.dependent_X[($$obs_name, j)] = ob
#                 end
#                 # if typeof(domains) != Vector{$$LatentVaraible}
#                 #     domains = reduce(vcat, domains)
#                 #     domains_new = []
#                 #     for D in domains
#                 #         if typeof(D) == $$CatsegoricalZVec
#                 #             domains_new = [domains_new; D.inside; D.outside]
#                 #         else
#                 #             domains_new = [domains_new; D]
#                 #         end
#                 #     end
#                 #     domains = domains_new
#                 # end
#                 global X_val[($$obs_name, j)] = ob
#             end
#         end)
#     end
# end



# macro ObservationUpdater(model, main_obj, index_set)
#     mod = esc(model)
#     s1 = string(main_obj)
#     obs_name = string(main_obj.args[1])
#     obs_indx = QuoteNode(main_obj.args[2])
#     indx = QuoteNode(index_set.args[1])
#     set = esc(index_set.args[2])
#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         re = r"^[a-zA-Z][a-zA-Z0-9_]*\[[a-zA-Z]*\]$"
#         re2 =r"^[a-zA-Z][a-zA-Z0-9_]*\[([a-zA-Z]*)\]$"
#         re3 =r"^[a-zA-Z]*$"
#         if match(re, $s1) == nothing
#             error("invalid notation")
#         end
#         if match(re3, string($indx)) == nothing
#             error("Invalid index variable")
#         end
#         if string($indx) != match(re2, $s1)[1]
#             error("unknown " * match(re2, $s1)[1])
#         end
#         Base.eval($mod, quote
#             ob = Vector{$$Observation}(undef, length($$set))
#             for (i, j) in enumerate($$set)
#                 ob[i] = X_val[($$obs_name, j)]
#             end
#             return function(x)
#                 for i in eachindex(x)
#                     ob[i].X = x[i]
#                 end
#             end
#         end)
#     end
# end


# macro Known(model, eq, index_set)
#     mod = esc(model)
#     name = eq.args[1]
#     vals = eq.args[2]
#     s1 = string(name)
#     s2 = string(vals)
#     loaded_vals = esc(esc(vals.args[1]))
#     vals_indx = QuoteNode(vals.args[2])
#     indx = QuoteNode(index_set.args[1])
#     indx_2 = QuoteNode(name.args[2])
#     set = esc(esc(index_set.args[2]))
#     Z = QuoteNode(name.args[1])
#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use ParsaBase function")
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
#         # tp = LatentVaraible
#         Base.eval($mod, quote
#             for j in $$set
#                 $$indx = j
#                 z = $$Z
#                 va = $$loaded_vals[$$vals_indx]
#                 ind = $$indx_2
#                 if !isa(ind, AbstractVector)
#                     ind = [ind]
#                 end
#                 z[ind...] = $$LatentVaraible(z, va, ind)
#             end
#         end)
#     end
# end

# macro Known(model, eq, index_set, index_set2)
#     mod = esc(model)
#     name = eq.args[1]
#     vals = eq.args[2]
#     s1 = string(name)
#     s2 = string(vals)
#     loaded_vals = esc(esc(vals.args[1].args[1]))
#     vals_indx1 = QuoteNode(vals.args[1].args[2])
#     vals_indx2 = QuoteNode(vals.args[2])
#     indx = QuoteNode(index_set.args[1])
#     indx2 = QuoteNode(index_set2.args[1])
#     indx_2 = QuoteNode(name.args[1].args[2])
#     indx2_2 = QuoteNode(name.args[2])
#     set = esc(esc(index_set.args[2]))
#     set2 = esc(esc(index_set2.args[2]))
#     Z = QuoteNode(name.args[1].args[1])
#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         re = r"^[(][a-zA-Z][a-zA-Z0-9_]*\[[a-zA-Z]*\][)]\[[a-zA-Z]*\]$"
#         re2 = r"^[(][a-zA-Z][a-zA-Z0-9_]*\[([a-zA-Z]*)\][)]\[[a-zA-Z]*\]$"
#         re3 = r"^[(][a-zA-Z][a-zA-Z0-9_]*\[[a-zA-Z]*\][)]\[([a-zA-Z]*)\]$"
#         re4 = r"^[a-zA-Z]*$"
#         if match(re, $s1) == nothing || match(re, $s2) == nothing
#             error("invalid notation")
#         end
#         if match(re4, string($indx)) == nothing
#             error("Invalid index variable")
#         end
#         if string($indx) != match(re2, $s1)[1]
#             error("unknown " * match(re2, $s1)[1])
#         end
#         if string($indx) != match(re2, $s2)[1]
#             error("unknown " * match(re2, $s2)[1])
#         end
#         if string($indx2) != match(re3, $s1)[1]
#             error("unknown " * match(re3, $s1)[1])
#         end
#         if string($indx2) != match(re3, $s2)[1]
#             error("unknown " * match(re3, $s2)[1])
#         end
#         Base.eval($mod, quote
#             for j_within_it in $$set
#                 $$indx = j_within_it
#                 for j_within_it2 in $$set2
#                     $$indx2 = j_within_it2
#                     ind = j_within_it
#                     if !isa(ind, AbstractVector)
#                         ind = [ind]
#                     end
#                     ind2 = j_within_it2
#                     if !isa(ind2, AbstractVector)
#                         ind2 = [ind2]
#                     end
#                     z = $$Z
#                     va = $$loaded_vals[$$vals_indx1][$$vals_indx2]
#                     zz = z[ind...]
#                     zz[ind2...] = $$LatentVaraible(zz, va, ind2)
#                     # z = $$Z
#                     # va = $$loaded_vals[$$vals_indx1][$$vals_indx2]
#                     # zz = z[$$indx...].outside[1]
#                     # zz[$$indx2_2...] = $$LatentVaraible(zz, va)
#                 end
#             end
#         end)
#     end
# end


# macro Initialize(model, eq, index_set)
#     mod = esc(model)
#     name = eq.args[1]
#     vals = eq.args[2]
#     s1 = string(name)
#     s2 = string(vals)
#     loaded_vals = esc(esc(vals.args[1]))
#     indx = QuoteNode(index_set.args[1])
#     indx_2 = QuoteNode(name.args[2])
#     set = esc(esc(index_set.args[2]))
#     Z = QuoteNode(name.args[1])
#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use ParsaBase function")
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
#                 z = $$Z
#                 va = $$loaded_vals[j]
#                 # $$lv_set(z[$$indx_2].outside[1], va)
#                 # $$lv_set(z[$$indx_2], va)
#                 $$lv_set_init(z[$$indx_2], va)
#             end
#         end)
#     end
# end

# macro Constant(model, main_obj, index_set)
#     mod = esc(model)
#     s1 = string(main_obj.args[1])
#     s2 = string(main_obj.args[2])
#     st = main_obj.args[1].args[1]
#     param_name = QuoteNode(main_obj.args[1].args[1])
#     param_indx = QuoteNode(main_obj.args[1].args[2])
#     vals_loaded = esc(esc(main_obj.args[2].args[1]))
#     vals_loaded_indx = QuoteNode(main_obj.args[2].args[2])
#     indx = QuoteNode(index_set.args[1])
#     set = esc(esc(index_set.args[2]))
#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         re_sym = r"^.*\[[a-zA-Z]*\]$"
#         re = r"^[a-zA-Z][a-zA-Z0-9_]*\[[a-zA-Z]*\]$"
#         re2 =r"^[a-zA-Z][a-zA-Z0-9_]*\[([a-zA-Z]*)\]$"
#         re2_sym =r"^.*\[([a-zA-Z]*)\]$"
#         re3 =r"^[a-zA-Z]*$"
#         if (match(re_sym, $s1) == nothing || typeof($st) != Symbol) || match(re, $s2) == nothing
#             error("invalid notation")
#         end
#         if match(re3, string($indx)) == nothing
#             error("Invalid index variable")
#         end
#         if string($indx) != match(re2_sym, $s1)[1]
#             error("unknown " * match(re2, $s1)[1])
#         end
#         if string($indx) != match(re2_sym, $s2)[1]
#             error("unknown " * match(re2, $s2)[1])
#         end
#         Base.eval($mod, quote
#             for j in $$set
#                 $$indx = j
#                 base_model.parameters[$$param_name][$$param_indx...].is_const = true
#                 base_model.parameters[$$param_name][$$param_indx...].value.value = $$vals_loaded[$$vals_loaded_indx]
#             end
#         end)
#     end
# end

# macro Constant_Init(model, main_obj, index_set)
#     mod = esc(model)
#     s1 = string(main_obj.args[1])
#     s2 = string(main_obj.args[2])
#     st = main_obj.args[1].args[1]
#     param_name = QuoteNode(main_obj.args[1].args[1])
#     param_indx = QuoteNode(main_obj.args[1].args[2])
#     vals_loaded = esc(esc(main_obj.args[2].args[1]))
#     vals_loaded_indx = QuoteNode(main_obj.args[2].args[2])
#     indx = QuoteNode(index_set.args[1])
#     set = esc(esc(index_set.args[2]))
#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         re_sym = r"^.*\[[a-zA-Z]*\]$"
#         re = r"^[a-zA-Z][a-zA-Z0-9_]*\[[a-zA-Z]*\]$"
#         re2 =r"^[a-zA-Z][a-zA-Z0-9_]*\[([a-zA-Z]*)\]$"
#         re2_sym =r"^.*\[([a-zA-Z]*)\]$"
#         re3 =r"^[a-zA-Z]*$"
#         if (match(re_sym, $s1) == nothing || typeof($st) != Symbol) || match(re, $s2) == nothing
#             error("invalid notation")
#         end
#         if match(re3, string($indx)) == nothing
#             error("Invalid index variable")
#         end
#         if string($indx) != match(re2_sym, $s1)[1]
#             error("unknown " * match(re2, $s1)[1])
#         end
#         if string($indx) != match(re2_sym, $s2)[1]
#             error("unknown " * match(re2, $s2)[1])
#         end
#         Base.eval($mod, quote
#             for j in $$set
#                 $$indx = j
#                 base_model.parameters[$$param_name][$$param_indx].value.value = $$vals_loaded[$$vals_loaded_indx]
#             end
#         end)
#     end
# end


# macro Parameter(model, main_obj)
#     mod = esc(model)
#     param_name = QuoteNode(main_obj)
#     # s_tmp = main_obj
#     s1 = string(main_obj)
#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         re = r"^[a-zA-Z][a-zA-Z0-9_]*$"
#         if $s1[1] != ':' && match(re, $s1) == nothing
#             error("invalid notation")
#         end

#         Base.eval($mod, quote
#             if typeof($$param_name) == Symbol
#                 local to_return = Dict()
#                 for (key, val) in base_model.parameters[$$param_name].parameter_map
#                     to_return[key] = val.value.value
#                 end
#                 return to_return
#             elseif typeof($$param_name) == $$CategoricalZ
#                 return $$param_name.Pi
#             else
#                 local to_return = Dict()
#                 for (key, val) in $$param_name.set
#                     to_return[key] = val.Pi
#                 end
#                 return to_return
#             end
#         end)
#     end

# end

# macro IndependentBy(model, name)
#     mod = esc(model)
#     na = (name)
#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         $mod.independent_by = [$mod.independent_by ;$mod.$na]
#     end
# end


# function EM!(model; args...)
#     space = eval(:($model))
#     if !(typeof(space) == Module)
#         error("First parameter must be a valid module. Use ParsaBase function")
#     end
#     if !isdefined(space, :is_parsa_model)
#         error("First parameter must be a valid module. Use ParsaBase function")
#     end
#     Base.eval(space, quote
#        local X::Vector{$Observation} = collect(values(X_val))
#        fit_model = $LMEM(X, base_model; independent_by = independent_by, $args...)
#        return nothing
#     end)
# end

# macro posterior_probability(model, conditions, index_set)
#     mod = esc(model)
#     indx = QuoteNode(index_set.args[1])
#     sets = esc(index_set.args[2])
#     cond = QuoteNode(conditions)
#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         re3 =r"^[a-zA-Z]*$"
#         if match(re3, string($indx)) == nothing
#             error("Invalid index variable")
#         end
#         Base.eval($mod, quote
#             post = Dict()
#             for j in $$sets
#                 $$indx = j
#                 local X::Vector{$$Observation} = collect(values(X_val))
#                 post[j] = $$posterior_initalize(() -> $$cond, X, base_model)
#             end
#             return function ()
#                 post_new = Dict()
#                 for (key, val) in post
#                     v = val()
#                     post_new[key] = (max = $$max_posterior(v)[1], probability = v)
#                 end
#                 return post_new
#             end
#         end)
#     end
# end

# macro BIC(model)
#     mod = esc(model)
#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         Base.eval($mod, quote
#           log_lik = fit_model.log_likelihood()
#           local M = 0
#           for (_, gen) in base_model.parameters
#             for (_, par) in gen.parameter_map
#                 if !par.is_const
#                     M = M + par.value.n_parameters
#                 end
#             end
#           end
#           Float64(M * log(fit_model.n) - 2 * log_lik)
#     end)
#     end
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

# macro likelihood(model, conditions, index_set)
#     mod = esc(model)
#     indx = QuoteNode(index_set.args[1])
#     set = esc(esc(index_set.args[2]))
#     obs = string(conditions.args[1])
#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         Base.eval($mod, quote
#             local X_collect = Vector{$$Observation}(undef, length($$set))
#             for (i___,j) in enumerate($$set)
#                 $$indx = j
#                 X_collect[i___] = X_val[($$obs, $$indx)]
#             end
#             domain_map = Dict([x => $$GetDependentVariable(x) for x in X_collect])
#             LL = $$initialize_density_evaluation(X_collect, Vector{}(), base_model, domain_map, Dict())
#             return LL
#         end)
#     end
# end


# macro likelihood(model)
#     mod = esc(model)
#     quote
#         if !(typeof($mod) == Module)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         if !isdefined($mod, :is_parsa_model)
#             error("First parameter must be a valid module. Use ParsaBase function")
#         end
#         Base.eval($mod, quote
#           log_lik = fit_model.log_likelihood()
#           return Float64(log_lik)
#     end)
#     end
# end

