module ParsaModel

using LinearAlgebra, UnicodePlots, ProgressBars
include("./Types.jl")
include("./Models.jl")
include("./Core.jl")
include("./Macros.jl")

export Parsa_Model,
       @Categorical,
       @Categorical_Set,
       @Observation,
       @Known,
       @Initialize,
       @Constant,
       @Parameter,
       EM!,
       @posterior_probability,
    #    @max_posterior,
       @BIC,
       @likelihood,
    #    @posterior_probability_generator,
    #    @likelihood_generator,
    #    @max_posterior_generator,
       @ObservationUpdater,
       Normal_Model,
       Double_Normal_Model,
       Normal_Parsa_Model,
       Normal_Model_singular
end

# using LinearAlgebra, UnicodePlots, ProgressBars, Distributions, Clustering
# include("./Types.jl")
# include("./Models.jl")
# include("./Core.jl")
# include("./Macros.jl")


# n = 500
# p = 8
# K = 2
# mu = [ones(p) .+ i  for i in 1:n];
# cov = [diagm(ones(p)) .+ i^4 for i in 1:2];
# class_id = [Int(ceil(i / 5)) for i in 1:n];
# n_classes = length(unique(class_id))
# true_id = rand(1:2, n_classes);
# X = [vec(rand(MvNormal(mu[class_id[i]], cov[true_id[class_id[i]]]), 1)) for i in 1:n];

# model_test = Parsa_Model(Normal_Model(p));
# @Categorical(model_test, class, n_classes)
# @Known(model_test, class[i] = class_id[i], i=1:n)
# @Categorical(model_test, Z, K)
# @Observation(model_test, Y[i] = X[i] = (:mu => class[i], :cov => Z[class[i]]), i = 1:n)
# EM!(model_test; n_init=1, n_wild=1)
# id = @posterior_probability(model_test, [Z[i]], i=1:n_classes)();
# id_ = [id[i] for i in 1:n_classes]
# randindex(Int.(id_), true_id)

# gen = @likelihood_generator(model_test, X[i] = (:mu => class[i], :cov => Z[class[i]]), i=1:5)


# p = 4
# K = 3
# n = 1000000
# true_id = rand(1:K, n);
# mu = [ones(p), ones(p) .+ 6, ones(p) .- 6];
# cov = [diagm(ones(p)), diagm(ones(p)), diagm(ones(p)) .+ 1];
# X = [vec(rand(MvNormal((mu[true_id[i]], cov[true_id[i]])...), 1)) for i in 1:n];

# model_test = Parsa_Model(Normal_Model(p));
# @Categorical(model_test, Z, K);
# @Observation(model_test, Y[i] = X[i] -> (:mu => Z[i], :cov => Z[i]), i = 1:n);
# EM!(model_test; n_init=1, n_wild=1, should_initialize=true, eps=10^-10);
# @Parameter(model_test, :cov)

# @Observation(model_test, G[i] = X[i] -> (:mu => Z[i, "T"], :cov => Z[i, "T"]), i=1)
# gen = @posterior_probability(model_test, [Z[i, "T"]], i = 1)
# upt = @ObservationUpdater(model_test, G[i], i = 1)
# LL = @likelihood(model_test, G[i], i = 1)

# pred(x) = (upt(x); gen()[1].max)
# lik(x) = (upt(x); LL())


# @profview [pred([X[i]]) for i in 1:n]
# @profview [lik([X[i]]) for i in 1:n]

# gen_2 = @max_posterior_generator(model_test, [Z[i, "T"]], (:mu => Z[i, "T"], :cov => Z[i, "T"]), i=1)


# ddd

# # struct testing2
# #     Int
# # end


# # @Observation(model_test, Y[i] = X[i] = (:mu => Z[i], :cov => Z[i]), i = 1:n);
# # @time @max_posterior(model_test, [Z[i]], i=1:n)


# n = 500
# p = 8
# K = 2
# mu = [ones(p) .+ i  for i in 1:n];
# cov = [diagm(ones(p)) .+ i^4 for i in 1:2];
# class_id = [Int(ceil(i / 5)) for i in 1:n];
# n_classes = length(unique(class_id))
# true_id = rand(1:2, n_classes);
# X = [vec(rand(MvNormal(mu[class_id[i]], cov[true_id[class_id[i]]]), 1)) for i in 1:n];

# model_test = Parsa_Model(Normal_Model(p));
# @Categorical(model_test, class, n_classes)
# @Known(model_test, class[i] = class_id[i], i=1:n)
# @Categorical(model_test, Z, K)
# @Observation(model_test, Y[i] = X[i] = (:mu => class[i], :cov => Z[class[i]]), i = 1:n)
# EM!(model_test; should_initialize=true)
# post_gen = @posterior_probability(model_test, [Z[i]], i=1:n_classes)()
# id_ = [post_gen[i].max for i in 1:n_classes]
# randindex(Int.(id_), true_id)



