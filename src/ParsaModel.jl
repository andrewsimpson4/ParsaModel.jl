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
       @max_posterior,
       @BIC,
       @likelihood,
       Normal_Model,
       Double_Normal_Model,
       Normal_Parsa_Model
end

# using LinearAlgebra, UnicodePlots, ProgressBars, Distributions
# include("./Types.jl")
# include("./Models.jl")
# include("./Core.jl")
# include("./Macros.jl")
# p = 4
# K = 3
# n = 600
# true_id = rand(1:K, n);
# mu = [ones(p), ones(p) .+ 6, ones(p) .- 6];
# cov = [diagm(ones(p)), diagm(ones(p)), diagm(ones(p)) .+ 1];
# X = [vec(rand(MvNormal((mu[true_id[i]], cov[true_id[i]])...), 1)) for i in 1:n];

# model_test = Parsa_Model(Normal_Model(p));
# @Categorical(model_test, Z, K);
# @IndependentBy(model_test, Z)
# @Observation(model_test, Y[i] = X[i] = (:mu => Z[i], :cov => Z[i]), i = 1:n);
# @time EM!(model_test; n_init=1, n_wild=1, should_initialize=true);
# @Parameter(model_test, :cov)

# NX = Dict("T" => X[1])
# @Observation(model_test, Y[i] = NX[i] = (:mu => Z[i], :cov => Z[i]), i = keys(NX));
# f = @max_posterior_initialize(model_test, [Z["T"]])

# predictor(x) = (NX["T"] = x; @UpdateObservation(model_test, Y[i] = NX[i], i = keys(NX)); f();)

# @time [predictor(X[i]) for i in 1:n]

# @time @max_posterior(model_test, [Z[i]], i=1:n)

# @UpdateObservation(model_test, Y[i] = Y[i], i = 1)

