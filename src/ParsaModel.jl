module ParsaModel

include("./Macros.jl")

export ParsaModel,
       Categorical,
       Categorical_Set,
       Observation,
       Known,
       Initialize,
       Constant,
       Parameter,
       EM!,
       posterior_probability,
       max_posterior,
       BIC,
       likelihood
end
