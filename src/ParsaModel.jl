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
