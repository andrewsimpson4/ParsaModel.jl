module ParsaModel

using LinearAlgebra, UnicodePlots, ProgressBars
include("./Types.jl")
include("./Models.jl")
include("./Core.jl")
include("./Macros.jl")

function __init__()
   println("")
   println("")
   println("")
   println("
                     \\       /
                        .-'-.
                  --  /     \\  --
   `~~^~^~^~^~^~^~^~^~^-=======-~^~^~^~~^~^~^~^~^~^~^~`
   `~^_~^~^~^~^~^_~^~^~^~^~^~^~^~^~^_~^~^~^~^~^~^_~^~^~`")
   println(" ____                      __  __           _      _
   |  _ \\ __ _ _ __ ___  __ _|  \\/  | ___   __| | ___| |
   | |_) / _` | '__/ __|/ _` | |\\/| |/ _ \\ / _` |/ _ \\ |
   |  __/ (_| | |  \\__ \\ (_| | |  | | (_) | (_| |  __/ |
   |_|   \\__,_|_|  |___/\\__,_|_|  |_|\\___/ \\__,_|\\___|_|")
   println("")
   println("")
   println("")

end

export Parsa_Model,
       @Categorical,
       @Observation,
       @Known,
       @Initialize,
       @Constant,
       @Parameter,
       EM!,
       @posterior_probability,
       @BIC,
       @likelihood,
       @ObservationUpdater,
       Normal_Model,
       Double_Normal_Model,
       Normal_Parsa_Model,
       Normal_Model_singular
end



