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
   println("    ____                      __  __           _      _
   |  _ \\ __ _ _ __ ___  __ _|  \\/  | ___   __| | ___| |
   | |_) / _` | '__/ __|/ _` | |\\/| |/ _ \\ / _` |/ _ \\ |
   |  __/ (_| | |  \\__ \\ (_| | |  | | (_) | (_| |  __/ |
   |_|   \\__,_|_|  |___/\\__,_|_|  |_|\\___/ \\__,_|\\___|_|")
   println("")
   println("")
   println("")

end

export ParsaBase,
      Observation,
      @|,
      EM!,
      MtvNormal,
      ParsimoniousNormal,
      Parameter,
      ParsaDensity,
      val
end