using ParsaModel

using CSV, DataFrames, Clustering, Distances, LinearAlgebra, StatsBase

iris = CSV.read("./examples/datasets/Iris.csv", DataFrame)
iris_matrix = Matrix(iris[:, 2:5])
iris_m = Observation.(eachrow(iris_matrix));
n=size(iris_m)[1];
p=length(iris_m[1]);
class_string = vec(iris[:,6]);
mapping = Dict(val => i for (i, val) in enumerate(unique(class_string)));
class = [mapping[val] for val in class_string];

K = 3
model = Parsa_Model(Normal_Model(p));
@Categorical(model, Z, K);
@Observation(model, X[i] = iris_m[i] -> (:mu => Z[i], :cov => Z[i]), i = 1:n)
EM!(model; n_init=100, n_wild=3s0)
@Parameter(model, :mu)
@Parameter(model, :cov)
id = @posterior_probability(model, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)


model = Parsa_Model(F=Normal_Model(p));
@| model Z = Categorical(K) iris_m[i=1:n] ~ F(:mu => Z[i], :cov => Z[i])
EM!(model; n_init=10, n_wild=10)
@| model Z :mu :cov

id_ = [(@| model f(Z[i=j]))().max[1] for j in 1:n];
randindex(id_, class)

new_x = Dict(n+1 => Observation(zeros(p)));
ff = @| model  new_x[i=(n+1)] ~ F(:mu => Z[i], :cov => Z[i]) f(Z[i=(n+1)]);
id_ = [ (new_x[n+1].X = x.X; ff().max[1]) for x in iris_m];
randindex(id_, class)



iris_hclust = hclust(pairwise(Euclidean(), iris_matrix'), :ward)
init_id = cutree(iris_hclust, k=3)

K = 3
model = Parsa_Model(Normal_Model(p));
@Categorical(model, Z, K);
@Initialize(model, Z[i] = init_id[i], i = 1:n)
@Observation(model, X[i] = iris_m[i] = (:mu => Z[i], :cov => Z[i]), i = 1:n)
EM!(model; should_initialize=false)
id = @posterior_probability(model, [Z[i]], i = 1:n)();
id_ = [id[i].max for i in 1:n]
randindex(id_, class)


model = Parsa_Model(Normal_Model(p));
@|(model,
    Z = Categorical(K),
    Z[i=1:n] = init_id[i],
    iris_m[i=1:n] ~ F(:mu => Z[i], :cov => Z[i]))
EM!(model; should_initialize=false)

id_ = [(@| model f(Z[i=j]))().max[1] for j in 1:n];
randindex(id_, class)

new_x = Dict(n+1 => Observation(zeros(p)));
ff = @| model  new_x[i=(n+1)] ~ F(:mu => Z[i], :cov => Z[i]) f(Z[i=(n+1)]);
id_ = [(new_x[n+1].X = x.X; ff().max[1]) for x in iris_m];
randindex(id_, class)


K = 3
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, Z, K);
@Observation(model, X[i] = iris_m[i] = (:mu => Z[i], :a => Z[i], :L => Z[i], :V => 1), i = 1:n)
EM!(model; n_init=20, n_wild=30)
id = @posterior_probability(model, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)

K = 3
model = Parsa_Model(F=Normal_Parsa_Model(p));
@|( model,
    Z = Categorical(K),
    iris_m[i=1:n] ~ F(:mu => Z[i], :a => Z[i], :L => Z[i], :V => 1))
EM!(model; n_init=20, n_wild=30)

id_ = [(@| model f(Z[i=j]))().max[1] for j in 1:n];
randindex(id_, class)

new_x = Dict(n+1 => Observation(zeros(p)));
ff = @| model  new_x[i=(n+1)] ~ F(:mu => Z[i], :a => Z[i], :L => Z[i], :V => 1) f(Z[i=(n+1)]);
id_ = [(new_x[n+1].X = x.X; ff().max[1]) for x in iris_m];
randindex(id_, class)



K = 3
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, Z, K);
@Initialize(model, Z[i] = init_id[i], i = 1:n)
@Observation(model, X[i] = iris_m[i] = (:mu => Z[i], :a => Z[i], :L => 1, :V => 1), i = 1:n)
EM!(model; should_initialize=false)
id = @posterior_probability(model, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)

K = 3
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, Z, K);
@Observation(model, X[i] = iris_m[i] = (:mu => Z[i], :a => Z[i], :L => Z[i], :V => 1), i = 1:n)
const_V = [diagm(ones(4))];
@Constant(model, :V[i] = const_V[i], i = 1)
EM!(model; n_init=20, n_wild=30)
id = @posterior_probability(model, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)

model = Parsa_Model(F = Normal_Parsa_Model(p));
const_V = [diagm(ones(4))];
@|( model,
    Z = Categorical(K),
    iris_m[i=1:n] ~ F(:mu => Z[i], :a => Z[i], :L => Z[i], :V => 1),
    :V[i=1] == const_V[i]
)
EM!(model; n_init=20, n_wild=30)
@| model :V

id_ = [(@| model f(Z[i=j]))().max[1] for j in 1:n];
randindex(id_, class)

new_x = Dict(n+1 => Observation(zeros(p)));
ff = @| model  new_x[i=(n+1)] ~ F(:mu => Z[i], :a => Z[i], :L => Z[i], :V => 1) f(Z[i=(n+1)]);
id_ = [(new_x[n+1].X = x.X; ff().max[1]) for x in iris_m];
randindex(id_, class)



K = 3
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, Z, K);
@Observation(model, X[i] = iris_m[i] = (:mu => Z[i], :a => Z[i], :L => 1, :V => 1), i = 1:n)
const_V = [diagm(ones(4))];
@Constant(model, :V[i] = const_V[i], i = 1)
EM!(model; n_init=20, n_wild=30)
id = @posterior_probability(model, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)

model = Parsa_Model(F = Normal_Parsa_Model(p));
const_V = [diagm(ones(4))];
@|( model,
    Z = Categorical(K),
    iris_m[i=1:n] ~ F(:mu => Z[i], :a => Z[i], :L => 1, :V => 1),
    :V[i=1] == const_V[i]
)
EM!(model; n_init=20, n_wild=30)
@| model :V :L

id_ = [(@| model f(Z[i=j]))().max[1] for j in 1:n];
randindex(id_, class)

new_x = Dict(n+1 => Observation(zeros(p)));
ff = @| model  new_x[i=(n+1)] ~ F(:mu => Z[i], :a => Z[i], :L => 1, :V => 1) f(Z[i=(n+1)]);
id_ = [(new_x[n+1].X = x.X; ff().max[1]) for x in iris_m];
randindex(id_, class)



known_samples = sample(1:n, 30; replace=false)
known_map = Dict([s => class[s] for s in known_samples])
K = 3
model = Parsa_Model(Normal_Model(p));
@Categorical(model, Z, K);
@Known(model, Z[i] = known_map[i], i = known_samples)
@Observation(model, X[i] = iris_m[i] = (:mu => Z[i], :cov => Z[i]), i = 1:n)
EM!(model; n_init=10, n_wild=10)
id = @posterior_probability(model, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)

model = Parsa_Model(F = Normal_Model(p));
@|( model,
    Z = Categorical(K),
    Z[i=known_samples] == known_map[i],
    iris_m[i=1:n] ~ F(:mu => Z[i], :cov => Z[i])
)
EM!(model; n_init=10, n_wild=10)

id_ = [(@| model f(Z[i=j]))().max[1] for j in 1:n];
randindex(id_, class)

new_x = Dict(n+1 => Observation(zeros(p)));
ff = @| model  new_x[i=(n+1)] ~ F(:mu => Z[i], :cov => Z[i]) f(Z[i=(n+1)]);
id_ = [(new_x[n+1].X = x.X; ff().max[1]) for x in iris_m];
randindex(id_, class)


blocks = Int.(repeat(1:(n/2),inner=2))
n_blocks = length(unique(blocks))
true_class_block = [class[i] for i in 1:n if i % 2 == 0]
model = Parsa_Model(Normal_Model(p));
@Categorical(model, Z, K);
@Categorical(model, B, n_blocks);
@Known(model, B[i] = blocks[i], i = 1:n)
@Observation(model, X[i] = iris_m[i] = (:mu => Z[B[i]], :cov => Z[B[i]]), i = 1:n)
EM!(model; n_init=20, n_wild=30)
id = @posterior_probability(model, [Z[i]], i = 1:n_blocks)()
id_ = [id[i].max for i in 1:n_blocks]
randindex(id_, true_class_block)

model = Parsa_Model(F = Normal_Model(p));
@|( model,
    Z = Categorical(K),
    B = Categorical(n_blocks),
    B[i=1:n] == blocks[i],
    iris_m[i=1:n] ~ F(:mu => Z[B[i]], :cov => Z[B[i]])
)
EM!(model; n_init=10, n_wild=30)

id_ = [(@| model f(Z[i=j]))().max[1] for j in 1:n_blocks];
randindex(id_, true_class_block)

new_x = Dict(n+1 => Observation(zeros(p)));
ff = @| model  new_x[i=(n+1)] ~ F(:mu => Z[B[i]], :cov => Z[B[i]]) f(Z[i=(n+1)]);
id_ = [(new_x[n+1].X = x.X; ff().max[1]) for x in iris_m];
randindex(id_, class)

blocks = [1;1:(n-1)]
II = [1;2; repeat([1], 148)]
n_blocks = length(unique(blocks))
perms = reduce(vcat, [[[i,j] for i in 1:K if i != j] for j in 1:K])
model = Parsa_Model(Normal_Model(p));
@Categorical(model, Z, K);
@Categorical(model, B, n_blocks);
@Known(model, B[i] = blocks[i], i = 1:n)
@Categorical(model, P, Int.([repeat([2], length(perms))][1]));
@Known(model, P[i][j] = perms[i][j], i = 1:6, j=1:2)
@Categorical(model, I, 2)
@Known(model, I[i] = I[i], i = 1:n)
@Categorical(model, PP, 6)
@Observation(model, X[i] = iris_m[i] = (:mu => Z[P[PP[B[i]]][I[i]], i], :cov => Z[P[PP[B[i]]][I[i]], i]), i = 1:n)
EM!(model; n_init=1, n_wild=1)
perms[@posterior_probability(model, [PP[B[i]]], i = 1)()[1].max]

model = Parsa_Model(F = Normal_Model(p));
@|( model,
    Z = Categorical(K),
    B = Categorical(n_blocks),
    B[i=1:n] == blocks[i],
    P = Categorical(Int.([repeat([2], length(perms))][1])),
    P[i=1:6][j=1:2] == perms[i][j],
    I = Categorical(2),
    I[i=1:n] == II[i],
    PP = Categorical(6),
    iris_m[i=1:n] ~ F(:mu => Z[P[PP[B[i]]][I[i]], i], :cov => Z[P[PP[B[i]]][I[i]], i])
    )
EM!(model; n_init=1, n_wild=1)
perms[(@| model f(PP[i=1]))().max[1]]

K = 3
model = Parsa_Model(Normal_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Observation(model, X[i] = iris_m[i] = (:mu => class[i], :cov => class[i]), i = 1:n)
EM!(model; n_init=1, n_wild=1)
@Observation(model, X_new[i] = iris_m[i] = (:mu => class[i, "T"], :cov => class[i, "T"]), i = 1:n)
id = @posterior_probability(model, [class[i, "T"]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)

model = Parsa_Model(F = Normal_Model(p));
@|( model,
    class = Categorical(K),
    class[i=1:n] == class[i],
    iris_m[i=1:n] ~ F(:mu => class[i], :cov => class[i]))
EM!(model; n_init=1, n_wild=1)

new_obs = Dict([(i+n) => Observation(x.X) for (i,x) in enumerate(iris_m)])
@| model new_obs[i=((1:n) .+ n)] ~ F(:mu => class[i], :cov => class[i])
id_ = [(@| model f(class[i=j]))().max[1] for j in (1:n).+n];
mean(id_ .== class)


K = 3
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Observation(model, X[i] = iris_m[i] = (:mu => class[i], :a => class[i], :L => 1, :V => 1), i = 1:n)
const_V = [diagm(ones(4))];
@Constant(model, :V[i] = const_V[i], i = 1)
EM!(model; n_init=1, n_wild=1)
@Observation(model, X_new[i] = iris_m[i] = (:mu => class[i, "T"], :a => class[i, "T"], :L => 1, :V => 1), i = 1:n)
id = @posterior_probability(model, [class[i, "T"]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)


model = Parsa_Model(F = Normal_Parsa_Model(p));
const_V = [diagm(ones(4))];
@|( model,
    class = Categorical(K),
    class[i=1:n] == class[i],
    iris_m[i=1:n] ~ F(:mu => class[i], :a => class[i], :L => 1, :V => 1),
    :V[i=1]=const_V[i])
EM!(model; n_init=1, n_wild=1)

new_obs = Dict([(i+n) => Observation(x.X) for (i,x) in enumerate(iris_m)])
@| model new_obs[i=((1:n) .+ n)] ~ F(:mu => class[i], :a => class[i], :L => 1, :V => 1)
id_ = [(@| model f(class[i=j]))().max[1] for j in (1:n).+n];
mean(id_ .== class)


model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Observation(model, X[i] = iris_m[i] -> (:mu => class[i],
                                         :a => class[i],
                                         :L => class[i],
                                         :V => 1), i = 1:n)
EM!(model; n_init=1, n_wild=1)

K = 3
model = Parsa_Model(Normal_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Categorical(model, Z, [2,2,2]);
@Observation(model, X[i] = iris_m[i] = (:mu => [class[i], Z[class[i]][i]], :cov => [class[i], Z[class[i]][i]]), i = 1:n)
EM!(model; n_init=10, n_wild=10)
@Observation(model, X_new[i] = iris_m[i] = (:mu => [class[i, "T"], Z[class[i, "T"]][i, "T"]], :cov => [class[i, "T"], Z[class[i, "T"]][i, "T"]]), i = 1:n)
id = @posterior_probability(model, [class[i, "T"]], i = 1:n)();
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)
@Parameter(model, :cov)

model = Parsa_Model(F = Normal_Model(p));
@|( model,
    class = Categorical(K),
    class[i=1:n] == class[i],
    Z = Categorical([2,2,2]),
    iris_m[i=1:n] ~ F(:mu => [class[i], Z[class[i]][i]], :cov => [class[i], Z[class[i]][i]]))
EM!(model; n_init=1, n_wild=1)

new_obs = Dict([(i+n) => Observation(x.X) for (i,x) in enumerate(iris_m)])
@| model new_obs[i=((1:n) .+ n)] ~ F(:mu => [class[i], Z[class[i]][i]], :cov => [class[i], Z[class[i]][i]])
id_ = [(@| model f(class[i=j]))().max[1] for j in (1:n).+n];
mean(id_ .== class)


K = 3
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Categorical(model, Z, [2,2,2]);
@Observation(model, X[i] = iris_m[i] = (:mu => [class[i], Z[class[i]][i]], :a => [class[i], Z[class[i]][i]], :L => [class[i], Z[class[i]][i]], :V => 1), i = 1:n)
EM!(model; n_init=10, n_wild=10)
@Observation(model, X_new[i] = iris_m[i] = (:mu => [class[i, "T"], Z[class[i, "T"]][i, "T"]], :a => [class[i, "T"], Z[class[i, "T"]][i, "T"]], :L => [class[i, "T"], Z[class[i, "T"]][i, "T"]], :V => 1), i = 1:n)
id = @posterior_probability(model, [class[i, "T"]], i = 1:n)();
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)


model = Parsa_Model(F = Normal_Parsa_Model(p));
const_V = [diagm(ones(4))];
@|( model,
    class = Categorical(K),
    class[i=1:n] == class[i],
    Z = Categorical([2,2,2]),
    iris_m[i=1:n] ~ F(:mu => [class[i], Z[class[i]][i]], :a => [class[i], Z[class[i]][i]], :L => [class[i], Z[class[i]][i]], :V => 1),
    :V[i=1]=const_V[i])
EM!(model; n_init=1, n_wild=1)

new_obs = Dict([(i+n) => Observation(x.X) for (i,x) in enumerate(iris_m)])
@| model new_obs[i=((1:n) .+ n)] ~ F(:mu => [class[i], Z[class[i]][i]], :a => [class[i], Z[class[i]][i]], :L => [class[i], Z[class[i]][i]], :V => 1)
id_ = [(@| model f(class[i=j]))().max[1] for j in (1:n).+n];
mean(id_ .== class)


K = 3
model = Parsa_Model(Normal_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Categorical(model, Z, [2,2,2]);
@Categorical(model, cov, 2);
@Observation(model, X[i] = iris_m[i] = (:mu => [class[i], Z[class[i]][i]], :cov => cov[class[i], Z[class[i]][i]]), i = 1:n)
EM!(model; n_init=3, n_wild=10)
G = @posterior_probability(model, [cov[i]], i = reduce(vcat, [[[i,j] for i in 1:K] for j in 1:2]))()
for (key, M) in G
    mm = Dict(key => M.max)
    @Known(model, cov[i] = mm[i], i = [key])
end
@Observation(model, X_new[i] = iris_m[i] = (:mu => [class[i, "T"], Z[class[i, "T"]][i, "T"]], :cov => cov[class[i, "T"], Z[class[i, "T"]][i, "T"]]), i = 1:n)
id = @posterior_probability(model, [class[i, "T"]], i = 1:n)();
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)


model = Parsa_Model(F = Normal_Model(p));
@|( model,
    class = Categorical(K),
    class[i=1:n] == class[i],
    Z = Categorical([2,2,2]),
    cov = Categorical(2),
    iris_m[i=1:n] ~ F(:mu => [class[i], Z[class[i]][i]], :cov => cov[class[i], Z[class[i]][i]]))
EM!(model; n_init=1, n_wild=1)
G = Dict([j => (@| model f(cov[i=[j]]))() for j in reduce(vcat, [[[i,j] for i in 1:K] for j in 1:2])]);
for (key, M) in G
    mm = Dict(key => M.max[1])
    @| model cov[i=[key]] == mm[i]
end
new_obs = Dict([(i+n) => Observation(x.X) for (i,x) in enumerate(iris_m)])
@| model new_obs[i=((1:n) .+ n)] ~ F(:mu => [class[i], Z[class[i]][i]], :cov => [class[i], Z[class[i]][i]])
id_ = [(@| model f(class[i=j]))().max[1] for j in (1:n).+n];
mean(id_ .== class)




K = 3
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Categorical_Set(model, Z, [2,2,2]);
@Categorical(model, cov, 2);
@Observation(model, X[i] = iris_m[i] -> (:mu => [class[i], Z[class[i]][i]], :a => cov[class[i], Z[class[i]][i]], :L => cov[class[i], Z[class[i]][i]], :V => 1), i = 1:n)
EM!(model; n_init=10, n_wild=10)
G = @posterior_probability(model, [cov[i]], i = reduce(vcat, [[[i,j] for i in 1:K] for j in 1:2]))()
for (key, M) in G
    mm = Dict(key => M.max)
    @Known(model, cov[i] = mm[i], i = [key])
end
@Observation(model, X_new[i] = iris_m[i] = (:mu => [class[i, "T"], Z[class[i, "T"]][i, "T"]], :a => cov[class[i, "T"], Z[class[i, "T"]][i, "T"]], :L => cov[class[i, "T"], Z[class[i, "T"]][i, "T"]], :V => 1), i = 1:n)
id = @posterior_probability(model, [class[i, "T"]], i = 1:n)();
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)


model = Parsa_Model(F = Normal_Parsa_Model(p));
@|( model,
    class = Categorical(K),
    class[i=1:n] == class[i],
    Z = Categorical([2,2,2]),
    cov = Categorical(2),
    iris_m[i=1:n] ~ F(:mu => [class[i], Z[class[i]][i]], :a => cov[class[i], Z[class[i]][i]], :L => cov[class[i], Z[class[i]][i]], :V => 1))
EM!(model; n_init=1, n_wild=1)
@| model :L :V
G = Dict([j => (@| model f(cov[i=[j]]))() for j in reduce(vcat, [[[i,j] for i in 1:K] for j in 1:2])]);
for (key, M) in G
    mm = Dict(key => M.max[1])
    @| model cov[i=[key]] == mm[i]
end
new_obs = Dict([(i+n) => Observation(x.X) for (i,x) in enumerate(iris_m)])
@| model new_obs[i=((1:n) .+ n)] ~ F(:mu => [class[i], Z[class[i]][i]], :a => cov[class[i], Z[class[i]][i]], :L => cov[class[i], Z[class[i]][i]], :V => 1)
id_ = [(@| model f(class[i=j]))().max[1] for j in (1:n).+n];
mean(id_ .== class)






# using ParsaModel

using LinearAlgebra, UnicodePlots, ProgressBars, Distributions, Clustering
include("../src/Types.jl")
include("../src/Models.jl")
include("../src/Core.jl")
include("../src/Macros.jl")

n = 200
p = 5
K = 2
mu = [ones(p) .+ i  for i in 1:n];
cov = [diagm(ones(p)) .+ i^4 for i in 1:2];
class_id = [Int(ceil(i / 5)) for i in 1:n];
n_classes = length(unique(class_id))
true_id = rand(1:2, n_classes);
X = [Observation(vec(rand(MvNormal(mu[class_id[i]], cov[true_id[class_id[i]]]), 1))) for i in 1:n];


model = Parsa_Model(F=Normal_Model(p));
@|( model,
    class = Categorical(n_classes),
    class[i=1:n] == class_id[i],
    Z = Categorical(K),
    X[i=1:n] ~ F(:mu => class[i], :cov => Z[class[i]]));
EM!(model; n_init=10, n_wild=10)
@| model Z :mu :cov


n = 500
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


model = Parsa_Model(F = Normal_Model(p));

macro |(tt1)
    println(string(tt1))
    # println(string(tt2))
    # println(string(tt3))
end

K = 3
model = Parsa_Model(Normal_Model(p));
@Categorical(model, Z, K);
@Observation(model, X[i] = iris_m[i] -> [:mu => Z[i], :cov => Z[i]], i = 1:n)
EM!(model; n_init=100, n_wild=30)

@PM model Z = Categorical(K)
@PM model X[i=1:n] ~ F(:mu => Z[i], :cov => Z[i])
EM!(model; n_init=100, n_wild=30)


p=4
K=3
model = Parsa_Model(Normal_Model(p));
@| model Z = Categorical(K)
@| model iris_m[i=1:n] ~ F(:mu => Z[i], :cov => Z[i])
EM!(model; n_init=100, n_wild=30)
model.X_val

blocks = Int.(repeat(1:(n/2),inner=2))
n_blocks = length(unique(blocks))
true_class_block = [class[i] for i in 1:n if i % 2 == 0]

model = Parsa_Model(Normal_Model(p));
@Categorical(model, Z, K);
@Categorical(model, B, n_blocks);
@Known(model, B[i] = blocks[i], i = 1:n)
@Observation(model, X[i] = iris_m[i] = (:mu => Z[B[i]], :cov => Z[B[i]]), i = 1:n)
EM!(model; n_init=20, n_wild=30)
id = @posterior_probability(model, [Z[i]], i = 1:n_blocks)()
id_ = [id[i].max for i in 1:n_blocks]
randindex(id_, true_class_block)

blocks = Int.(repeat(1:(n/2),inner=2))
n_blocks = length(unique(blocks))
true_class_block = [class[i] for i in 1:n if i % 2 == 0]

model = Parsa_Model(Normal_Model(p));
@PM model Z = @Categorical(K)
@PM model B = @Categorical(n_blocks)
@PM model X[i=1:n] ~ F(:mu => Z[B[i]], :cov => Z[B[i]])
EM!(model; n_init=20, n_wild=30)
@PM model f(Z[i=1:n] | X[i])
@PM model f(X[i])
@PM model Z
@PM model :mu
@PM model :mu[i=1:n]
@PM model Z[i=1:n] = known[i]
@PM model :mu[i=1:n] = zz[i]
@PM model X[i]
@PM model Z[i=1:n] -> init[i]

