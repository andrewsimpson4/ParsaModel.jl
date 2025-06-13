using ParsaModel

using CSV, DataFrames, Clustering, Distances, LinearAlgebra, StatsBase, ProgressBars

include("../src/Types.jl")
include("../src/Models.jl")
include("../src/Core.jl")
include("../src/Macros.jl")

p = 4
K = 3
n = 100
true_id = rand(1:K, n);
mu = [ones(p), ones(p) .+ 6, ones(p) .- 6];
cov = [diagm(ones(p)), diagm(ones(p)), diagm(ones(p)) .+ 1];
X = [Observation(vec(rand(MvNormal(mu[true_id[i]], cov[true_id[i]]), 1))) for i in 1:n];

model = ParsaModel(F=Normal(p));
@| model Z = Categorical(3) X[i=1:n] ~ F(:mu => Z[i], :cov => Z[i])
EM!(model; n_init=1, n_wild=1)

model.Z[1].inside[1]

new_x = Dict(n+1 => Observation(zeros(p)));
ff = @| model  new_x[i=(n+1)] ~ F(:mu => Z[i], :cov => Z[i]) f(Z[i=(n+1)]);
gen(x) = (new_x[n+1].X = x; ff())
@time [gen(x.X) for x in X]

iris = CSV.read("./examples/datasets/Iris.csv", DataFrame)
iris_matrix = Matrix(iris[:, 2:5])
iris_m = Observation.(eachrow(iris_matrix));
n=size(iris_m)[1];
p=length(iris_m[1].X);
class_string = vec(iris[:,6]);
mapping = Dict(val => i for (i, val) in enumerate(unique(class_string)));
class = [mapping[val] for val in class_string];

K = 3
model = ParsaModel(F=Normal(p));
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
model = ParsaModel(F=Normal(p));
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
model = ParsaModel(F = ParsimoniousNormal(p));
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
model = ParsaModel(F = ParsimoniousNormal(p));
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
model = ParsaModel(F = ParsimoniousNormal(p));
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
model = ParsaModel(F = Normal(p));
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
model = ParsaModel(F = Normal(p));
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

K=3
blocks = [1;1:(n-1)]
II = [1;2; repeat([1], 148)]
n_blocks = length(unique(blocks))
perms = reduce(vcat, [[[i,j] for i in 1:K if i != j] for j in 1:K])
model = ParsaModel(F = Normal(p));
@|( model,
    B = Categorical(n_blocks),
    B[i=1:n] == blocks[i],
    P = Categorical([i => 3 for i in 1:6]),
    P[i=1:6][j=1:2] == perms[i][j],
    I = Categorical(2),
    I[i=1:n] == II[i],
    PP = Categorical(6),
    iris_m[i=1:n] ~ F(:mu => P[PP[B[i]]][I[i]], :cov => P[PP[B[i]]][I[i]])
    )
EM!(model; n_init=1, n_wild=1)
@| model :mu :cov
perms[(@| model f(PP[i=1]))().max[1]]



K = 3
model = ParsaModel(F = Normal(p));
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
model = ParsaModel(F = ParsimoniousNormal(p));
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


K = 3
model = ParsaModel(F = Normal(p));
@|( model,
    class = Categorical(K),
    class[i=1:n] == class[i],
    Z = Categorical([1=>2,2=>2,3=>2]),
    iris_m[i=1:n] ~ F(:mu => [class[i], Z[class[i]][i]], :cov => [class[i], Z[class[i]][i]]))
EM!(model; n_init=1, n_wild=1)
@| model :cov

new_obs = Dict([(i+n) => Observation(x.X) for (i,x) in enumerate(iris_m)])
@| model new_obs[i=((1:n) .+ n)] ~ F(:mu => [class[i], Z[class[i]][i]], :cov => [class[i], Z[class[i]][i]])
id_ = [(@| model f(class[i=j]))().max[1] for j in (1:n).+n];
mean(id_ .== class)


K = 3
model = ParsaModel(F = ParsimoniousNormal(p));
const_V = [diagm(ones(4))];
@|( model,
    class = Categorical(K),
    class[i=1:n] == class[i],
    Z = Categorical([1=>2,2=>2,3=>2]),
    iris_m[i=1:n] ~ F(:mu => [class[i], Z[class[i]][i]], :a => [class[i], Z[class[i]][i]], :L => [class[i], Z[class[i]][i]], :V => 1),
    :V[i=1]=const_V[i])
EM!(model; n_init=1, n_wild=1)

new_obs = Dict([(i+n) => Observation(x.X) for (i,x) in enumerate(iris_m)])
@| model new_obs[i=((1:n) .+ n)] ~ F(:mu => [class[i], Z[class[i]][i]], :a => [class[i], Z[class[i]][i]], :L => [class[i], Z[class[i]][i]], :V => 1)
id_ = [(@| model f(class[i=j]))().max[1] for j in (1:n).+n];
mean(id_ .== class)


K = 3
model = ParsaModel(F = Normal(p));
@|( model,
    class = Categorical(K),
    class[i=1:n] == class[i],
    Z = Categorical([1=>2,2=>2,3=>2]),
    cov = Categorical(2),
    iris_m[i=1:n] ~ F(:mu => [class[i], Z[class[i]][i]], :cov => cov[class[i], Z[class[i]][i]]))
EM!(model; n_init=1, n_wild=1)
G = Dict([j => (@| model f(cov[i=[j]]))() for j in reduce(vcat, [[[i,j] for i in 1:K] for j in 1:2])]);
for (key, M) in G
    mm = Dict(key => M.max[1])
    @| model cov[i=[key]] == mm[i]
end
new_obs = Dict([(i+n) => Observation(x.X) for (i,x) in enumerate(iris_m)])
@| model new_obs[i=((1:n) .+ n)] ~ F(:mu => [class[i], Z[class[i]][i]], :cov => cov[class[i], Z[class[i]][i]])
id_ = [(@| model f(class[i=j]))().max[1] for j in (1:n).+n];
mean(id_ .== class)

countmap([lv_v(v) for (cc, v) in model.class.LV])

K = 3
model = ParsaModel(F = ParsimoniousNormal(p));
@|( model,
    class = Categorical(K),
    class[i=1:n] == class[i],
    Z = Categorical([1=>2,2=>2,3=>2]),
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

