using ParsaModel

using CSV, DataFrames, Clustering, Distances, LinearAlgebra, StatsBase


iris = CSV.read("./examples/datasets/Iris.csv", DataFrame)
iris_matrix = Matrix(iris[:, 2:5])
iris_m = eachrow(iris_matrix);
n=size(iris_m)[1];
p=length(iris_m[1]);
class_string = vec(iris[:,6]);
mapping = Dict(val => i for (i, val) in enumerate(unique(class_string)));
class = [mapping[val] for val in class_string];

K = 3
model = Parsa_Model(Normal_Model(p));
@Categorical(model, Z, K);
@Observation(model, X[i] = iris_m[i] -> (:mu => Z[i], :cov => Z[i]), i = 1:n)
EM!(model; n_init=100, n_wild=30)

@Parameter(model, :mu)
@Parameter(model, :cov)

id = @posterior_probability(model, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
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

K = 3
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, Z, K);
@Observation(model, X[i] = iris_m[i] = (:mu => Z[i], :a => Z[i], :L => Z[i], :V => 1), i = 1:n)
EM!(model; n_init=20, n_wild=30)
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



blocks = [1;1:(n-1)]
I = [1;2; repeat([1], 148)]
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




