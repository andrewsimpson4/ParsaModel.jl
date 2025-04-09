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
model_test = Parsa_Model(Normal_Model(p));
@Categorical(model_test, Z, K);
@Observation(model_test, X[i] = iris_m[i] = (:mu => Z[i], :cov => Z[i]), i = 1:n)
EM!(model_test; n_init=100, n_wild=30)
id = @posterior_probability(model_test, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)


iris_hclust = hclust(pairwise(Euclidean(), iris_matrix'), :ward)
init_id = cutree(iris_hclust, k=3)

K = 3
model_test = Parsa_Model(Normal_Model(p));
@Categorical(model_test, Z, K);
@Initialize(model_test, Z[i] = init_id[i], i = 1:n)
@Observation(model_test, X[i] = iris_m[i] = (:mu => Z[i], :cov => Z[i]), i = 1:n)
EM!(model_test; should_initialize=false)
id = @posterior_probability(model_test, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)



K = 3
model_test = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model_test, Z, K);
@Observation(model_test, X[i] = iris_m[i] = (:mu => Z[i], :a => Z[i], :L => Z[i], :V => 1), i = 1:n)
EM!(model_test; n_init=20, n_wild=30)
id = @posterior_probability(model_test, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)

K = 3
model_test = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model_test, Z, K);
@Observation(model_test, X[i] = iris_m[i] = (:mu => Z[i], :a => Z[i], :L => Z[i], :V => 1), i = 1:n)
const_V = [diagm(ones(4))];
@Constant(model_test, :V[i] = const_V[i], i = 1)
EM!(model_test; n_init=20, n_wild=30)
id = @posterior_probability(model_test, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)

K = 3
model_test = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model_test, Z, K);
@Observation(model_test, X[i] = iris_m[i] = (:mu => Z[i], :a => Z[i], :L => 1, :V => 1), i = 1:n)
const_V = [diagm(ones(4))];
@Constant(model_test, :V[i] = const_V[i], i = 1)
EM!(model_test; n_init=20, n_wild=30)
id = @posterior_probability(model_test, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)



K = 3
model_test = Parsa_Model(Normal_Model(p));
@Categorical(model_test, class, K);
@Known(model_test, class[i] = class[i], i = 1:n)
@Observation(model_test, X[i] = iris_m[i] = (:mu => class[i], :cov => class[i]), i = 1:n)
EM!(model_test; n_init=1, n_wild=1)
@Observation(model_test, X_new[i] = iris_m[i] = (:mu => class[i, "T"], :cov => class[i, "T"]), i = 1:n)
id = @posterior_probability(model_test, [class[i, "T"]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)


K = 3
model_test = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model_test, class, K);
@Known(model_test, class[i] = class[i], i = 1:n)
@Observation(model_test, X[i] = iris_m[i] = (:mu => class[i], :a => class[i], :L => 1, :V => 1), i = 1:n)
const_V = [diagm(ones(4))];
@Constant(model_test, :V[i] = const_V[i], i = 1)
EM!(model_test; n_init=1, n_wild=1)
@Observation(model_test, X_new[i] = iris_m[i] = (:mu => class[i, "T"], :a => class[i, "T"], :L => 1, :V => 1), i = 1:n)
id = @posterior_probability(model_test, [class[i, "T"]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)


K = 3
model_test = Parsa_Model(Normal_Model(p));
@Categorical(model_test, class, K);
@Known(model_test, class[i] = class[i], i = 1:n)
@Categorical_Set(model_test, Z, [2,2,2], 1:3);
@Observation(model_test, X[i] = iris_m[i] = (:mu => [class[i], Z[class[i]][i]], :cov => [class[i], Z[class[i]][i]]), i = 1:n)
EM!(model_test; n_init=10, n_wild=10)
@Observation(model_test, X_new[i] = iris_m[i] = (:mu => [class[i, "T"], Z[class[i, "T"]][i, "T"]], :cov => [class[i, "T"], Z[class[i, "T"]][i, "T"]]), i = 1:n)
id = @posterior_probability(model_test, [class[i, "T"]], i = 1:n)();
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)


K = 3
model_test = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model_test, class, K);
@Known(model_test, class[i] = class[i], i = 1:n)
@Categorical_Set(model_test, Z, [2,2,2], 1:3);
@Observation(model_test, X[i] = iris_m[i] = (:mu => [class[i], Z[class[i]][i]], :a => [class[i], Z[class[i]][i]], :L => [class[i], Z[class[i]][i]], :V => 1), i = 1:n)
EM!(model_test; n_init=10, n_wild=10)
@Observation(model_test, X_new[i] = iris_m[i] = (:mu => [class[i, "T"], Z[class[i, "T"]][i, "T"]], :a => [class[i, "T"], Z[class[i, "T"]][i, "T"]], :L => [class[i, "T"], Z[class[i, "T"]][i, "T"]], :V => 1), i = 1:n)
id = @posterior_probability(model_test, [class[i, "T"]], i = 1:n)();
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)


K = 3
model_test = Parsa_Model(Normal_Model(p));
@Categorical(model_test, class, K);
@Known(model_test, class[i] = class[i], i = 1:n)
@Categorical_Set(model_test, Z, [2,2,2], 1:3);
@Categorical(model_test, cov, 2)
@Observation(model_test, X[i] = iris_m[i] = (:mu => [class[i], Z[class[i]][i]], :cov => cov[class[i], Z[class[i]][i]]), i = 1:n)
EM!(model_test; n_init=10, n_wild=10)
G = @posterior_probability(model_test, [cov[i]], i = reduce(vcat, [[[i,j] for i in 1:K] for j in 1:2]))()
for (key, M) in G
    mm = Dict(key => M.max)
    @Known(model_test, cov[i] = mm[i], i = [key])
end
@Observation(model_test, X_new[i] = iris_m[i] = (:mu => [class[i, "T"], Z[class[i, "T"]][i, "T"]], :cov => cov[class[i, "T"], Z[class[i, "T"]][i, "T"]]), i = 1:n)
id = @posterior_probability(model_test, [class[i, "T"]], i = 1:n)();
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)

model_test.cov.LV

K = 3
model_test = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model_test, class, K);
@Known(model_test, class[i] = class[i], i = 1:n)
@Categorical_Set(model_test, Z, [2,2,2], 1:3);
@Categorical(model_test, cov, 2)
@Observation(model_test, X[i] = iris_m[i] = (:mu => [class[i], Z[class[i]][i]], :a => cov[class[i], Z[class[i]][i]], :L => cov[class[i], Z[class[i]][i]], :V => 1), i = 1:n)
EM!(model_test; n_init=10, n_wild=10)
@Parameter(model_test, :mu)
for (_, L) in @Parameter(model_test, :cov)
    display(L'*L)
end
@Observation(model_test, X_new[i] = iris_m[i] = (:mu => [class[i, "T"], Z[class[i, "T"]][i, "T"]], :a => cov[class[i, "T"], Z[class[i, "T"]][i, "T"]], :L => cov[class[i, "T"], Z[class[i, "T"]][i, "T"]], :V => 1), i = 1:n)
id = @posterior_probability(model_test, [class[i, "T"]], i = 1:n)();
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)
