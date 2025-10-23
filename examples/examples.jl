# using ParsaModel

using CSV, DataFrames, Clustering, Distances, LinearAlgebra, StatsBase, ProgressBars, Distributions
include("../src/Types.jl")
include("../src/Core.jl")
# include("../src/Macros.jl")
include("../src/Models.jl")
include("../src/Notation.jl")

p = 20
K = 3
n = 100
true_id = rand(1:K, n);
mu = [ones(p), ones(p) .+ 6, ones(p) .- 6];
cov = [diagm(ones(p)), diagm(ones(p)), diagm(ones(p)) .+ 1];
X = Observation.([vec(rand(MvNormal(mu[true_id[i]], cov[true_id[i]]), 1)) for i in 1:n]);

N = MtvNormalSafe(p);
Z = categorical(6;name="Z");
for i in eachindex(X)
    X[i] ~ N(:mu => Z[i], :cov => Z[i])
end
EM!(N; verbose=true)
val(Z)
val(N[:cov])


n = 50
p = 5
K = 2
mu = [ones(p) .+ i for i in 1:n];
cov = [diagm(ones(p)) .+ i^4 for i in 1:2];
class_id = [Int(ceil(i / 5)) for i in 1:n];
n_classes = length(unique(class_id))
true_id = rand(1:2, n_classes);
X = Observation.([vec(rand(MvNormal(mu[class_id[i]], cov[true_id[class_id[i]]]), 1)) for i in 1:n]);

F = MtvNormal(p);
class = categorical(n_classes)
Z = categorical(K)
for i in eachindex(X)
    class[i] = class_id[i]
    X[i] ~ F(:mu => class[i], :cov => Z[class[i]])
end
EM!(F; n_init = 1, n_wild = 1)
i = length(X) + 1
n_new = Observation(zeros(p))
n_new ~ F(:mu => class[i], :cov => Z[class[i]])
@time f(class[i]);



F = MtvNormal(p);
class = categorical(n_classes)
Z = categorical(K)
for i in eachindex(X)[1:(end-1)]
    class[i] = class_id[i]
    X[i] ~ F(:mu => class[i], :cov => Z[class[i]])
end
i = length(X)
X[i] ~ F(:mu => class[i], :cov => Z[class[i]])
EM!(F; n_init = 1, n_wild = 1)
# for x in X prime_X(x) end
f(class[i])()

# xx = [length(va) for (d, va) in getDependentOnX(X)]

# maximum([length(d) for d in getDependentOnX(X)])

# indo_sets = getDependentOnX(X);
# x = X[1];
# conditional_dependent_search!(x, indo_sets, indo_sets[x], X);
# xx = [length(va) for (d, va) in getDependentOnX(X)]

# # length(GetDependentVariable(X[i-2])[2].dependent_X)

# GetDependentVariable(X[1])[2] == GetDependentVariable(X[i])[2]

# EM!(F; n_init = 1, n_wild = 1)


N = ParsimoniousNormal(p);
Z = categorical(3);
for i in eachindex(X)
    X[i] ~ N(:mu => Z[i], :a => Z[i], :L => Z[i], :V => Z[i])
end
EM!(N)
val(N[:V])[3] * val(N[:V])[3]'


display(Z[1].LV())

iris = CSV.read("./examples/datasets/Iris.csv", DataFrame)
iris_matrix = Matrix(iris[:, 2:5])
iris_m = Observation.(eachrow(iris_matrix));
n=size(iris_m)[1];
p=length(iris_m[1].X);
class_string = vec(iris[:,6]);
mapping = Dict(val => i for (i, val) in enumerate(unique(class_string)));
class = [mapping[val] for val in class_string];

K = 3
F = MtvNormal(p);
Z = categorical(K);
for i in eachindex(iris_m);
    iris_m[i] ~ F(:mu => Z[i], :cov => Z[i])
end
EM!(F; n_init=10, n_wild = 10)

n_params(F)
BIC(F)

iris_hclust = hclust(pairwise(Euclidean(), iris_matrix'), :ward)
init_id = cutree(iris_hclust, k=3)

K = 3
F = MtvNormal(p);
Z = categorical(K);
for i in eachindex(iris_m);
    iris_m[i] ~ F(:mu => Z[i], :cov => Z[i])
    Z[i] <-- init_id[i]
end
EM!(F)

id = [f(Z[i])().max[1] for i in 1:n];
randindex(id, class)




K = 3
F = ParsimoniousNormal(p);
Z = categorical(K);
for i in eachindex(iris_m);
    iris_m[i] ~ F(:mu => Z[i], :a => Z[i], :L => Z[i], :V => 1)
    Z[i] <-- init_id[i]
end
EM!(F)

id = [f(Z[i])().max[1] for i in 1:n];
randindex(id, class)

isa(Z[1], Function)

K = 3
F = ParsimoniousNormal(p);
Z = categorical(K);
for i in eachindex(iris_m);
    iris_m[i] ~ F(:mu => Z[i], :a => Z[i], :L => Z[i], :V => 1)
    Z[i] <-- init_id[i]
end
F[:V][1] = diagm(ones(p))
EM!(F)

val(F[:V])
val(F[:L])
val(Z)

id = [f(Z[i])().max[1] for i in 1:n];
randindex(id, class)


K = 3
F = ParsimoniousNormal(p);
Z = categorical(K);
for i in eachindex(iris_m);
    iris_m[i] ~ F(:mu => Z[i], :a => Z[i], :L => 1, :V => 1)
    Z[i] <-- init_id[i]
end
F[:L][1] = ones(p);
EM!(F)

val(F[:V])
val(F[:L])
val(F[:a])
val(Z)

id = [f(Z[i])().max[1] for i in 1:n];
randindex(id, class)



known_samples = sample(1:n, 30; replace=false)
known_map = Dict([s => class[s] for s in known_samples])
K = 3
F = MtvNormal(p);
Z = categorical(K)
for i in eachindex(iris_m);
    iris_m[i] ~ F(:mu => Z[i], :cov => Z[i])
end
for (ke, va) in known_map
    Z[ke] = va
end
EM!(F; n_init=10, n_wild = 10)

id = [f(Z[i])().max[1] for i in 1:n];
randindex(id, class)


K=3
blocks = Int.(repeat(1:(n/2),inner=2))
n_blocks = length(unique(blocks))
true_class_block = [class[i] for i in 1:n if i % 2 == 0]
F = MtvNormal(p);
Z = categorical(K);
B = categorical(n_blocks);
for i in 1:n;
    iris_m[i] ~ F(:mu => Z[B[i]], :cov => Z[B[i]])
    B[i] = blocks[i]
end
EM!(F)
val(F[:cov])

id = [f(Z[i])().max[1] for i in 1:n_blocks];
randindex(id, true_class_block)



K=3
blocks = [1;1:(n-1)]
II = [1;2; repeat([1], 148)]
n_blocks = length(unique(blocks))
perms = reduce(vcat, [[[i,j] for i in 1:K if i != j] for j in 1:K])
F = MtvNormal(p);
B = categorical(n_blocks);
P = categorical([i => 3 for i in 1:6]);
I = categorical(2);
PP = categorical(6);
for i in 1:n;
    iris_m[i] ~ F(:mu => P[PP[B[i]]][I[i]], :cov => P[PP[B[i]]][I[i]])
    B[i] = blocks[i]
    I[i] = II[i]
end
for i in 1:6
    for j in 1:2
        P[i][j] = perms[i][j]
    end
end
EM!(F)
perms[f(PP[1])().max[1]]

K = 3
F = MtvNormal(p);
cl = categorical(K);
for i in eachindex(iris_m);
    iris_m[i] ~ F(:mu => cl[i], :cov => cl[i])
    cl[i] = class[i]
end
EM!(F)

new_x = Observation(zeros(p));
new_x ~ F(:mu => cl[n+1], :cov => cl[n+1]);
pr = f(cl[n+1]);
post(x) = (new_x.X = x; pr().max[1])
class_pred = [post(x.X) for x in iris_m];
mean(class_pred .== class)


K = 3
F = MtvNormalDouble(p);
cl = categorical(K);
Z = categorical(3);
for i in eachindex(iris_m);
    iris_m[i] ~ F(:mu1 => cl[i], :mu2 => Z[i], :cov => Z[i])
    cl[i] = class[i]
end
EM!(F)

new_x = Observation(zeros(p));
new_x ~ F(:mu1 => cl[n+1], :mu2 => Z[n+1], :cov => Z[n+1]);
pr = f(cl[n+1]);
post(x) = (new_x.X = x; pr().max[1])
class_pred = [post(x.X) for x in iris_m];
mean(class_pred .== class)


K = 3
F = ParsimoniousNormal(p);
cl = categorical(K);
for i in eachindex(iris_m);
    iris_m[i] ~ F(:mu => cl[i], :a => cl[i], :L => 1, :V => 1)
    cl[i] = class[i]
end
EM!(F)

new_x = Observation(zeros(p));
new_x ~ F(:mu => cl[n+1], :a => cl[n+1], :L => 1, :V => 1);
pr = f(cl[n+1]);
post(x) = (new_x.X = x; pr().max[1])
class_pred = [post(x.X) for x in iris_m];
mean(class_pred .== class)


K = 3
F = MtvNormal(p);
cl = categorical(K);
Z = categorical([1=>2,2=>2,3=>2]);
for i in eachindex(iris_m);
    iris_m[i] ~ F(:mu => [cl[i], Z[cl[i]][i]], :cov => [cl[i], Z[cl[i]][i]])
    cl[i] = class[i]
end
EM!(F)

new_x = Observation(zeros(p));
i = n+1
new_x ~ F(:mu => [cl[i], Z[cl[i]][i]], :cov => [cl[i], Z[cl[i]][i]])
pr = f(cl[n+1]);
post(x) = (new_x.X = x; pr().max[1])
class_pred = [post(x.X) for x in iris_m];
mean(class_pred .== class)

length(getDependentOnX(new_x))


K = 3
F = ParsimoniousNormal(p);
cl = categorical(K);
Z = categorical([1=>2,2=>2,3=>2]);
for i in eachindex(iris_m);
    iris_m[i] ~ F(:mu => [cl[i], Z[cl[i]][i]], :a => [cl[i], Z[cl[i]][i]], :L => [cl[i], Z[cl[i]][i]], :V => 1)
    cl[i] = class[i]
end
F[:V][1] = diagm(ones(p));
EM!(F)


new_x = Observation(zeros(p));
i = n+1
new_x ~ F(:mu => [cl[i], Z[cl[i]][i]], :a => [cl[i], Z[cl[i]][i]], :L => [cl[i], Z[cl[i]][i]], :V => 1)
pr = f(cl[n+1]);
post(x) = (new_x.X = x; pr().max[1])
class_pred = [post(x.X) for x in iris_m];
mean(class_pred .== class)


K = 3
F = MtvNormal(p);
cl = categorical(K; name="cl");
Z = categorical([1=>2,2=>2,3=>2];name="Z");
cov = categorical(2; name="cov");
for i in eachindex(iris_m);
    iris_m[i] ~ F(:mu => [cl[i], Z[cl[i]][i]], :cov => cov[cl[i], Z[cl[i]][i]])
    cl[i] = class[i]
end
EM!(F)
val(F[:cov])
val(cov)
val(Z)


new_x = Observation(zeros(p));
i = n+1
new_x ~ F(:mu => [cl[i], Z[cl[i]][i]], :cov => cov[cl[i], Z[cl[i]][i]])
pr = f(cl[n+1]);
post(x) = (new_x.X = x; pr().max[1])
class_pred = [post(x.X) for x in iris_m];
mean(class_pred .== class)



K = 3
F = ParsimoniousNormal(p);
cl = categorical(K);
Z = categorical([1=>2,2=>2,3=>2]);
cov = categorical(2);
for i in eachindex(iris_m);
    iris_m[i] ~ F(:mu => [cl[i], Z[cl[i]][i]], :a => cov[cl[i], Z[cl[i]][i]], :L => cov[cl[i], Z[cl[i]][i]], :V => 1)
    cl[i] = class[i]
end
EM!(F)
val(F[:L])
val(F[:V])

new_x = Observation(zeros(p));
i = n+1
new_x ~  F(:mu => [cl[i], Z[cl[i]][i]], :a => cov[cl[i], Z[cl[i]][i]], :L => cov[cl[i], Z[cl[i]][i]], :V => 1)
pr = f(cl[n+1]);
post(x) = (new_x.X = x; pr().max[1])
class_pred = [post(x.X) for x in iris_m];
mean(class_pred .== class)