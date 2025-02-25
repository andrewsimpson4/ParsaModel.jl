using ParsaModel
using Test, LinearAlgebra, Distributions

@testset "ParsaModel.jl" begin
    n = 200
    p = 5
    K = 2
    mu = [ones(p) .+ i  for i in 1:n];
    cov = [diagm(ones(p)) .+ i^4 for i in 1:2];
    class_id = [Int(ceil(i / 5)) for i in 1:n];
    n_classes = length(unique(class_id))
    true_id = rand(1:2, n_classes);
    X = [vec(rand(MvNormal(mu[class_id[i]], cov[true_id[class_id[i]]]), 1)) for i in 1:n];

    model_test = Parsa_Model(Normal_Model(p));
    @Categorical(model_test, class, n_classes)
    @Known(model_test, class[i] = class_id[i], i=1:n)
    @Categorical(model_test, Z, K)
    @Observation(model_test, Y[i] = X[i] = (:mu => class[i], :cov => Z[class[i]]), i = 1:n)
    EM!(model_test; n_init=1, n_wild=1)
end