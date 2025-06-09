using ParsaModel
using Test, LinearAlgebra, Distributions

@testset "ParsaModel.jl" begin

	n = 200
	p = 5
	K = 2
	mu = [ones(p) .+ i for i in 1:n];
	cov = [diagm(ones(p)) .+ i^4 for i in 1:2];
	class_id = [Int(ceil(i / 5)) for i in 1:n];
	n_classes = length(unique(class_id))
	true_id = rand(1:2, n_classes);
	X = Observation.([vec(rand(MvNormal(mu[class_id[i]], cov[true_id[class_id[i]]]), 1)) for i in 1:n]);

	model = Parsa_Model(F = Normal_Model(p));
	@|(model,
		class = Categorical(n_classes),
		class[i=1:n] == class_id[i],
		Z = Categorical(K),
		X[i=1:n] ~ F(:mu => class[i], :cov => Z[class[i]]));
	EM!(model; n_init = 10, n_wild = 10)
	@| model Z :mu :cov

	model = Parsa_Model(F = Normal_Parsa_Model(p));
	@|(model,
		class = Categorical(n_classes),
		class[i=1:n] == class_id[i],
		Z = Categorical(K),
		X[i=1:n] ~ F(:mu => class[i], :a => Z[class[i]], :L => Z[class[i]], :V => 1))
	EM!(model; n_init = 10, n_wild = 10)

end
