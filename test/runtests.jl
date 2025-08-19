using Test
using LinearAlgebra, Distributions, ParsaModel
import ParsaModel: ~
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

	F = MtvNormal(p);
	class = categorical(n_classes)
	Z = categorical(K)
	for i in eachindex(X)
		class[i] <| class_id[i]
		X[i] ~ F(:mu => class[i], :cov => Z[class[i]])
	end
	EM!(F; n_init = 10, n_wild = 10)

	F = ParsimoniousNormal(p);
	class = categorical(n_classes)
	Z = categorical(K)
	for i in eachindex(X)
		class[i] <| class_id[i]
		X[i] ~ F(:mu => class[i], :a => Z[class[i]], :L => Z[class[i]], :V => 1)
	end
	EM!(F; n_init = 10, n_wild = 10)

end
