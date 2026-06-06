using Test
using LinearAlgebra, Distributions, ParsaModel
# import ParsaModel: ~
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
		class[i] = class_id[i]
		X[i] ~ F(:mu => class[i], :cov => Z[class[i]])
	end
	EM!(F; n_init = 30, init_eps=10^-6)

	F = ParsimoniousNormal(p);
	class = categorical(n_classes)
	Z = categorical(K)
	for i in eachindex(X)
		class[i] = class_id[i]
		X[i] ~ F(:mu => class[i], :a => Z[class[i]], :L => Z[class[i]], :V => 1)
	end
	EM!(F; n_init = 10, init_eps=10^-3)

	iris_matrix = [5.1 3.5 1.4 0.2; 4.9 3.0 1.4 0.2; 4.7 3.2 1.3 0.2; 4.6 3.1 1.5 0.2; 5.0 3.6 1.4 0.2; 5.4 3.9 1.7 0.4; 4.6 3.4 1.4 0.3; 5.0 3.4 1.5 0.2; 4.4 2.9 1.4 0.2; 4.9 3.1 1.5 0.1; 5.4 3.7 1.5 0.2; 4.8 3.4 1.6 0.2; 4.8 3.0 1.4 0.1; 4.3 3.0 1.1 0.1; 5.8 4.0 1.2 0.2; 5.7 4.4 1.5 0.4; 5.4 3.9 1.3 0.4; 5.1 3.5 1.4 0.3; 5.7 3.8 1.7 0.3; 5.1 3.8 1.5 0.3; 5.4 3.4 1.7 0.2; 5.1 3.7 1.5 0.4; 4.6 3.6 1.0 0.2; 5.1 3.3 1.7 0.5; 4.8 3.4 1.9 0.2; 5.0 3.0 1.6 0.2; 5.0 3.4 1.6 0.4; 5.2 3.5 1.5 0.2; 5.2 3.4 1.4 0.2; 4.7 3.2 1.6 0.2; 4.8 3.1 1.6 0.2; 5.4 3.4 1.5 0.4; 5.2 4.1 1.5 0.1; 5.5 4.2 1.4 0.2; 4.9 3.1 1.5 0.1; 5.0 3.2 1.2 0.2; 5.5 3.5 1.3 0.2; 4.9 3.1 1.5 0.1; 4.4 3.0 1.3 0.2; 5.1 3.4 1.5 0.2; 5.0 3.5 1.3 0.3; 4.5 2.3 1.3 0.3; 4.4 3.2 1.3 0.2; 5.0 3.5 1.6 0.6; 5.1 3.8 1.9 0.4; 4.8 3.0 1.4 0.3; 5.1 3.8 1.6 0.2; 4.6 3.2 1.4 0.2; 5.3 3.7 1.5 0.2; 5.0 3.3 1.4 0.2; 7.0 3.2 4.7 1.4; 6.4 3.2 4.5 1.5; 6.9 3.1 4.9 1.5; 5.5 2.3 4.0 1.3; 6.5 2.8 4.6 1.5; 5.7 2.8 4.5 1.3; 6.3 3.3 4.7 1.6; 4.9 2.4 3.3 1.0; 6.6 2.9 4.6 1.3; 5.2 2.7 3.9 1.4; 5.0 2.0 3.5 1.0; 5.9 3.0 4.2 1.5; 6.0 2.2 4.0 1.0; 6.1 2.9 4.7 1.4; 5.6 2.9 3.6 1.3; 6.7 3.1 4.4 1.4; 5.6 3.0 4.5 1.5; 5.8 2.7 4.1 1.0; 6.2 2.2 4.5 1.5; 5.6 2.5 3.9 1.1; 5.9 3.2 4.8 1.8; 6.1 2.8 4.0 1.3; 6.3 2.5 4.9 1.5; 6.1 2.8 4.7 1.2; 6.4 2.9 4.3 1.3; 6.6 3.0 4.4 1.4; 6.8 2.8 4.8 1.4; 6.7 3.0 5.0 1.7; 6.0 2.9 4.5 1.5; 5.7 2.6 3.5 1.0; 5.5 2.4 3.8 1.1; 5.5 2.4 3.7 1.0; 5.8 2.7 3.9 1.2; 6.0 2.7 5.1 1.6; 5.4 3.0 4.5 1.5; 6.0 3.4 4.5 1.6; 6.7 3.1 4.7 1.5; 6.3 2.3 4.4 1.3; 5.6 3.0 4.1 1.3; 5.5 2.5 4.0 1.3; 5.5 2.6 4.4 1.2; 6.1 3.0 4.6 1.4; 5.8 2.6 4.0 1.2; 5.0 2.3 3.3 1.0; 5.6 2.7 4.2 1.3; 5.7 3.0 4.2 1.2; 5.7 2.9 4.2 1.3; 6.2 2.9 4.3 1.3; 5.1 2.5 3.0 1.1; 5.7 2.8 4.1 1.3; 6.3 3.3 6.0 2.5; 5.8 2.7 5.1 1.9; 7.1 3.0 5.9 2.1; 6.3 2.9 5.6 1.8; 6.5 3.0 5.8 2.2; 7.6 3.0 6.6 2.1; 4.9 2.5 4.5 1.7; 7.3 2.9 6.3 1.8; 6.7 2.5 5.8 1.8; 7.2 3.6 6.1 2.5; 6.5 3.2 5.1 2.0; 6.4 2.7 5.3 1.9; 6.8 3.0 5.5 2.1; 5.7 2.5 5.0 2.0; 5.8 2.8 5.1 2.4; 6.4 3.2 5.3 2.3; 6.5 3.0 5.5 1.8; 7.7 3.8 6.7 2.2; 7.7 2.6 6.9 2.3; 6.0 2.2 5.0 1.5; 6.9 3.2 5.7 2.3; 5.6 2.8 4.9 2.0; 7.7 2.8 6.7 2.0; 6.3 2.7 4.9 1.8; 6.7 3.3 5.7 2.1; 7.2 3.2 6.0 1.8; 6.2 2.8 4.8 1.8; 6.1 3.0 4.9 1.8; 6.4 2.8 5.6 2.1; 7.2 3.0 5.8 1.6; 7.4 2.8 6.1 1.9; 7.9 3.8 6.4 2.0; 6.4 2.8 5.6 2.2; 6.3 2.8 5.1 1.5; 6.1 2.6 5.6 1.4; 7.7 3.0 6.1 2.3; 6.3 3.4 5.6 2.4; 6.4 3.1 5.5 1.8; 6.0 3.0 4.8 1.8; 6.9 3.1 5.4 2.1; 6.7 3.1 5.6 2.4; 6.9 3.1 5.1 2.3; 5.8 2.7 5.1 1.9; 6.8 3.2 5.9 2.3; 6.7 3.3 5.7 2.5; 6.7 3.0 5.2 2.3; 6.3 2.5 5.0 1.9; 6.5 3.0 5.2 2.0; 6.2 3.4 5.4 2.3; 5.9 3.0 5.1 1.8];
	iris_m = Observation.(eachrow(iris_matrix));
	class = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3];
	n = length(class)
	p = size(iris_matrix)[2]
	K = 3
	F = MtvNormal(p);
	Z = categorical(K);
	for i in eachindex(iris_m);
		iris_m[i] ~ F(:mu => Z[i], :cov => Z[i])
	end
	EM!(F; n_init=10, init_eps=10^-3)

	init_id = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 2]

	K = 3
	F = MtvNormal(p);
	Z = categorical(K);
	for i in eachindex(iris_m);
		iris_m[i] ~ F(:mu => Z[i], :cov => Z[i])
		Z[i] <-- init_id[i]
	end
	EM!(F)

	K = 3
	F = ParsimoniousNormal(p);
	Z = categorical(K);
	for i in eachindex(iris_m);
		iris_m[i] ~ F(:mu => Z[i], :a => Z[i], :L => Z[i], :V => 1)
		Z[i] <-- init_id[i]
	end
	F[:V][1] = diagm(ones(p))
	EM!(F)


	known_samples = sample(1:n, 20; replace=false)
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
	EM!(F; n_init=1, init_eps = 10^-5, verbose=true)

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


	K = 3
	F = MtvNormal(p);
	cl = categorical(K);
	for i in eachindex(iris_m);
		iris_m[i] ~ F(:mu => cl[i], :cov => 1)
		cl[i] = class[i]
	end
	EM!(F;verbose=true, allow_desc_likelihood=true)

	n = length(iris_m)
	new_x = Observation();
	new_x ~ F(:mu => cl[n+1], :cov => 1);
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

	new_x = Observation();
	new_x ~ F(:mu => cl[n+1], :a => cl[n+1], :L => 1, :V => 1);
	pr = f(cl[n+1]);
	post(x) = (new_x.X = x; pr().max[1])
	class_pred = [post(x.X) for x in iris_m];
	mean(class_pred .== class)


	K = 3
	F = ParsimoniousNormal(p);
	cl = categorical(K);
	Z = categorical([1=>2,2=>2,3=>2]);
	for i in eachindex(iris_m);
		iris_m[i] ~ F(:mu => [cl[i], Z[cl[i]][i]], :a => [cl[i], Z[cl[i]][i]], :L => [cl[i], Z[cl[i]][i]], :V => 1)
		cl[i] = class[i]
	end
	F[:V][1] = diagm(ones(p));
	EM!(F; n_init=3, init_eps=10^-5)

	new_x = Observation();
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
	EM!(F; n_init=5, eps=10^-6, init_eps=10^-2, verbose=true)

	new_x = Observation();
	i = n+1
	new_x ~ F(:mu => [cl[i], Z[cl[i]][i]], :cov => cov[cl[i], Z[cl[i]][i]])
	pr = f(cl[n+1]);
	post(x) = (new_x(x); pr().max[1])
	class_pred = [post(x.X) for x in iris_m];
	mean(class_pred .== class)




end

