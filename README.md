<!-- # ParsaModel -->

<!-- [![Build Status](https://github.com/andrewsimpson4/ParsaModel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/andrewsimpson4/ParsaModel.jl/actions/workflows/CI.yml?query=branch%3Amain) -->
<p align="center">
<img src="./Assets/logo.png" alt="drawing" width="200"/>

ParaModel is a Julia package for creating, estimating and predicting using Parsa Models. A Parsa Model is a generic framework for models of the form
```math
X_i | Z = \gamma \sim F(T^i_{1}(\gamma), \dots, T^i_{G}(\gamma); \Psi)
```
where $Z_{mj} \sim \text{Categorical}(\pi_{m1}, \pi_{m2}, \dots, \pi_{mK_m})$. See the [paper](https://apple.com) for more details on Parsa Models.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Tutorial](#usage-examples)
- [Package Reference](#api-reference)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## ✨ Features

- **Develop**: define custom and novel parsa models
- **Fit models**: maximum likelihood estimation is used to fit models
- **Clustering, Predictions, and Likelihood**: Given your problem, the fitted model can be used to cluster observations, predict new observations, or get likelihoods

## 🚀 Installation

### Julia
```bash
] add ParsaModel
```

## 🏁 Quick Start

This is a minimal example of how to define and fit a $p$-dimensional Gaussian mixture model with $K$ components where observations are stored in the variable $X$.

```julia
model = Parsa_Model(Normal_Model(p));
@Categorical(model, Z, K);
@Observation(model, X[i] = X[i] -> (:mu => Z[i], :cov => Z[i]), i = 1:n)
EM!(model)
```

## 💡 Usage Tutorial

For the examples listed below, 15 different Parsa Models are defined and fit all on the iris dataset. It should be noted most of this models are not actually good models for the iris dataset but are none the less possible to fit for the sake of simplicity.

### Setup the iris dataset
First the iris dataset is processed into the correct format for the package.
```julia
iris = CSV.read("./examples/datasets/Iris.csv", DataFrame)
iris_matrix = Matrix(iris[:, 2:5])
iris_m = eachrow(iris_matrix);
n=size(iris_m)[1];
p=length(iris_m[1]);
class_string = vec(iris[:,6]);
mapping = Dict(val => i for (i, val) in enumerate(unique(class_string)));
class = [mapping[val] for val in class_string];
```
Here we have a vector of vectors `iris_m` where each element is one of the observations from the dataset. Next is `class` which is a vector containing the species of the respective elements in `iris_m`. We also define `n`, the number of observations, as well as `p` which is the dimension of each observation.

### Gaussian Mixture Model

The first example is how to implement a Gaussian mixture model using ParsaModel. In particular we are looking at the model $X_i | Z_i = k \sim N(\mu_k, \Sigma_k)$. This package is manly interacted with via macros which allows for a custom and minimal syntax. Here we are using a finite mixture model to cluster the observations in the iris dataset with the goal of clustering and recovering species. Thus we will look look for $3$ clusters.

```julia
K = 3
model = Parsa_Model(Normal_Model(p));
@Categorical(model, Z, K);
@Observation(model, X[i] = iris_m[i] -> (:mu => Z[i], :cov => Z[i]), i = 1:n)
EM!(model; n_init=100, n_wild=30)
```
The following is a description of what each function above is doing.
- `Normal_Model(p)` defines the base distributional assumption we are making for the data. In this case, a $p$-dimensional multivariate normal distribution.
- `Parsa_Model` returns an isolated "space" where we will build the rest of the model.
- `@Categorical(model, Z, K)` creates a new categorical distribution named `Z` with `K` categories inside of our space `model`.
- `@Observation(model, X[i] = iris_m[i] -> (:mu => Z[i], :cov => Z[i]), i = 1:n)` loops through `i` from $1$ to `n` and assigns `iris_m[i]` to `X[i]` which is a variable now defined locally inside of `model`. Finally `(:mu => Z[i], :cov => Z[i])` defines the mapping of the observation. The parameters `:mu` and `:cov` are exposed by `Normal_Model` and different base models with have different associated parameters. `Z[i]` respresents a random varaible sampled from `Z` which can take on values from $1$ to `K`.
- `EM!(model; n_init=100, n_wild=30)` simply fits the model! `n_init` is the number of initializations to run and `n_wild` is the number of steps per initializations run.

⚠️ This package currently uses random initialization by default. This can have mixed results for finding the maximum likelihood estimates but allows for package to fit ANY model which can be defined using the package. Just proceed with caution and watch the likelihood plot output for incite.

After running `EM!(model; n_init=100, n_wild=30)` you should see sometime like the following with your terminal.

Note: A alternate mathematical notation for the model that may better align with the syntax of the package may be $X_i | Z[i] \sim N(\mu(Z[i]), \Sigma(Z[i]))$.

![](./Assets/ex_vid.gif)

The purple lines here are each of the initialization runs and the final green line is taking the best initialization and running the algorithm until convergence is reached.

Now that the model has been fit, we can look at the parameter estimates of the model. Since we used `Normal_Model` as the base, we have parameters for `:mu` and `:cov`. This can be viewed with the following.

```julia
@Parameter(model, :mu)
@Parameter(model, :cov)
@Parameter(model, Z)
```

Notice here that we have $3$ `:mu` parameters and $3$ `:cov` paramters since we fit a $3$ component mixture model. We also can see the estimated proportions of the categorical distribution `Z`.

Since our goal was to use a gaussian mixture model to cluster observations from the iris dataset, we need to get the max posterior probability for each `Z[i]`. This can be done by the following.

```julia
id = @posterior_probability(model, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)
```
- `@posterior_probability(model, [Z[i]], i = 1:n)()` returns the $P(Z[i] = k | X)$ for $k=1,2,\dots, K$ since our categorical distribution `Z` has `K` categories.
- `id[i].max` simply gets the max of $P(Z[i] = k | X)$ for $k=1,2,\dots, K$
- `randindex` is from the `Clustering` package and gives values such as the adjusted rand index to see how well the clustering solution compared to the ground truth.

#### Custom Initialization

While the default initialization method is extremely flexible and works with any model which can be defined with this package, it may take a lot of initialization runs to achieve the desired performance. In cases when a custom initialization method can be defined for a given model structure, initial values can be passed directly into the model.

For the iris dataset with a Gaussian mixture model we will used hierarchical clustering found in the `Clustering` package to get initial ID's for each `Z[i]`.

```julia
iris_hclust = hclust(pairwise(Euclidean(), iris_matrix'), :ward)
init_id = cutree(iris_hclust, k=3)
```
We can now build and run the model which is the same as above with a few small changes.

```julia
K = 3
model = Parsa_Model(Normal_Model(p));
@Categorical(model, Z, K);
@Initialize(model, Z[i] = init_id[i], i = 1:n);
@Observation(model, X[i] = iris_m[i] -> (:mu => Z[i], :cov => Z[i]), i = 1:n);
EM!(model; should_initialize=false);
id = @posterior_probability(model, [Z[i]], i = 1:n)();
id_ = [id[i].max for i in 1:n];
randindex(id_, class)
```
-`@Initialize(model, Z[i] = init_id[i], i = 1:n)` takes our initial values from init_id and assigns them to the respective random variable `Z[i]`.
-`should_initialize=false` disables the default initialization method.

You should see and output the following which shows very good clustering performance

![](./Assets/ex_vid2.gif)
```
(0.9038742317748124, 0.9574944071588367, 0.042505592841163314, 0.9149888143176734)
```
Notice here there are no purple lines since we did pre-initialize the algorithm.


### Parsimonious Gaussian Mixture Models

A common method in finite mixture models is to consider the parsimonious parameterization of the multivariate normal distribution. In particular we are considering the parameterization $N(\mu, a V\Lambda V')$. This is implemented in `R` using the very well known package `mclust`. We can do a similar thing in this package but because of the generalization of the method, we have the ability to implement far more models than existing packages. In the previous parameterization using the `Normal_Model` function we had access to two parameters. With this parameterization we have four.

#### Shared eigenvectors between components
Suppose we wish to fit a finite mixture model like before except where every component has the same eigenvector structure. This is implemented with the following.
```julia
K = 3
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, Z, K);
@Observation(model, X[i] = iris_m[i] -> (:mu => Z[i],
                                         :a => Z[i],
                                         :L => Z[i],
                                         :V => 1), i = 1:n)
EM!(model; n_init=20, n_wild=30)
id = @posterior_probability(model, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)
```
- `Normal_Parsa_Model` gives the Parsimonious Gaussian parameterization as the base distributional assumption.
- `:V => 1` enforces that every component from the mixture has the same $V_1$ parameter.

#### Diagonal covariance matrices
Suppose we wish to enforce that each covariance matrix is diagonal in the mixture.

```julia
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, Z, K);
@Observation(model, X[i] = iris_m[i] -> (:mu => Z[i],
                                         :a => Z[i],
                                         :L => Z[i],
                                         :V => 1), i = 1:n)
const_V = [diagm(ones(4))];
@Constant(model, :V[i] = const_V[i], i = 1)
EM!(model; n_init=20, n_wild=30)
id = @posterior_probability(model, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)
```
-`const_V = [diagm(ones(4))];` simply defined a diagonal matrix.
-`@Constant(model, :V[i] = const_V[i], i = 1)` sets $V_1$ as a constant and to be the diagonal matrix.

#### Diagonal covariance matrices with shared eigenvalues

We can extend this further to a model where we assume diagonal covariance matrices and assume that each covariance matrix has the same eigenvalues.

```julia
K = 3
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, Z, K);
@Observation(model, X[i] = iris_m[i] -> (:mu => Z[i],
                                        :a => Z[i],
                                        :L => 1,
                                        :V => 1), i = 1:n)
const_V = [diagm(ones(4))];
@Constant(model, :V[i] = const_V[i], i = 1)
EM!(model; n_init=20, n_wild=30)
id = @posterior_probability(model, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)
```
One should now start to see a pattern of how simply changing the mapping for each observation leads to different models.

### Discriminant Analysis

So far we have considered using Parsa Models for clustering, but we can also use then to build classifiers. We will again consider the iris dataset but now assume we know the classes (species) and wish to build a model to predict the species of new observations.

#### Linear Discriminant Analysis

To implement LDA using this package, it will look similar to a mixture model but now we instead know the true labels.

```julia
K = 3
model = Parsa_Model(Normal_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Observation(model, X[i] = iris_m[i] -> (:mu => class[i], :cov => 1), i = 1:n)
EM!(model; n_init=1, n_wild=1)
```
-`@Known(model, class[i] = class[i], i = 1:n)` simply assigns the value of `class[i]` to the respective random variable `class[i]` within our space `model` and changes it to a known variable. Thus `class[i]` can no longer take the values $1,2,\dots, K$ and instead is always the value of `class[i]`.

Note here that there are no unknown categorical variables, thus EM algorithm is constant and we don't see and increase in the likelihood like before.

What if we wish to predict new observations? This can be done the following way

```julia
@Observation(model, X_new[i] = iris_m[i] -> (:mu => class[i, "T"], :cov => 1), i = 1:n)
id = @posterior_probability(model, [class[i, "T"]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)
```
- `class[i, "T"]` ensures that our new observation does not overlap i.e. share a random variable with observation defined for model training. If we have `class[i]`, that multiple observations would share the random variable `class[i]` which is not desired in this case.

#### Quadratic Discriminant Analysis

For QDA we simply have

```julia
K = 3
model = Parsa_Model(Normal_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Observation(model, X[i] = iris_m[i] -> (:mu => class[i], :cov => class[i]), i = 1:n)
EM!(model; n_init=1, n_wild=1)
@Observation(model, X_new[i] = iris_m[i] -> (:mu => class[i, "T"], :cov => class[i, "T"]), i = 1:n)
id = @posterior_probability(model, [class[i, "T"]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)
```
#### Common component desciminant analysis

Suppose we wish to do a model like LDA and QDA but where each component shares the same covariance matrix. This can be done as follows.

```julia
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Observation(model, X[i] = iris_m[i] -> (:mu => class[i],
                                         :a => class[i],
                                         :L => class[i],
                                         :V => 1), i = 1:n)
EM!(model; n_init=1, n_wild=1)
@Observation(model, X_new[i] = iris_m[i] = (:mu => class[i, "T"],
                                            :a => class[i, "T"],
                                            :L => class[i, "T"],
                                            :V => 1), i = 1:n)
id = @posterior_probability(model, [class[i, "T"]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)
```

### Mixture desciminant analysis

Suppose we wish to assume that each class follows Gaussian mixture model. This can be done by

```julia
K = 3
model = Parsa_Model(Normal_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Categorical(model, Z, [2,2,2]);
@Observation(model, X[i] = iris_m[i] -> (:mu => [class[i], Z[class[i]][i]],
                                        :cov => [class[i], Z[class[i]][i]]), i = 1:n)
EM!(model; n_init=10, n_wild=10)
@Parameter(model, :mu)

@Observation(model, X_new[i] = iris_m[i] -> (:mu => [class[i, "T"], Z[class[i, "T"]][i, "T"]],
                                            :cov => [class[i, "T"], Z[class[i, "T"]][i, "T"]]), i = 1:n)
id = @posterior_probability(model, [class[i, "T"]], i = 1:n)();
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)
```
-`@Categorical(model, Z, [2,2,2])` creates a set of three categorical distributions each which has two categories. This is to set up a different categorical distribution for each class.
-`[class[i], Z[class[i]][i]]` ensures that our parameters are index by two variables. The first indicates the class and the second indicates the component of the mixture model within that class.
- Note that `Z[class[i]]` returns the categorical distribution of class `class[i]` at which point `Z[class[i]][i]` gets the random variable of  categorical distribution `Z[class[i]]`
- Take note of the output from `@Parameter(model, :mu)`

Note that this is very similar to `MclustDA` in `R`.

#### Mixture desciminant analysis with shared eigenvectors within classes

To do the same except assume the covariance matrices for the components within each class share the same eigenvectors we can do the following

```julia
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Categorical(model, Z, [2,2,2]);
@Observation(model, X[i] = iris_m[i] -> (:mu => [class[i], Z[class[i]][i]],
                                        :a => [class[i], Z[class[i]][i]],
                                        :L => [class[i], Z[class[i]][i]],
                                        :V => class[i]), i = 1:n)
EM!(model; n_init=10, n_wild=10)
@Observation(model, X_new[i] = iris_m[i] -> (:mu => [class[i, "T"], Z[class[i, "T"]][i, "T"]],
                                            :a => [class[i, "T"], Z[class[i, "T"]][i, "T"]],
                                            :L => [class[i, "T"], Z[class[i, "T"]][i, "T"]],
                                            :V => class[i]), i = 1:n)
id = @posterior_probability(model, [class[i, "T"]], i = 1:n)();
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)
```

#### Mixture desciminant analysis with shared eigenvectors within classes

What is we wish to do the same but now assume there is a single set of eigenvectors shared between all components across all classes. This is done with

```julia
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Categorical(model, Z, [2,2,2]);
@Observation(model, X[i] = iris_m[i] -> (:mu => [class[i], Z[class[i]][i]],
                                        :a => [class[i], Z[class[i]][i]],
                                        :L => [class[i], Z[class[i]][i]],
                                        :V => 1), i = 1:n)
EM!(model; n_init=10, n_wild=10)
@Observation(model, X_new[i] = iris_m[i] -> (:mu => [class[i, "T"], Z[class[i, "T"]][i, "T"]],
                                            :a => [class[i, "T"], Z[class[i, "T"]][i, "T"]],
                                            :L => [class[i, "T"], Z[class[i, "T"]][i, "T"]],
                                            :V => 1), i = 1:n)
id = @posterior_probability(model, [class[i, "T"]], i = 1:n)();
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)
```

#### Mixture desciminant analysis with two covariance matrices shared across between all components

This model is much more complicated but allows for a dramatic reduce is the number of parameters in the model. We again assume that each class follows a mixture model but now assume that each covariance matrix of each component is one of two possible covariance matrices

```julia
model = Parsa_Model(Normal_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Categorical(model, Z, [2,2,2]);
@Categorical(model, cov, 2);
@Observation(model, X[i] = iris_m[i] -> (:mu => [class[i], Z[class[i]][i]],
                                        :cov => cov[class[i], Z[class[i]][i]]), i = 1:n)
EM!(model; n_init=3, n_wild=10)
@Parameter(model, :cov)
```
- Because of the nesting of multiple random variables via `cov[class[i], Z[class[i]][i]]`, many of the observations are dependent of eachother. This creates a complex likelihood structures and leads to a slowing algorithm as the computational expense has been increased.
- Take note of the output of `@Parameter(model, :cov)` and the number of covariance parameters.

If we where to introduce a new random variable of unknown class and try to predict its class, because of the complex dependency structure, is it nearly computationally infeasible. To fix this, we can predict the most likely covariance matrix for each component and assign it to the respective component.

```julia
G = @posterior_probability(model, [cov[i]], i = reduce(vcat, [[[i,j] for i in 1:K] for j in 1:2]))()
for (key, M) in G
    mm = Dict(key => M.max)
    @Known(model, cov[i] = mm[i], i = [key])
end
```

We have now avoided the computational complexity and can predict new observations.

```julia
@Observation(model, X_new[i] = iris_m[i] = (:mu => [class[i, "T"], Z[class[i, "T"]][i, "T"]],
                                            :cov => cov[class[i, "T"], Z[class[i, "T"]][i, "T"]]), i = 1:n)
id = @posterior_probability(model, [class[i, "T"]], i = 1:n)();
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)
```

#### Mixture desciminant analysis with two covariance matrices shared across between all components and one set of eigenvectors shared across the two covariance matrices

We can now do something a little crazy and assume that the two covariance matrices within our model share the same eigenvectors.

```julia
K = 3
model = Parsa_Model(Normal_Parsa_Model(p));
@Categorical(model, class, K);
@Known(model, class[i] = class[i], i = 1:n)
@Categorical_Set(model, Z, [2,2,2]);
@Categorical(model, cov, 2);
@Observation(model, X[i] = iris_m[i] -> (:mu => [class[i], Z[class[i]][i]],
                                         :a => cov[class[i], Z[class[i]][i]],
                                         :L => cov[class[i], Z[class[i]][i]],
                                         :V => 1), i = 1:n)
EM!(model; n_init=10, n_wild=10)
@Parameter(model, :L)
@Parameter(model, :V)
G = @posterior_probability(model, [cov[i]], i = reduce(vcat, [[[i,j] for i in 1:K] for j in 1:2]))()
for (key, M) in G
    mm = Dict(key => M.max)
    @Known(model, cov[i] = mm[i], i = [key])
end
@Observation(model, X_new[i] = iris_m[i] = (:mu => [class[i, "T"], Z[class[i, "T"]][i, "T"]],
                                            :a => cov[class[i, "T"], Z[class[i, "T"]][i, "T"]],
                                            :L => cov[class[i, "T"], Z[class[i, "T"]][i, "T"]],
                                            :V => 1), i = 1:n)
id = @posterior_probability(model, [class[i, "T"]], i = 1:n)();
id_ = [id[i].max for i in 1:n]
mean(id_ .== class)
```
- Note the outputs of `@Parameter(model, :L)` and `@Parameter(model, :V)` and the number of parameters estimated.

### Semi-Supervised Gaussian Mixture Models

Suppose for whatever reason we know the true species of some of the observations in the iris dataset. We will simulate this using the following where we assume we know the label of $30$ of the observations.

```julia
known_samples = sample(1:n, 30; replace=false)
known_map = Dict([s => class[s] for s in known_samples])
```

We can now run the model which is similar to before but with one additional line.
```julia
model = Parsa_Model(Normal_Model(p));
@Categorical(model, Z, K);
@Known(model, Z[i] = known_map[i], i = known_samples)
@Observation(model, X[i] = iris_m[i] -> (:mu => Z[i], :cov => Z[i]), i = 1:n)
EM!(model; n_init=10, n_wild=10)
id = @posterior_probability(model, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)
```
-`@Known(model, Z[i] = known_map[i], i = known_samples)` simply assigns the value of `known_map[i]` to the respective random variable `Z[i]` and changes it to a known variable. Thus `Z[i]` can no longer take the values $1,2,\dots, K$ and instead is always the value of `known_map[i]`.

#### Semi-Supervised Gaussian Mixture Models with Positive Constraints

Assume we know every two observations in the iris dataset came from the same yet unknown component. This information can be incorporated as follows by first setting up some needed information.

```julia
blocks = Int.(repeat(1:(n/2),inner=2))
n_blocks = length(unique(blocks))
true_class_block = [class[i] for i in 1:n if i % 2 == 0]
```
We can now setup the model as follows
```julia
model = Parsa_Model(Normal_Model(p));
@Categorical(model, Z, K);
@Categorical(model, B, n_blocks);
@Known(model, B[i] = blocks[i], i = 1:n)
@Observation(model, X[i] = iris_m[i] -> (:mu => Z[B[i]], :cov => Z[B[i]]), i = 1:n)
EM!(model; n_init=20, n_wild=30)
id = @posterior_probability(model, [Z[i]], i = 1:n_blocks)()
id_ = [id[i].max for i in 1:n_blocks]
randindex(id_, true_class_block)
```

#### Semi-Supervised Gaussian Mixture Models with Negative Constraints

Suppose we know that the first two observations are from different components or species (this is not true but we will go with it). We can incorporate this into the model with negative constraints in the following way.

```julia
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
```
- The final line outputs the predicted species of the first and second observation in the iris dataset. Notice that they are not the same as was enforced.

## 📖 Package Reference

#### `Parsa_Model(base)`

Setup the model and set the base distributional assumption of the model.

<!-- **Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `base` | Parsa_Base | No | distributional assumption |

**Returns:**

Returns a Parsa_Module to build the model -->

#### `@Categorical(model, name, K)`

Creates a new categorical distribution inside of `model` with `K` categories with the name of `name`.

<!-- **Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | Parsa_Module | Yes | model space |
| `var`   | None | Yes | Name of the distribution|
| `K`     | Int  | Yes | Number of categories -->


#### `@Observation(model, X[i] = Y[i] -> (:par => ), i = N)`

Adds a new observations named `X` indexed by `i` with the value of `Y[i]` and the mapping `(:par => )`. This is repeated for each `i` in `N`

#### `@Known(model, name[i] = C[i], i = N)`
Sets the random variables `name[i]` to known with the value of `C[i]`. This is repeated for each `i` in `N`.

#### `@Initialize(model, name[i] = C[i], i = N)`
Sets the random variables `name[i]` to the value of `C[i]`. This is repeated for each `i` in `N`.

#### `@Constant(model, :par[i] = C[i], i = N)`
Sets the parameter `:par[i]` to be constant with the value of `C[i]`. This is repeated for each `i` in `N`.

#### `@Parameter(model, :par)` or `@Parameter(model, name)`
Returns either the parameters of `:par` or the probabilities of the categorical distribution `name`.

#### `EM!(model; n_init, n_wild, should_initialize)`
Fits the model by running the EM algorithm and finding parameter estimates.

#### `@posterior_probability(model, [var1, var2, ...], i=N)`
Returns a function that returns the posterior probabilities of `[var1, var2, ...]` for each `i` in `N`.

#### `@BIC(model)`
Returns the bayesian information criterion of the model.

#### `@likelihood(model, X[i], i = N)`
Returns a function that returns the likelihood `$f(X[N_1], X[N_2], \dots, X[N_m])$`.


#### `@ObservationUpdater(model, X[i], i=N)`
Returns a function that takes in `$|N|$` observations and will assign them to the corresponding `X[i]`.

#### `Normal_Model(p)`
Returns the base for a `p`-dimensional Gaussian distribution.

### `Normal_Parsa_Model(p)`
Returns the base for a `p`-dimensional Gaussian distribution with the parsimonious parameterization.

<!-- **Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | Parsa_Module | Yes | model space |
| `var`   | None | Yes | Name of the distribution|
| `K`     | Int  | Yes | Number of categories -->

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgements

- [Library Name](https://github.com/user/repo) - For inspiration and some code patterns
- [Another Library](https://github.com/user/repo) - For the excellent algorithms
- All our contributors and users

---
