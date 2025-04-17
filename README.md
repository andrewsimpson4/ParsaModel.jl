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

The first example is how to implement a Gaussian mixture model using ParsaModel. This package is manly interacted with via macros which allows for a custom and minimal syntax. Here we are using a finite mixture model to cluster the observations in the iris dataset with the goal of clustering and recovering species. Thus we will look look for $3$ clusters.

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

![](./Assets/ex_vid.gif)

The purple lines here are each of the initialization runs and the final green line is taking the best initialization and running the algorithm until convergence is reached.

Now that the model has been fit, we can look at the parameter estimates of the model. Since we used `Normal_Model` as the base, we have parameters for `:mu` and `:cov`. This can be viewed with the following.

```julia
@Parameter(model, :mu)
@Parameter(model, :cov)
```

Notice here that we have $3$ `:mu` parameters and $3$ `:cov` paramters since we fit a $3$ component mixture model.

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
@Observation(model, X[i] = iris_m[i] = (:mu => Z[i], :cov => Z[i]), i = 1:n);
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

## 📖 Package Reference

### `mainFunction(options)`

The main entry point for the library.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `options` | Object | No | Configuration options |
| `options.option1` | String | No | Description of option1 |
| `options.option2` | Boolean | No | Description of option2 |

**Returns:**

Returns an instance with the following methods:

- `doSomething()`: Description of what this method does
- `getData()`: Description of what this method does

### `utilities.transform(data)`

A utility function for transforming data.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | Any | Yes | The data to transform |

**Returns:**

The transformed data.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgements

- [Library Name](https://github.com/user/repo) - For inspiration and some code patterns
- [Another Library](https://github.com/user/repo) - For the excellent algorithms
- All our contributors and users

---
