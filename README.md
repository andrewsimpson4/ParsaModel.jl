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

The first example is how to implement a Gaussian mixture model using ParsaModel. This package is manly interacted with via macros which allows for a custom and minimal syntax.

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

```julia
id = @posterior_probability(model, [Z[i]], i = 1:n)()
id_ = [id[i].max for i in 1:n]
randindex(id_, class)
```

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
