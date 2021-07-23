# compute

<!-- [![Build Status](https://travis-ci.org/al-jshen/compute.svg?branch=master)](https://travis-ci.org/al-jshen/compute) -->

[![Crates.io](https://img.shields.io/crates/v/compute)](https://crates.io/crates/compute)
[![Documentation](https://docs.rs/compute/badge.svg)](https://docs.rs/compute)
![License](https://img.shields.io/crates/l/compute?label=License)

A crate for scientific and statistical computing. For a list of what this crate provides, see [`FEATURES.md`](FEATURES.md). For more detailed explanations, see the [documentation](https://docs.rs/compute).

To use the latest stable version in your Rust program, add the following to your `Cargo.toml` file:

```rust
// Cargo.toml
[dependencies]
compute = "0.1"
```

For the latest version, add the following to your `Cargo.toml` file:

```rust
[dependencies]
compute = { git = "https://github.com/al-jshen/compute" }
```

There are many functions which rely on linear algebra methods. You can either use the provided Rust methods (default), or use BLAS and/or LAPACK by activating the `"blas"` and/or the `"lapack"` feature flags in `Cargo.toml`:

```rust
// example with BLAS only
compute = {version = "0.1", features = ["blas"]}
```

## Examples

### Statistical distributions

```rust
use compute::distributions::*;

let beta = Beta::new(2., 2.);
let betadata = b.sample_n(1000); // vector of 1000 variates

println!("{}", beta.mean()); // analytic mean
println!("{}", beta.var()); // analytic variance
println!("{}", beta.pdf(0.5)); // probability distribution function

let binom = Binomial::new(4, 0.5);

println!("{}", p.sample()); // sample single value
println!("{}", p.pmf(2));  // probability mass function
```

### Polynomial Regression and GLMs

```rust
use compute::prelude::*;

let x = vec![1., 2., 3., 4.];
let xd = design(&x, x.len()); // make a design matrix
let y = vec![3., 5., 7., 9.];

let mut clf = PolynomialRegressor::new(2); // degree 2 (i.e. quadratic)
clf.fit(&x, &y);                           // linear least squares fitting
println!("{:?}", clf.coef);                // get model coefficients

let y_bin = vec![0., 0., 1., 1.];
let mut glm = GLM::new(ExponentialFamily::Bernoulli);  // logistic regression
glm.set_penalty(1.);                                   // L2 penalty
glm.fit(&xd, &y, 25).unwrap();                         // fit with scoring algorithm (MLE), cap iterations at 25
println!("{:?}", glm.coef().unwrap());                          // get estimated coefficients
println!("{:?}", glm.coef().coef_covariance_matrix().unwrap()); // get covariance matrix for estimated coefficients
```

### Time series models

```rust
use compute::timeseries::*;

let x = vec![-2.584, -3.474, -1.977, -0.226, 1.166, 0.923, -1.075, 0.732, 0.959];

let mut ar = AR::new(1);             // AR(1) model
ar.fit(&x);                          // fit model with Yule-Walker equations
println!("{:?}", ar.coeffs);         // get model coefficients
println!("{:?}", ar.predict(&x, 5)); // forecast 5 steps ahead
```

### Numerical integration

```rust
use compute::integrate::*;

let f = |x: f64| x.sqrt() + x.sin() - (3. * x).cos() - x.powi(2);
println!("{}", trapz(f, 0., 1., 100));        // trapezoid integration with 100 segments
println!("{}", quad5(f, 0., 1.));             // gaussian quadrature integration
println!("{}", romberg(f, 0., 1., 1e-8, 10)); // romberg integration with tolerance and max steps
```

### Linear algebra

```rust
use compute::linalg::*;

let x = vec![
  2., 3.,
  4., 5.,
];
let y = vec![
5., 2.,
6., 1.,
];
println!("{:?}", lu(&x));                            // calculate LU decomposition for x
println!("{:?}", solve_sys(&x, &y));                 // solve 2x2 system of equations (each column of y is a system)
println!("{:?}", matmul(&x, &y, 2, 2, false, true)); // matrix multiply, transposing y
```

### Data summary functions

```rust
use compute::statistics::*;
use compute::linalg::Vector;

let x = Vector::from([2.2, 3.4, 5., 10., -2.1, 0.1]);

println!("{}", x.mean());
println!("{}", x.var());
println!("{}", x.max());
println!("{}", x.argmax());
```

### Mathematical and statistical functions

```rust
use compute::functions::*;

println!("{}", logistic(4.));
println!("{}", boxcox(5., 2.);      // boxcox transform
println!("{}", digamma(2.));
println!("{}", binom_coeff(10, 4)); // n choose k
```
