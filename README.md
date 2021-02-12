# compute

<!-- [![Build Status](https://travis-ci.org/al-jshen/compute.svg?branch=master)](https://travis-ci.org/al-jshen/compute) -->

[![Crates.io](https://img.shields.io/crates/v/compute)](https://crates.io/crates/compute)
[![Documentation](https://docs.rs/compute/badge.svg)](https://docs.rs/compute)

A crate for statistical computing.

To use this in your Rust program, add the following to your `Cargo.toml` file:

```rust
// Cargo.toml
[dependencies]
compute = "0.1"
```

There are many functions which rely on linear algebra methods. You can either use the provided Rust methods (default), or use BLAS and/or LAPACK. To do so, activate the `"blas"` and/or the `"lapack"` feature flags in `Cargo.toml`:

```rust
// example with BLAS only
compute = {version = "0.1", features = ["blas"]}
```

For a list of what this crate provides, see [`FEATURES.md`](FEATURES.md). For more detailed explanations, see the [documentation](https://docs.rs/compute).

## Examples

### Statistical distributions
```rust
use compute::distributions::*;

let beta = Beta::new(2., 2.); 
let betadata: Vec<f64> = b.sample_vec(1000); // vector of 1000 variates

println!("{}", beta.mean());
println!("{}", beta.var());
println!("{}", beta.pdf(0.5)); // probability distribution function

let binom = Binomial::new(4, 0.5);

println!("{}", p.sample()); // sample single value
println!("{}", p.pmf(2));  // probability mass function
```

### Regression
```rust
use compute::predict::*;

let x = vec![1., 2., 3., 4.];
let y = vec![3., 5., 7., 9.];

let mut clf = PolynomialRegressor::new(1); // degree 1 (i.e. linear regressor)
clf.fit(&x, &y);                           // linear least squares fitting
println!("{:?}", clf.get_coeffs());        // get model coefficients
```

### Time series models
```rust
use compute::timeseries::*;

let x = vec![-2.582184,-3.44017,-1.979827,-0.268826,1.162776,0.9260983,-1.075229,0.7232999,0.9659502,0.2425384];

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

### Data summary functions
```rust
use compute::statistics::*;

let x = vec![2.2, 3.4, 5., 10., -2.1, 0.1];
let y = vec![1.,  2., -2., 5.7, -0.7, 5.7];

println!("{}", mean(&x));
println!("{}", var(&x));
println!("{}", max(&y));
println!("{}", sample_std(&y));
println!("{}", covariance(&x, &y));
```

### Linear algebra functions
```rust
use compute::utils::*;

let x = vec![
  2., 3.,
  4., 5.,
];
let y = vec![
  5., 2.,
  6., 1.,
];
println!("{:?}", invert_matrix(&x));                 // invert matrix
println!("{:?}", xtx(&y));                           // x transpose times x
println!("{:?}", dot(&x, &y));                       // dot product of x and y (two vectors)
println!("{:?}", matmul(&x, &y, 2, 2, false, true)); // matrix multiply, transposing y
```

### Mathematical and statistical functions
```rust
use compute::functions::*;

println!("{}", logistic(4.));
println!("{}", boxcox(5., 2.);      // boxcox transform
println!("{}", digamma(2.));
println!("{}", binom_coeff(10, 4)); // n choose k
```
