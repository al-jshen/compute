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

## Statistical distributions
```rust
use compute::distributions::*;

let beta = Beta::new(2., 2.); 
let betadata: Vec<f64> = b.sample_vec(1000); // vector of 1000 variates

println!("{}", beta.mean());
println!("{}", beta.var());
println!("{}", beta.pdf(0.5)); // probability distribution function

let binom = Binomial::new(4, 0.5);

println!("{}", p.sample()); // sample single value
println!("{}", p.pmf(2)); // probability mass function
```

## Summary functions
```rust
use compute::summary::*;

let x = vec![2.2, 3.4, 5., 10., -2.1, 0.1];
let y = vec![1.,  2., -2., 5.7, -0.7, 5.7];

println!("{}", mean(&x));
println!("{}", var(&x));
println!("{}", max(&y));
println!("{}", sample_std(&y));
println!("{}", covariance(&x, &y));
```

## Numerical integration
```rust
use compute::integrate::*;

let f = |x: f64| x.sqrt() + x.sin() - (3. * x).cos() - x.powi(2);
println!("{}", trapz(f, 0., 1., 100)); // trapezoid integration with 100 segments
println!("{}", quad5(f, 0., 1.)); // gaussian quadrature integration
println!("{}", romberg(f, 0., 1., 1e-8, 10)); // romberg integration with tolerance and max steps
```

## Regression
```rust
use compute::predict::*;

let x = vec![1., 2., 3., 4.];
let y = vec![3., 5., 7., 9.];

let mut clf = Polynomial::new(1); // degree 1 (i.e. linear regressor)
clf.fit(&x, &y); // linear least squares fitting
println!("{:?}", clf.get_coeffs());
```

## Functions
```rust
use compute::functions::*;

println!("{}", logistic(4.));
println!("{}", boxcox(5., 2.); // boxcox transform
println!("{}", digamma(2.));
println!("{}", binom_coeff(10, 4)); // n choose k
```

## Utility functions
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
println!("{:?}", invert_matrix(&x)); // invert matrix
println!("{:?}", xtx(&y)); // x transpose times x
println!("{:?}", dot(&x, &y)); // dot product of x and y (two vectors)
println!("{:?}", matmul(&x, &y, 2, 2, false, true)); // matrix multiply, transposing y
```

