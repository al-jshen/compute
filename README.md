# compute

<!-- [![Build Status](https://travis-ci.org/al-jshen/compute.svg?branch=master)](https://travis-ci.org/al-jshen/compute) -->

[![Crates.io](https://img.shields.io/crates/v/compute.svg?style=for-the-badge&color=fc8d62&logo=rust)](https://crates.io/crates/compute)
[![Documentation](https://img.shields.io/badge/docs.rs-compute-5E81AC?style=for-the-badge&labelColor=555555&logoColor=white)](https://docs.rs/compute)
![License](https://img.shields.io/crates/l/compute?label=License&style=for-the-badge&color=62a69b)

A crate for scientific and statistical computing. For a list of what this crate provides, see [`FEATURES.md`](FEATURES.md). For more detailed explanations, see the [documentation](https://docs.rs/compute).

To use the latest stable version in your Rust program, add the following to your `Cargo.toml` file:

```rust
// Cargo.toml
[dependencies]
compute = "0.2"
```

For the latest version, add the following to your `Cargo.toml` file:

```rust
[dependencies]
compute = { git = "https://github.com/al-jshen/compute" }
```

There are many functions which rely on linear algebra methods. You can either use the provided Rust methods (default), or use BLAS and/or LAPACK by activating the `"blas"` and/or the `"lapack"` feature flags in `Cargo.toml`:

```rust
// example with BLAS only
compute = {version = "0.2", features = ["blas"]}
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

### Linear algebra

```rust
use compute::linalg::*;

let x = arange(1., 4., 0.1).ln_1p().reshape(-1, 3);  // automatic shape detection
let y = Vector::from([1., 2., 3.]);  // vector struct
let pd = x.t().dot(x);               // transpose and matrix multiply
let jitter = Matrix::eye(3) * 1e-6;  // elementwise operations
let c = (pd + jitter).cholesky();    // matrix decompositions
let s = c.solve(&y.exp());           // linear solvers
println!("{}", s);
```

### Linear models

```rust
use compute::prelude::*;

let x = vec![
    0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00,
    4.25, 4.50, 4.75, 5.00, 5.50,
];
let y = vec![
    0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1.,
];
let n = y.len();
let xd = design(&x, n);

let mut glm = GLM::new(ExponentialFamily::Bernoulli); // logistic regression
glm.set_penalty(1.);                                  // L2 penalty
glm.fit(&xd, &y, 25).unwrap();                        // with fit scoring algorithm (MLE)
let coef = glm.coef().unwrap();                       // get estimated parameters
let errors = glm.coef_standard_error().unwrap();      // get errors on parameters

println!("{:?}", coef);
println!("{:?}", errors);

```

### Optimization

```rust
use compute::optimize::*;

// define a function using a consistent optimization interface
fn rosenbrock<'a>(p: &[Var<'a>], d: &[&[f64]]) -> Var<'a> {
    assert_eq!(p.len(), 2);
    assert_eq!(d.len(), 1);
    assert_eq!(d[0].len(), 2);

    let (x, y) = (p[0], p[1]);
    let (a, b) = (d[0][0], d[0][1]);

    (a - x).powi(2) + b * (y - x.powi(2)).powi(2)
}

// set up and run optimizer
let init = [0., 0.];
let optim = Adam::with_stepsize(5e-4);
let popt = optim.optimize(rosenbrock, &init, &[&[1., 100.]], 10000);

println!("{:?}", popt);
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
