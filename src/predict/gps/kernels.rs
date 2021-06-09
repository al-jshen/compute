//! Kernels (covariance functions) for Gaussian processes. See the [Kernel
//! Cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/) for more details about kernels.

use crate::linalg::{Dot, Matrix, Vector};

pub trait Kernel<T, S> {
    fn forward(&self, x: T, y: T) -> S;
}

/// The [radial basis function kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel).
/// Also called the squared exponential kernel.
pub struct RBFKernel {
    /// output variance parameter
    var: f64,
    /// length scale parameter
    length_scale: f64,
}

/// Type alias for the RBF kernel.
pub type SquaredExponentialKernel = RBFKernel;

impl RBFKernel {
    pub fn new(var: f64, length_scale: f64) -> Self {
        Self { var, length_scale }
    }
}

/// The [rational quadratic
/// kernel](https://en.wikipedia.org/wiki/Rational_quadratic_covariance_function).
pub struct RationalQuadraticKernel {
    /// output variance parameter
    var: f64,
    /// scale mixture parameter
    alpha: f64,
    /// length scale of kernel
    length_scale: f64,
}

/// Type alias for the rational quadratic kernel.
pub type RQKernel = RationalQuadraticKernel;

impl RQKernel {
    pub fn new(var: f64, alpha: f64, length_scale: f64) -> Self {
        Self {
            var,
            alpha,
            length_scale,
        }
    }
}

/// Periodic kernel.
struct PeriodicKernel {
    /// output variance parameter
    var: f64,
    /// period parameter
    p: f64,
    /// length scale parameter
    length_scale: f64,
}

macro_rules! impl_kernel_f64_for_rbf {
    ($t1: ty) => {
        impl Kernel<$t1, f64> for RBFKernel {
            fn forward(&self, x: $t1, y: $t1) -> f64 {
                (-(x - y).powi(2) / (2. * self.length_scale.powi(2))).exp() * self.var
            }
        }
    };
}

impl_kernel_f64_for_rbf!(f64);
impl_kernel_f64_for_rbf!(&f64);

macro_rules! impl_kernel_f64_for_rq {
    ($t1: ty) => {
        impl Kernel<$t1, f64> for RationalQuadraticKernel {
            fn forward(&self, x: $t1, y: $t1) -> f64 {
                (1. + (x - y).powi(2) / (2. * self.alpha * self.length_scale.powi(2)))
                    .powf(self.alpha)
                    * self.var
            }
        }
    };
}

impl_kernel_f64_for_rq!(f64);
impl_kernel_f64_for_rq!(&f64);

macro_rules! impl_kernel_vec_for_rbf {
    ($t1: ty, $t2: ty) => {
        impl Kernel<$t1, $t2> for RBFKernel {
            fn forward(&self, x: $t1, y: $t1) -> $t2 {
                let (x, y) = (x.reshape(-1, 1), y.reshape(-1, 1));
                (-(x.powi(2).reshape(-1, 1) + y.powi(2).reshape(1, -1) - 2. * x.dot_t(y))
                    / (2. * self.length_scale.powi(2)))
                .exp()
                    * self.var
            }
        }
    };
}

impl_kernel_vec_for_rbf!(Matrix, Matrix);
impl_kernel_vec_for_rbf!(Vector, Matrix);
impl_kernel_vec_for_rbf!(&Matrix, Matrix);
impl_kernel_vec_for_rbf!(&Vector, Matrix);

macro_rules! impl_kernel_vec_for_rq {
    ($t1: ty, $t2: ty) => {
        impl Kernel<$t1, $t2> for RationalQuadraticKernel {
            fn forward(&self, x: $t1, y: $t1) -> $t2 {
                let (x, y) = (x.reshape(-1, 1), y.reshape(-1, 1));
                (1. + (x.powi(2).reshape(-1, 1) + y.powi(2).reshape(1, -1) - 2. * x.dot_t(y))
                    / (2. * self.alpha * self.length_scale.powi(2)))
                .powf(self.alpha)
                    * self.var
            }
        }
    };
}

impl_kernel_vec_for_rq!(Matrix, Matrix);
impl_kernel_vec_for_rq!(Vector, Matrix);
impl_kernel_vec_for_rq!(&Matrix, Matrix);
impl_kernel_vec_for_rq!(&Vector, Matrix);
