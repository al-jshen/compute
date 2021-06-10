//! Provides a unified interface for working with probability distributions. Also implements
//! commonly used (maximum entropy) distributions.

mod bernoulli;
mod beta;
mod binomial;
mod chi_squared;
mod discreteuniform;
mod exponential;
mod gamma;
mod multivariatenormal;
mod normal;
mod pareto;
mod poisson;
mod t;
mod uniform;

use crate::linalg::{Matrix, Vector};

/// The primary trait defining a probability distribution.
pub trait Distribution: Send + Sync {
    type Output;
    /// Samples from the given probability distribution.
    fn sample(&self) -> Self::Output;
}

/// A trait defining a one dimensional distribution.
pub trait Distribution1D: Distribution<Output = f64> {
    /// Generates a vector of `n` randomly sampled values from the given probability distribution.
    fn sample_n(&self, n: usize) -> Vector {
        (0..n).map(|_| self.sample()).collect()
    }
    /// Generates a matrix of size `n x m` with values randomly sampled from the given
    /// distribution.
    fn sample_matrix(&self, nrows: usize, ncols: usize) -> Matrix {
        Matrix::new(self.sample_n(nrows * ncols), nrows as i32, ncols as i32)
    }
    /// Update the parameters of the distribution.
    fn update(&mut self, params: &[f64]);
}

/// A trait defining a multidimensional probability distribution.
pub trait DistributionND: Distribution<Output = Vector> {
    fn get_dim(&self) -> usize;
    /// Generate a matrix of samples, where each row in the matrix is a random sample from the
    /// given distribution.
    fn sample_n(&self, n: usize) -> Matrix {
        let mut data = Vector::with_capacity(n * self.get_dim());
        for _ in 0..n {
            data.extend(self.sample());
        }
        Matrix::new(data, n as i32, self.get_dim() as i32)
    }
}

/// Provides a trait for computing the mean of a distribution where there is a closed-form
/// expression.
pub trait Mean {
    type MeanType;
    /// Calculates the mean of the distribution.
    fn mean(&self) -> Self::MeanType;
}

/// Provides a trait for computing the variance of a distribution where there is a closed-form
/// solution.
pub trait Variance {
    type VarianceType;
    fn var(&self) -> Self::VarianceType;
}

/// Provides a trait for interacting with continuous probability distributions.
pub trait Continuous {
    type PDFType;
    /// Calculates the [probability density
    /// function](https://en.wikipedia.org/wiki/Probability_density_function) at some value `x`.
    fn pdf(&self, x: Self::PDFType) -> f64;
}

/// Provides a trait for interacting with discrete probability distributions.
pub trait Discrete: Distribution1D {
    /// Calculates the [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function) at some value `x`.
    fn pmf(&self, x: i64) -> f64;
}

pub use self::bernoulli::Bernoulli;
pub use self::beta::Beta;
pub use self::binomial::Binomial;
pub use self::chi_squared::ChiSquared;
pub use self::discreteuniform::DiscreteUniform;
pub use self::exponential::Exponential;
pub use self::gamma::Gamma;
pub use self::multivariatenormal::*;
pub use self::normal::Normal;
pub use self::pareto::Pareto;
pub use self::poisson::Poisson;
pub use self::t::*;
pub use self::uniform::Uniform;
