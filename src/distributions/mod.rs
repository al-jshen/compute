//! Provides a unified interface for working with probability distributions. Also implements
//! commonly used (maximum entropy) distributions.

mod bernoulli;
mod beta;
mod chi_squared;
mod discreteuniform;
mod exponential;
mod gamma;
mod normal;
mod poisson;
mod uniform;

/// The primary trait defining a probability distribution.
pub trait Distribution: Send + Sync {
    /// Samples from the given probability distribution.
    fn sample(&self) -> f64;
    /// Generates a vector of `n` randomly sampled values from the given probability distribution.
    fn sample_iter(&self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample()).collect()
    }
    fn update(&mut self, params: &[f64]);
}

/// Provides a trait for computing the mean of a distribution where there is a closed-form
/// expression.
pub trait Mean: Distribution {
    /// Calculates the mean of the distribution.
    fn mean(&self) -> f64;
}

/// Provides a trait for computing the variance of a distribution where there is a closed-form
/// solution. Requires the `Mean` trait to be implemented because of the definition of variance.
pub trait Variance: Mean {
    fn var(&self) -> f64;
}

/// Provides a trait for interacting with continuous probability distributions.
pub trait Continuous: Distribution {
    /// Calculates the [probability density
    /// function](https://en.wikipedia.org/wiki/Probability_density_function) at some value `x`.
    fn pdf(&self, x: f64) -> f64;
}

/// Provides a trait for interacting with discrete probability distributions.
pub trait Discrete: Distribution {
    /// Calculates the [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function) at some value `x`.
    fn pmf(&self, x: i64) -> f64;
}

pub use self::bernoulli::Bernoulli;
pub use self::beta::Beta;
pub use self::chi_squared::ChiSquared;
pub use self::discreteuniform::DiscreteUniform;
pub use self::exponential::Exponential;
pub use self::gamma::Gamma;
pub use self::normal::Normal;
pub use self::poisson::Poisson;
pub use self::uniform::Uniform;
