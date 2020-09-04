//! Provides a unified interface for working with probability distributions. Also implements
//! commonly used (maximum entropy) distributions.

mod exponential;
mod gamma;
mod normal;
mod uniform;

/// The primary trait defining a probability distribution.
pub trait Distribution {
    /// Samples from the given probability distribution.
    fn sample(&self) -> f64;
    /// Generates a vector of `n` randomly sampled values from the given probability distribution.
    fn sample_iter(&self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample()).collect()
    }
}

/// Provides a trait for interacting with continuous probability distributions.
pub trait Continuous: Distribution {
    /// Calculates the [probability density
    /// function](https://en.wikipedia.org/wiki/Probability_density_function) at some value `x`.
    fn pdf(&self, x: f64) -> f64;
}

pub use self::exponential::Exponential;
pub use self::gamma::Gamma;
pub use self::normal::Normal;
pub use self::uniform::Uniform;
