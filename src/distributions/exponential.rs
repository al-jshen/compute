use crate::distributions::*;

/// Implements the [Exponential](https://en.wikipedia.org/wiki/Exponential_distribution)
/// distribution.
#[derive(Debug)]
pub struct Exponential {
    /// Rate parameter λ
    lambda: f64,
    /// Random number generator used to sample from the distribution. Uses a Uniform distribution
    /// in order to perform inverse transform sampling.
    rng: Uniform,
}

impl Exponential {
    /// Create a new Exponential distribution with rate parameter `lambda`.
    ///
    /// # Errors
    /// Panics if `lambda <= 0`.
    pub fn new(lambda: f64) -> Self {
        if lambda <= 0. {
            panic!("Lambda must be positive.");
        }
        Exponential {
            lambda,
            rng: Uniform::new(0., 1.),
        }
    }
    pub fn set_lambda(&mut self, lambda: f64) -> &mut Self {
        if lambda <= 0. {
            panic!("Lambda must be positive.")
        }
        self.lambda = lambda;
        self
    }
}

impl Distribution for Exponential {
    /// Samples from the given Exponential distribution.
    ///
    /// # Remarks
    /// Uses the [inverse transform
    /// sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling) method.
    fn sample(&self) -> f64 {
        -self.rng.sample().ln() / self.lambda
    }
}

impl Continuous for Exponential {
    /// Calculates the [probability density
    /// function](https://en.wikipedia.org/wiki/Probability_density_function) for the given Exponential
    /// distribution at `x`.
    ///
    /// # Remarks
    ///
    /// Returns `0.` if `x` is negative.
    fn pdf(&self, x: f64) -> f64 {
        if x < 0. {
            0.
        } else {
            self.lambda * (-self.lambda * x).exp()
        }
    }
}
