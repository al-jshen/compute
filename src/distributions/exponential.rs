use crate::distributions::*;

/// Implements the [Exponential](https://en.wikipedia.org/wiki/Exponential_distribution)
/// distribution.
#[derive(Debug, Clone, Copy)]
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

impl Default for Exponential {
    fn default() -> Self {
        Self::new(1.)
    }
}

impl Distribution for Exponential {
    type Output = f64;
    /// Samples from the given Exponential distribution.
    ///
    /// # Remarks
    /// Uses the [inverse transform
    /// sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling) method.
    fn sample(&self) -> f64 {
        -self.rng.sample().ln() / self.lambda
    }
}

impl Distribution1D for Exponential {
    fn update(&mut self, params: &[f64]) {
        self.set_lambda(params[0]);
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
            return 0.;
        }
        self.lambda * (-self.lambda * x).exp()
    }
}

impl Mean for Exponential {
    type MeanType = f64;
    /// Returns the mean of the given exponential distribution.
    fn mean(&self) -> f64 {
        1. / self.lambda
    }
}

impl Variance for Exponential {
    type VarianceType = f64;
    fn var(&self) -> f64 {
        1. / self.lambda.powi(2)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::statistics::{mean, var};
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_moments() {
        let data2 = Exponential::new(5.).sample_vec(1e6 as usize);
        assert_approx_eq!(1. / 5., mean(&data2), 1e-2);
        assert_approx_eq!(1. / 25., var(&data2), 1e-2);
    }
}
