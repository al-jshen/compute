use crate::distributions::*;
use crate::functions::beta;

/// Implements the [Beta](https://en.wikipedia.org/wiki/Beta_distribution) distribution.
#[derive(Debug, Clone, Copy)]
pub struct Beta {
    /// Shape parameter α.
    alpha: f64,
    /// Shape parameter β.
    beta: f64,
    /// Gamma(alpha, 1) distribution used to sample gamma variables for the creation of beta
    /// variables.
    alpha_gen: Gamma,
    /// Gamma(beta, 1) distribution used to sample gamma variables for the creation of beta
    /// variables.
    beta_gen: Gamma,
}

impl Beta {
    /// Create a new Beta distribution with parameters `alpha` and `beta`.
    ///
    /// # Errors
    /// Panics if `alpha <= 0` or `beta <= 0`.
    pub fn new(alpha: f64, beta: f64) -> Self {
        if alpha <= 0. || beta <= 0. {
            panic!("Both alpha and beta must be positive.");
        }
        Beta {
            alpha,
            beta,
            alpha_gen: Gamma::new(alpha, 1.),
            beta_gen: Gamma::new(beta, 1.),
        }
    }
    pub fn set_alpha(&mut self, alpha: f64) -> &mut Self {
        if alpha <= 0. {
            panic!("Alpha must be positive.");
        }
        self.alpha = alpha;
        self.alpha_gen = Gamma::new(alpha, 1.);
        self
    }
    pub fn set_beta(&mut self, beta: f64) -> &mut Self {
        if beta <= 0. {
            panic!("Beta must be positive.");
        }
        self.beta = beta;
        self.beta_gen = Gamma::new(beta, 1.);
        self
    }
}

impl Default for Beta {
    fn default() -> Self {
        Self::new(1., 1.)
    }
}

impl Distribution for Beta {
    /// Samples from the given Beta distribution using the Gamma distribution.
    fn sample(&self) -> f64 {
        let x = self.alpha_gen.sample();
        x / (x + self.beta_gen.sample())
    }
    fn update(&mut self, params: &[f64]) {
        self.set_alpha(params[0]).set_beta(params[1]);
    }
}

impl Continuous for Beta {
    /// Calculates the probability density function for the given Beta function at `x`.
    ///
    /// # Remarks
    /// Returns 0. if x is not in `[0, 1]`
    fn pdf(&self, x: f64) -> f64 {
        if !(0. ..=1.).contains(&x) {
            return 0.;
        }
        x.powf(self.alpha - 1.) * (1. - x).powf(self.beta - 1.) / beta(self.alpha, self.beta)
    }
}

impl Mean for Beta {
    /// Returns the mean of the beta distribution, which for a B(a, b)
    /// distribution is given by `a / (a + b)`.
    fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }
}

impl Variance for Beta {
    /// Returns the variance of the beta distribution.
    fn var(&self) -> f64 {
        (self.alpha * self.beta)
            / ((self.alpha + self.beta).powi(2) * (self.alpha + self.beta + 1.))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::{mean, var};
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_moments() {
        let dist = Beta::new(2., 4.);
        let data = dist.sample_vec(1e6 as usize);
        assert_approx_eq!(dist.mean(), mean(&data), 1e-2);
        assert_approx_eq!(dist.var(), var(&data), 1e-2);
    }
}
