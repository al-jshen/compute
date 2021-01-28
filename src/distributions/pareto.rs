use crate::distributions::*;

/// Implements the [Pareto](https://en.wikipedia.org/wiki/Pareto_distribution) distribution.
#[derive(Debug, Clone, Copy)]
pub struct Pareto {
    /// Shape parameter α.
    alpha: f64,
    /// Parameter which controls the minimum value of the distribution.
    minval: f64,
}

impl Pareto {
    /// Create a new Pareto distribution with shape `alpha` and minimum value `minval`.
    ///
    /// # Errors
    /// Panics if `alpha <= 0` or `minval <= 0`.
    pub fn new(alpha: f64, minval: f64) -> Self {
        if alpha <= 0. || minval <= 0. {
            panic!("Both alpha and beta must be positive.");
        }
        Pareto { alpha, minval }
    }
    pub fn set_alpha(&mut self, alpha: f64) -> &mut Self {
        if alpha <= 0. {
            panic!("Alpha must be positive.");
        }
        self.alpha = alpha;
        self
    }
    pub fn set_minval(&mut self, minval: f64) -> &mut Self {
        if minval <= 0. {
            panic!("minval must be positive.");
        }
        self.minval = minval;
        self
    }
}

impl Default for Pareto {
    fn default() -> Self {
        Self::new(1., 1.)
    }
}

impl Distribution for Pareto {
    /// Samples from the given Pareto distribution using inverse transform sampling.
    fn sample(&self) -> f64 {
        let u = fastrand::f64();
        self.minval / u.powf(1. / self.alpha)
    }
    fn update(&mut self, params: &[f64]) {
        assert!(params.len() == 2);
        self.set_alpha(params[0]).set_minval(params[1]);
    }
}

impl Continuous for Pareto {
    /// Calculates the probability density function for the given Pareto function at `x`.
    ///
    /// # Remarks
    /// This returns 0 if `x < minval`
    fn pdf(&self, x: f64) -> f64 {
        if x < self.minval {
            return 0.;
        }
        self.alpha * self.minval.powf(self.alpha) / x.powf(self.alpha - 1.)
    }
}

impl Mean for Pareto {
    /// Calculates the mean of the Pareto distribution.
    fn mean(&self) -> f64 {
        if self.alpha <= 1. {
            f64::INFINITY
        } else {
            self.alpha * self.minval / (self.alpha - 1.)
        }
    }
}

impl Variance for Pareto {
    /// Calculates the variance of the Pareto distribution.
    fn var(&self) -> f64 {
        if self.alpha <= 2. {
            f64::INFINITY
        } else {
            self.minval.powi(2) * self.alpha / ((self.alpha - 1.).powi(2) * (self.alpha - 2.))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::{mean, var};
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_moments() {
        let dist = Pareto::new(4., 4.);
        let data = dist.sample_vec(1e6 as usize);
        assert_approx_eq!(dist.mean(), mean(&data), 0.05);
        assert_approx_eq!(dist.var(), var(&data), 0.05);
    }
}
