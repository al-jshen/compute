#![allow(clippy::float_cmp)]
use crate::distributions::*;

/// Implements the [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution).
#[derive(Debug, Clone, Copy)]
pub struct Bernoulli {
    /// Probability `p` of the Bernoulli distribution
    p: f64,
}

impl Bernoulli {
    /// Create a new Bernoulli distribution with probability `p`.
    ///
    /// # Errors
    /// Panics if p is not in [0, 1].
    pub fn new(p: f64) -> Self {
        if !(0. ..=1.).contains(&p) {
            panic!("`p` must be in [0, 1].");
        }
        Bernoulli { p }
    }
    pub fn set_p(&mut self, p: f64) -> &mut Self {
        if !(0. ..=1.).contains(&p) {
            panic!("`p` must be in [0, 1].");
        }
        self.p = p;
        self
    }
}

impl Default for Bernoulli {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Distribution for Bernoulli {
    /// Samples from the given Bernoulli distribution.
    fn sample(&self) -> f64 {
        if self.p == 1. {
            return 1.;
        } else if self.p == 0. {
            return 0.;
        }

        if self.p > fastrand::f64() {
            1.
        } else {
            0.
        }
    }
    fn update(&mut self, params: &[f64]) {
        self.set_p(params[0]);
    }
}

impl Discrete for Bernoulli {
    /// Calculates the [probability mass
    /// function](https://en.wikipedia.org/wiki/Probability_mass_function) for the given  Bernoulli
    /// distribution at `x`.
    ///
    fn pmf(&self, k: i64) -> f64 {
        if k == 0 {
            1. - self.p
        } else if k == 1 {
            self.p
        } else {
            0.
        }
    }
}

impl Mean for Bernoulli {
    /// Calculates the mean of the Bernoulli distribution, which is `p`.
    fn mean(&self) -> f64 {
        self.p
    }
}

impl Variance for Bernoulli {
    /// Calculates the variance, given by `p*q = p(1-p)`.
    fn var(&self) -> f64 {
        self.p * (1. - self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::{mean, var};
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_bernoulli() {
        let data = Bernoulli::new(0.75).sample_vec(1e6 as usize);
        for i in &data {
            assert!(*i == 0. || *i == 1.);
        }
        assert_approx_eq!(0.75, mean(&data), 1e-2);
        assert_approx_eq!(0.75 * 0.25, var(&data), 1e-2);
        assert!(Bernoulli::default().pmf(2) == 0.);
        assert!(Bernoulli::default().pmf(0) == 0.5);
    }
}
