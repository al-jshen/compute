use crate::distributions::*;
use crate::functions::gamma;

/// Implements the [Chi square](https://en.wikipedia.org/wiki/Chi-square_distribution) distribution.
#[derive(Debug, Clone, Copy)]
pub struct ChiSquared {
    /// Degrees of freedom (k)
    dof: usize,
    sampler: Gamma,
}

impl ChiSquared {
    /// Create a new Chi square distribution with
    ///
    /// # Errors
    /// Panics if degrees of freedom is not positive.
    pub fn new(dof: usize) -> Self {
        if dof <= 0 {
            panic!("Degrees of freedom must be positive.");
        }
        ChiSquared {
            dof,
            sampler: Gamma::new((dof as f64) / 2., 0.5),
        }
    }
    pub fn set_dof(&mut self, dof: usize) -> &mut Self {
        if dof <= 0 {
            panic!("Degrees of freedom must be positive.");
        }
        self.dof = dof;
        self
    }
}

impl Default for ChiSquared {
    fn default() -> Self {
        Self::new(1)
    }
}

impl Distribution for ChiSquared {
    /// Samples from the given Chi square distribution.
    fn sample(&self) -> f64 {
        self.sampler.sample()
    }
    fn update(&mut self, params: &[f64]) {
        self.set_dof(params[0] as usize);
    }
}

impl Continuous for ChiSquared {
    /// Calculates the probability density function for the given Chi square distribution at `x`.
    fn pdf(&self, x: f64) -> f64 {
        let half_k = (self.dof as f64) / 2.;
        1. / (2_f64.powf(half_k) * gamma(half_k)) * x.powf(half_k - 1.) * (-x / 2.).exp()
    }
}
