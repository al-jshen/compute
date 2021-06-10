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
        assert!(dof > 0, "Degrees of freedom must be positive.");
        ChiSquared {
            dof,
            sampler: Gamma::new((dof as f64) / 2., 0.5),
        }
    }
    pub fn set_dof(&mut self, dof: usize) -> &mut Self {
        assert!(dof > 0, "Degrees of freedom must be positive.");
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
    type Output = f64;
    /// Samples from the given Chi square distribution.
    fn sample(&self) -> f64 {
        self.sampler.sample()
    }
}

impl Distribution1D for ChiSquared {
    fn update(&mut self, params: &[f64]) {
        self.set_dof(params[0] as usize);
    }
}

impl Continuous for ChiSquared {
	type PDFType = f64;
    /// Calculates the probability density function for the given Chi square distribution at `x`.
    ///
    /// # Remarks
    /// If `dof = 1` then x should be positive. Otherwise, x should be non-negative. If these
    /// conditions are not met, then the probability of x is 0.
    fn pdf(&self, x: f64) -> f64 {
        if (self.dof == 1 && x <= 0.) || (x < 0.) {
            return 0.;
        }
        let half_k = (self.dof as f64) / 2.;
        1. / (2_f64.powf(half_k) * gamma(half_k)) * x.powf(half_k - 1.) * (-x / 2.).exp()
    }
}

impl Mean for ChiSquared {
    type MeanType = f64;
    /// Calculates the mean of the Chi square distribution, which is the same as its degrees of
    /// freedom.
    fn mean(&self) -> f64 {
        self.dof as f64
    }
}

impl Variance for ChiSquared {
    type VarianceType = f64;
    /// Calculates the variance of the Chi square distribution.
    fn var(&self) -> f64 {
        self.mean() * 2.
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::statistics::{mean, var};
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_moments() {
        let data1 = ChiSquared::new(2).sample_n(1e6 as usize);
        assert_approx_eq!(2., mean(&data1), 1e-2);
        assert_approx_eq!(4., var(&data1), 1e-2);

        let data2 = ChiSquared::new(5).sample_n(1e6 as usize);
        assert_approx_eq!(5., mean(&data2), 1e-2);
        assert_approx_eq!(10., var(&data2), 1e-2);
    }
}
