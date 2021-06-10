use crate::distributions::*;
use crate::functions::gamma;

/// Implements the [Student's T](https://en.wikipedia.org/wiki/Student%27s_t-distribution) distribution.
#[derive(Debug, Clone, Copy)]
pub struct T {
    /// Degrees of freedom
    dof: f64,
}

pub type StudentsT = T;

impl T {
    /// Create a new t distribution with
    ///
    /// # Errors
    /// Panics if degrees of freedom is not positive.
    pub fn new(dof: f64) -> Self {
        assert!(dof > 0., "Degrees of freedom must be positive.");
        T { dof }
    }
    pub fn set_dof(&mut self, dof: f64) -> &mut Self {
        assert!(dof > 0., "Degrees of freedom must be positive.");
        self.dof = dof;
        self
    }
}

impl Default for T {
    fn default() -> Self {
        Self::new(1.)
    }
}

impl Distribution for T {
    type Output = f64;
    /// Samples from the given T distribution.
    fn sample(&self) -> f64 {
        (self.dof / 2.).sqrt() * Normal::default().sample()
            / Gamma::new(self.dof / 2., 1.).sample().sqrt()
    }
}

impl Distribution1D for T {
    fn update(&mut self, params: &[f64]) {
        self.set_dof(params[0]);
    }
}

impl Continuous for T {
	type PDFType = f64;
    /// Calculates the probability density function for the given T distribution at `x`.
    fn pdf(&self, x: f64) -> f64 {
        gamma((self.dof + 1.) / 2.)
            / ((self.dof * std::f64::consts::PI).sqrt() * gamma(self.dof / 2.))
            * (1. + x.powi(2) / self.dof).powf(-(self.dof - 1.) / 2.)
    }
}

impl Mean for T {
    type MeanType = f64;
    /// Calculates the mean of the T distribution, which is 0 when the degrees of freedom is
    /// greater than 1, and undefined otherwise.
    ///
    fn mean(&self) -> f64 {
        if self.dof > 1. {
            0.
        } else {
            f64::NAN
        }
    }
}

impl Variance for T {
    type VarianceType = f64;
    /// Calculates the variance of the T distribution.
    ///
    /// # Remarks
    /// This is not defined when degrees of freedom is less than or equal to 1, and infinity when
    /// degrees of freedom is in (1, 2].
    fn var(&self) -> f64 {
        if self.dof > 2. {
            self.dof / (self.dof - 2.)
        } else if (1. < self.dof) & (self.dof <= 2.) {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::statistics::mean;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_moments() {
        let t = T::new(2.);
        let data = t.sample_n(1e6 as usize);
        assert_approx_eq!(mean(&data), 0., 1e-2);
    }
}
