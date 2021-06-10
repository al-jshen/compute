use crate::distributions::*;

/// Implements the [Uniform](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))
/// distribution.
#[derive(Debug, Clone, Copy)]
pub struct Uniform {
    /// Lower bound for the Uniform distribution.
    lower: f64,
    /// Upper bound for the Uniform distribution.
    upper: f64,
}

impl Uniform {
    /// Create a new Uniform distribution with lower bound `lower` and upper bound `upper`.
    ///
    /// # Errors
    /// Panics if `lower > upper`.
    pub fn new(lower: f64, upper: f64) -> Self {
        if lower > upper {
            panic!("`Upper` must be larger than `lower`.");
        }
        Uniform { lower, upper }
    }
    pub fn set_lower(&mut self, lower: f64) -> &mut Self {
        if lower > self.upper {
            panic!("Upper must be larger than lower.")
        }
        self.lower = lower;
        self
    }
    pub fn set_upper(&mut self, upper: f64) -> &mut Self {
        if self.lower > upper {
            panic!("Upper must be larger than lower.")
        }
        self.upper = upper;
        self
    }
}

impl Default for Uniform {
    fn default() -> Self {
        Self::new(0., 1.)
    }
}

impl Distribution for Uniform {
    type Output = f64;
    /// Samples from the given Uniform distribution.
    fn sample(&self) -> f64 {
        (self.upper - self.lower) * fastrand::f64() + self.lower
    }
}

impl Distribution1D for Uniform {
    fn update(&mut self, params: &[f64]) {
        self.set_lower(params[0]).set_upper(params[1]);
    }
}

impl Continuous for Uniform {
	type PDFType = f64;
    /// Calculates the [probability density
    /// function](https://en.wikipedia.org/wiki/Probability_density_function) for the given Uniform
    /// distribution at `x`.
    ///
    /// # Remarks
    ///
    /// Returns `0.` if `x` is not in `[lower, upper]`
    fn pdf(&self, x: f64) -> f64 {
        if x < self.lower || x > self.upper {
            0.
        } else {
            1. / (self.upper - self.lower)
        }
    }
}

impl Mean for Uniform {
    type MeanType = f64;
    /// Calculates the mean, which for a Uniform(a, b) distribution is given by `(a + b) / 2`.
    fn mean(&self) -> f64 {
        (self.lower + self.upper) / 2.
    }
}

impl Variance for Uniform {
    type VarianceType = f64;
    /// Calculates the variance of the given Uniform distribution.
    fn var(&self) -> f64 {
        (self.upper - self.lower).powi(2) / 12.
    }
}

#[test]
fn inrange() {
    let u = self::Uniform::new(-2., 6.);
    let samples = u.sample_n(100);
    samples.into_iter().for_each(|x| {
        assert!(-2. <= x);
        assert!(x <= 6.);
    })
}
