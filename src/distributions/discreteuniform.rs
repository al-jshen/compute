use crate::distributions::*;

/// Implements the [discrete uniform distribution](https://en.wikipedia.org/wiki/Discrete_uniform_distribution).
#[derive(Debug, Clone, Copy)]
pub struct DiscreteUniform {
    /// Lower bound for the discrete uniform distribution.
    lower: i64,
    /// Upper bound for the discrete uniform distribution.
    upper: i64,
}

impl DiscreteUniform {
    /// Create a new discrete uniform distribution with lower bound `lower` and upper bound `upper` (inclusive on both ends).
    ///
    /// # Errors
    /// Panics if `lower > upper`.
    pub fn new(lower: i64, upper: i64) -> Self {
        if lower > upper {
            panic!("`Upper` must be larger than `lower`.");
        }
        DiscreteUniform { lower, upper }
    }
    pub fn set_lower(&mut self, lower: i64) -> &mut Self {
        if lower > self.upper {
            panic!("Upper must be larger than lower.")
        }
        self.lower = lower;
        self
    }
    pub fn set_upper(&mut self, upper: i64) -> &mut Self {
        if self.lower > upper {
            panic!("Upper must be larger than lower.")
        }
        self.upper = upper;
        self
    }
}

impl Default for DiscreteUniform {
    fn default() -> Self {
        Self::new(0, 1)
    }
}

impl Distribution for DiscreteUniform {
    type Output = f64;
    /// Samples from the given discrete uniform distribution.
    fn sample(&self) -> f64 {
        alea::i64_in_range(self.lower, self.upper) as f64
    }
}

impl Distribution1D for DiscreteUniform {
    fn update(&mut self, params: &[f64]) {
        self.set_lower(params[0] as i64).set_upper(params[1] as i64);
    }
}

impl Discrete for DiscreteUniform {
    /// Calculates the [probability mass
    /// function](https://en.wikipedia.org/wiki/Probability_mass_function) for the given discrete uniform
    /// distribution at `x`.
    ///
    /// # Remarks
    ///
    /// Returns `0.` if `x` is not in `[lower, upper]`
    fn pmf(&self, x: i64) -> f64 {
        if x < self.lower || x > self.upper {
            0.
        } else {
            1. / (self.upper - self.lower + 1) as f64
        }
    }
}

impl Mean for DiscreteUniform {
    type MeanType = f64;
    /// Calculates the mean, which for a Uniform(a, b) distribution is given by `(a + b) / 2`.
    fn mean(&self) -> f64 {
        ((self.lower + self.upper) / 2) as f64
    }
}

impl Variance for DiscreteUniform {
    type VarianceType = f64;
    /// Calculates the variance of the given Uniform distribution.
    fn var(&self) -> f64 {
        (((self.upper - self.lower + 1) as f64).powi(2) - 1.) / 12.
    }
}

#[test]
fn inrange() {
    let u = self::DiscreteUniform::new(-2, 6);
    let samples = u.sample_n(100);
    samples.into_iter().for_each(|x| {
        assert!(-2. <= x);
        assert!(x <= 6.);
    })
}
