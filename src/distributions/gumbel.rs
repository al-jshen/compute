use crate::distributions::*;

const EULER_MASCHERONI: f64 = 0.577215664901532860606512090082402431042159335939923598805767234884867726777664670936947063291746749;
const PISQ6: f64 = std::f64::consts::PI;

/// Implements the [Gumbel](https://en.wikipedia.org/wiki/Gumbel_distribution) distribution.
#[derive(Debug, Clone, Copy)]
pub struct Gumbel {
    /// Shape parameter μ.
    mu: f64,
    /// Rate parameter β.
    beta: f64,
    uniform_gen: Uniform,
}

impl Gumbel {
    /// Create a new Gumbel distribution with location `mu` and scale `beta`.
    ///
    /// # Errors
    /// Panics if `beta <= 0`.
    pub fn new(mu: f64, beta: f64) -> Self {
        if beta <= 0. {
            panic!("Beta must be positive.");
        }
        Gumbel {
            mu,
            beta,
            uniform_gen: Uniform::new(0., 1.),
        }
    }
    pub fn set_mu(&mut self, mu: f64) -> &mut Self {
        self.mu = mu;
        self
    }
    pub fn set_beta(&mut self, beta: f64) -> &mut Self {
        if beta <= 0. {
            panic!("Beta must be positive.");
        }
        self.beta = beta;
        self
    }
}

impl Default for Gumbel {
    fn default() -> Self {
        Self::new(0., 1.)
    }
}

impl Distribution for Gumbel {
    type Output = f64;
    /// Samples from the given Gumbel distribution.
    fn sample(&self) -> Self::Output {
        self.mu - self.beta * (-self.uniform_gen.sample().ln()).ln()
    }
}

impl Distribution1D for Gumbel {
    fn update(&mut self, params: &[f64]) {
        self.set_mu(params[0]).set_beta(params[1]);
    }
}

impl Continuous for Gumbel {
    type PDFType = f64;
    /// Calculates the probability density function for the given Gumbel function at `x`.
    fn pdf(&self, x: f64) -> Self::PDFType {
        let z = (x - self.mu) / self.beta;
        1. / self.beta * (-(z + (-z).exp())).exp()
    }
}

impl Mean for Gumbel {
    type MeanType = f64;
    /// Calculates the mean of the given Gumbel distribution.
    fn mean(&self) -> Self::MeanType {
        self.mu + self.beta * EULER_MASCHERONI
    }
}

impl Variance for Gumbel {
    type VarianceType = f64;
    /// Calculates the variance of the given Gumbel distribution.
    fn var(&self) -> Self::VarianceType {
        PISQ6 * self.beta.powi(2)
    }
}
