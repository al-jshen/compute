use crate::distributions::*;
use crate::functions::gamma;

/// Implements the [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) distribution.
#[derive(Debug)]
pub struct Gamma {
    /// Shape parameter α.
    alpha: f64,
    /// Rate parameter β.
    beta: f64,
    normal_gen: Normal,
    uniform_gen: Uniform,
}

impl Gamma {
    /// Create a new Gamma distribution with shape `alpha` and rate `beta`.
    ///
    /// # Errors
    /// Panics if `alpha <= 0` or `beta <= 0`.
    pub fn new(alpha: f64, beta: f64) -> Self {
        if alpha <= 0. || beta <= 0. {
            panic!("Both alpha and beta must be positive.");
        }
        Gamma {
            alpha,
            beta,
            normal_gen: Normal::new(0., 1.),
            uniform_gen: Uniform::new(0., 1.),
        }
    }
    pub fn set_alpha(&mut self, alpha: f64) -> &mut Self {
        if alpha <= 0. {
            panic!("Alpha must be positive.");
        }
        self.alpha = alpha;
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

impl Default for Gamma {
    fn default() -> Self {
        Self::new(1., 1.)
    }
}

impl Distribution for Gamma {
    /// Samples from the given Gamma distribution.
    ///
    /// # Remarks
    /// Uses the algorithm from Marsaglia and Tsang 2000. Applies the squeeze
    /// method and has nearly constant average time for `alpha >= 1`.
    fn sample(&self) -> f64 {
        let d = self.alpha - 1. / 3.;
        loop {
            let (x, v) = loop {
                let x = self.normal_gen.sample();
                let v = (1. + x / (9. * d).sqrt()).powi(3);
                if v > 0. {
                    break (x, v);
                }
            };
            let u = self.uniform_gen.sample();
            if u < 1. - 0.0331 * x.powi(4) {
                return d * v * self.beta;
            }
            if u.ln() < 0.5 * x.powi(2) + d * (1. - v + v.ln()) {
                return d * v * self.beta;
            }
        }
    }
}

impl Continuous for Gamma {
    /// Calculates the probability density function for the given Gamma function at `x`.
    fn pdf(&self, x: f64) -> f64 {
        self.beta.powf(self.alpha) / gamma(self.alpha)
            * x.powf(self.alpha - 1.)
            * (-self.beta * x).exp()
    }
}
