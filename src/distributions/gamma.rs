use crate::distributions::*;

/// Implements the [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) distribution.
pub struct Gamma {
    /// Shape parameter α.
    alpha: f64,
    /// Rate parameter β.
    beta: f64,
}

impl Gamma {
    /// Create a new Gamma distribution with shape `alpha` and rate `beta`.
    ///
    /// # Errors
    /// Panics if `alpha <= 0` or `beta <= 0`.
    pub fn new(alpha: f64, beta: f64) -> Self {
        Gamma { alpha, beta }
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
        let normal_gen = Normal::new(0., 1.);
        let unif_gen = Uniform::new(0., 1.);
        loop {
            let (x, v) = loop {
                let x = normal_gen.sample();
                let v = (1. + x / (9. * d).sqrt()).powi(3);
                if v > 0. {
                    break (x, v);
                }
            };
            let u = unif_gen.sample();
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
    fn pdf(&self, _: f64) -> f64 {
        1.
    }
}
