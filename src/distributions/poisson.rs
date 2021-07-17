use crate::distributions::*;
use crate::functions::gamma;

/// Implements the [Poisson](https://en.wikipedia.org/wiki/https://en.wikipedia.org/wiki/Poisson_distribution)
/// distribution.
#[derive(Debug, Clone, Copy)]
pub struct Poisson {
    /// Rate parameter for the Poisson distribution.
    lambda: f64,
}

impl Poisson {
    /// Create a new Poisson distribution with rate parameter `lambda`.
    ///
    /// # Errors
    /// Panics if `lambda <= 0.0`.
    pub fn new(lambda: f64) -> Self {
        if lambda <= 0. {
            panic!("`Lambda` must be positive.");
        }
        Poisson { lambda }
    }
    pub fn set_lambda(&mut self, lambda: f64) -> &mut Self {
        if lambda <= 0. {
            panic!("`Lambda` must be positive.")
        }
        self.lambda = lambda;
        self
    }
}

impl Default for Poisson {
    fn default() -> Self {
        Self::new(1.)
    }
}

impl Distribution for Poisson {
    type Output = f64;
    /// Samples from the given Poisson distribution. For `lambda < 10.0`, this is done with the direct (multiplication) method,
    /// and for `lambda >= 10.0`, this is done the PTRS transformed rejection method from [Hoermann](https://doi.org/10.1016/0167-6687(93)90997-4).
    fn sample(&self) -> f64 {
        if self.lambda < 10. {
            sample_mult(self.lambda)
        } else {
            sample_ptrs(self.lambda)
        }
    }
}

impl Distribution1D for Poisson {
    fn update(&mut self, params: &[f64]) {
        self.set_lambda(params[0]);
    }
}

impl Discrete for Poisson {
    /// Calculates the [probability mass
    /// function](https://en.wikipedia.org/wiki/Probability_mass_function) for the given Poisson
    /// distribution at `k`.
    ///
    fn pmf(&self, k: i64) -> f64 {
        if k < 0 {
            0.
        } else {
            self.lambda.powi(k as i32) * (-self.lambda).exp() / gamma(k as f64)
        }
    }
}

impl Mean for Poisson {
    type MeanType = f64;
    /// Calculates the mean, which is given by the rate parameter.
    fn mean(&self) -> f64 {
        self.lambda
    }
}

impl Variance for Poisson {
    type VarianceType = f64;
    /// Calculates the variance, which is given by the rate parameter.
    fn var(&self) -> f64 {
        self.lambda
    }
}

fn sample_mult(lambda: f64) -> f64 {
    let limit: f64 = (-lambda).exp();
    let mut count = 0.;
    let mut product: f64 = alea::f64();
    while product > limit {
        count += 1.;
        product *= alea::f64();
    }
    count
}

#[allow(non_snake_case)]
fn sample_ptrs(lam: f64) -> f64 {
    let slam = lam.sqrt();
    let loglam = lam.ln();
    let b = 0.931 + 2.53 * slam;
    let a = -0.059 + 0.02483 * b;
    let invalpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.);

    loop {
        let U = alea::f64() - 0.5;
        let V = alea::f64();
        let us = 0.5 - U.abs();
        let k = f64::floor((2. * a / us + b) * U + lam + 0.43);
        if (us >= 0.07) && (V <= vr) {
            return k;
        }
        if (k < 0.) || (us < 0.013) && (V > us) {
            continue;
        }
        if (V.ln() + invalpha.ln() - (a / (us * us) + b).ln())
            <= (-lam + k * loglam - gamma(k + 1.).ln())
        {
            return k;
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
        let data5 = self::Poisson::new(5.).sample_n(1e6 as usize);
        let mean5 = mean(&data5);
        let var5 = var(&data5);
        assert_approx_eq!(mean5, 5., 1e-2);
        assert_approx_eq!(var5, 5., 1e-2);

        let data42 = self::Poisson::new(42.).sample_n(1e6 as usize);
        let mean42 = mean(&data42);
        let var42 = var(&data42);
        assert_approx_eq!(mean42, 42., 1e-2);
        assert_approx_eq!(var42, 42., 1e-2);
    }
}
