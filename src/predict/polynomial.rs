use super::Predictor;
use crate::optimize::*;

/// Implements a polynomial regressor with coefficients `coeffs`.
///
/// In the case of a two coefficients, this reduces to simple linear regression,
/// where the first parameter is the intercept, and the second is the slope.
pub struct PolynomialRegressor {
    coeffs: Vec<f64>,
}

impl PolynomialRegressor {
    /// Create a new polynomial regressor with coefficients `coeffs`, where the first number is
    /// the 0th order coefficient, the second is the 1st order coefficient, and so on. Accepts an
    /// arbitrary number of coefficients.
    pub fn new(coeffs: &[f64]) -> Self {
        PolynomialRegressor {
            coeffs: coeffs.to_owned(),
        }
    }
    /// Prints the coefficients of the polynomial regressor.
    pub fn get_coeffs(&self) -> Vec<f64> {
        self.coeffs.clone()
    }
}

impl Predictor for PolynomialRegressor {
    /// Update the coefficients of the polynomial regressor.
    fn update(&mut self, params: &[f64]) -> &mut Self {
        self.coeffs = params.to_owned();
        self
    }
    /// Fit the polynomial regressor to some observed data `y` given some explanatory variables `x`
    /// using the given optimizer. See [Optimizer](/compute/optimize/trait.Optimizer.html).
    fn fit<O>(&mut self, x: &[f64], y: &[f64], mut optimizer: O) -> &mut Self
    where
        O: Optimizer,
    {
        // let mut optimizer = Adam::new(0.01, 0.99, 0.999, 1e-8, &self.coeffs);
        self.coeffs = optimizer.optimize(
            |evalat: &[f64], dim: usize| {
                partial(|params: &[f64]| mse(&predict(params, &x), &y), evalat, dim)
            },
            self.get_coeffs(),
            1e6 as usize,
        );
        self
    }
    /// Returns c0 + c[1] * x + c[2] * x^2 ... + cn + x^n, where c[i] are the coefficients of the
    /// polynomial regressor, and `x` is some vector of explanatory variables.
    fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|val| {
                (0..self.coeffs.len())
                    .into_iter()
                    .map(|ith| self.coeffs[ith] * val.powi(ith as i32))
                    .sum::<f64>()
            })
            .collect::<Vec<_>>()
    }
}

fn predict(coeffs: &[f64], x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|val| {
            (0..coeffs.len())
                .into_iter()
                .map(|ith| coeffs[ith] * val.powi(ith as i32))
                .sum::<f64>()
        })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slr() {
        let x = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let y = vec![5., 7., 9., 11., 13., 15., 17., 19., 21., 23.];
        let mut slr = PolynomialRegressor::new(&[5., 2.]);
        assert_eq!(slr.predict(&x), y);
        slr.update(&[0., 1.]);
        assert_eq!(slr.predict(&x), x);
    }
}
