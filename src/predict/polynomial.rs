use super::{ClosedPredictor, Predictor};
use crate::optimize::*;
use ndarray::{stack, Array, Axis, Ix1, Ix2};
use ndarray_linalg::Inverse;

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
    fn fit_with_optimizer<O>(&mut self, x: &[f64], y: &[f64], mut optimizer: O) -> &mut Self
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
    /// Returns `c0 + c[1] * x + c[2] * x^2 ... + cn + x^n`, where `c[i]` are the coefficients of the
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

impl ClosedPredictor for PolynomialRegressor {
    /// Fit the polynomial regressor to some observed data `y` given some explanatory variables
    /// `x`. Uses least squares fitting.
    fn fit(&mut self, x: &Array<f64, Ix1>, y: &Array<f64, Ix1>) -> &mut Self {
        let xd: Array<f64, Ix2> = design(x.clone());
        let coeffs: Array<f64, Ix1> = (xd.t().dot(&xd)).inv().unwrap().dot(&xd.t()).dot(y);
        self.coeffs = coeffs.to_vec();
        self
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

fn design(x: Array<f64, Ix1>) -> Array<f64, Ix2> {
    let d = Array::ones((x.len(), 1));
    stack![Axis(1), d, x.insert_axis(Axis(1))]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{Distribution, Normal};
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_slr() {
        let x = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let y = vec![5., 7., 9., 11., 13., 15., 17., 19., 21., 23.];
        let mut slr = PolynomialRegressor::new(&[5., 2.]);
        assert_eq!(slr.predict(&x), y);
        slr.update(&[0., 1.]);
        assert_eq!(slr.predict(&x), x);
    }

    #[test]
    fn test_fits() {
        let x: Array<f64, Ix1> = Array::range(0., 50., 0.1);
        let yv: Array<f64, Ix1> = 5. + 2. * &x;
        let y = &yv.mapv(|x| x + Normal::new(0., 10.).sample());

        let mut p = PolynomialRegressor::new(&[2., 2.]);
        p.fit(&x, &y);
        let coeffs1 = p.get_coeffs();

        p.update(&[2., 2.]);
        let o = Adam::default();
        p.fit_with_optimizer(&x.to_vec(), &y.to_vec(), o);
        let coeffs2 = p.get_coeffs();

        for (i, j) in coeffs1.iter().zip(coeffs2.iter()) {
            assert_approx_eq!(*i, *j, 1e-4);
        }
    }
}
