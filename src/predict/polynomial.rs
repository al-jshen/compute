use super::Predictor;
use crate::linalg::*;
use crate::optimize::{loss::mse, num_gradient::partial, optimizers::Optimizer};

/// Implements a polynomial regressor with coefficients `coeffs`.
///
/// In the case of a two coefficients, this reduces to simple linear regression,
/// where the first parameter is the intercept, and the second is the slope.
#[derive(Debug)]
pub struct PolynomialRegressor {
    coeffs: Vec<f64>,
}

impl PolynomialRegressor {
    /// Create a new polynomial regressor with degree `deg` (e.g., deg = 1 is a linear model).
    pub fn new(deg: usize) -> Self {
        PolynomialRegressor {
            coeffs: vec![0.; deg + 1],
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

impl PolynomialRegressor {
    /// Fit the polynomial regressor to some observed data `y` given some explanatory variables
    /// `x`. Uses least squares fitting.
    pub fn fit(&mut self, x: &[f64], y: &[f64]) -> &mut Self {
        assert_eq!(x.len(), y.len());
        let xv = vandermonde(x, self.coeffs.len());
        let xtx = xtx(&xv, x.len());
        let xtxinv = invert_matrix(&xtx);
        let xty = matmul(&xv, y, x.len(), y.len(), true, false);
        let coeffs = matmul(
            &xtxinv,
            &xty,
            self.coeffs.len(),
            self.coeffs.len(),
            false,
            false,
        );
        self.update(&coeffs)
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
    use crate::distributions::{Distribution, Normal};
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_slr() {
        let x = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let y = vec![5., 7., 9., 11., 13., 15., 17., 19., 21., 23.];
        let mut slr = PolynomialRegressor::new(1);
        slr.update(&[5., 2.]);
        assert_eq!(slr.predict(&x), y);
        slr.update(&[0., 1.]);
        assert_eq!(slr.predict(&x), x);
    }

    #[test]
    fn test_fits() {
        let x: Vec<f64> = (0..250).into_iter().map(|x| x as f64 / 10.).collect();
        let yv: Vec<f64> = (&x).into_iter().map(|v| 5. + 2. * v).collect();
        let scatter = Normal::new(0., 5.);
        let y: Vec<f64> = (&yv).into_iter().map(|v| v + scatter.sample()).collect();

        let mut p = PolynomialRegressor::new(1);
        p.fit(&x, &y);
        let coeffs1 = p.get_coeffs();

        p.update(&[2., 2.]);
        let o = crate::optimize::optimizers::Adam::default();
        p.fit_with_optimizer(&x.to_vec(), &y.to_vec(), o);
        let coeffs2 = p.get_coeffs();

        println!("{:?}", coeffs1);
        println!("{:?}", coeffs2);
        assert_approx_eq!(coeffs1[0], coeffs2[0], 1e-4);
        assert_approx_eq!(coeffs1[1], coeffs2[1], 1e-4);
    }
}
