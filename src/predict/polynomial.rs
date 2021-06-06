use crate::linalg::*;
use crate::optimize::{optimizers::GradFn, optimizers::Optimizer};
use autodiff::{Float, F1};

/// Implements a [polynomial regressor](https://en.wikipedia.org/wiki/Polynomial_regression).
///
/// In the case of a two coefficients, this reduces to simple linear regression,
/// where the first parameter is the intercept, and the second is the slope.
#[derive(Debug)]
pub struct PolynomialRegressor {
    pub coef: Vec<f64>,
}

impl PolynomialRegressor {
    /// Create a new polynomial regressor with degree `deg` (e.g., deg = 1 is a linear model).
    pub fn new(deg: usize) -> Self {
        PolynomialRegressor {
            coef: vec![0.; deg + 1],
        }
    }

    /// Fit the polynomial regressor to some observed data `y` given some explanatory variables `x`
    /// using the given optimizer. See [Optimizer](/compute/optimize/trait.Optimizer.html).
    pub fn fit_with_optimizer<O>(
        &mut self,
        x: &[f64],
        y: &[f64],
        optimizer: O,
        maxsteps: usize,
    ) -> &mut Self
    where
        O: Optimizer,
    {
        let resid_fn = match optimizer.grad_fn_type() {
            GradFn::Residual => |x: &[F1]| (x[0] - (x[1] * x[3] + x[2])).powi(2),
            GradFn::Predictive => |x: &[F1]| (x[0] * x[2] + x[1]),
        };
        self.coef = optimizer.optimize(x, y, resid_fn, &self.coef, maxsteps);
        self
    }
    /// Update the coefficients of the polynomial regressor.
    fn update(&mut self, params: &[f64]) -> &mut Self {
        self.coef = params.to_owned();
        self
    }

    /// Returns `c0 + c[1] * x + c[2] * x^2 ... + cn + x^n`, where `c[i]` are the coefficients of the
    /// polynomial regressor, and `x` is some vector of explanatory variables. Evaluation is done
    /// using [Horner's method](https://en.wikipedia.org/wiki/Horner%27s_method).
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|val| {
                self.coef
                    .iter()
                    .rev()
                    .fold(0., |acc, coeff| acc * val + coeff)
            })
            .collect::<Vec<_>>()
    }

    /// Fit the polynomial regressor to some observed data `y` given some explanatory variables
    /// `x`. Uses least squares fitting.
    pub fn fit(&mut self, x: &[f64], y: &[f64]) -> &mut Self {
        assert_eq!(x.len(), y.len());
        let xv = vandermonde(x, self.coef.len());
        let xtx = xtx(&xv, x.len());
        let xtxinv = invert_matrix(&xtx);
        let xty = matmul(&xv, y, x.len(), y.len(), true, false);
        let coeffs = matmul(
            &xtxinv,
            &xty,
            self.coef.len(),
            self.coef.len(),
            false,
            false,
        );
        self.update(&coeffs)
    }
}

// fn predict(coeffs: &[f64], x: &[f64]) -> Vec<f64> {
//     x.iter()
//         .map(|val| {
//             (0..coeffs.len())
//                 .into_iter()
//                 .map(|ith| coeffs[ith] * val.powi(ith as i32))
//                 .sum::<f64>()
//         })
//         .collect::<Vec<_>>()
// }

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::distributions::{Distribution, Normal};
    // use crate::optimize::optimizers::LM;
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
        // let scatter = Normal::new(0., 5.);
        // let y: Vec<f64> = (&yv).into_iter().map(|v| v + scatter.sample()).collect();
        let coeffs = [5., 2.];

        let mut p = PolynomialRegressor::new(1);
        p.fit(&x, &yv);
        let coeffs1 = p.coef;

        // p.update(&[2., 2.]);
        // let o = LM::default();
        // p.fit_with_optimizer(&x.to_vec(), &y.to_vec(), o, 50);
        // let coeffs2 = p.get_coeffs();

        for i in 0..2 {
            assert_approx_eq!(coeffs[i], coeffs1[i], 1e-3);
        }
    }
}
