use super::Predictor;
use crate::optimize::*;

/// Implements a logistic regressor.
///
/// ```rust
/// use compute::predict::{LogisticRegressor, Predictor};
/// use compute::optimize::Adam;
///
/// let x = vec![
///     0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00,
///     4.25, 4.50, 4.75, 5.00, 5.50,
/// ];
/// let y = vec![
///     0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1.,
/// ];
/// assert_eq!(x.len(), y.len());
/// let mut clf = LogisticRegressor::default();
/// let optim = Adam::new(1e-3, 0.9, 0.99, 1e-8);
/// clf.fit(&x, &y, optim);
/// println!("{:?}", clf.get_coeffs());
/// println!("{:?}", clf.predict(&[1., 2., 3., 4., 5.]));
/// ```
pub struct LogisticRegressor {
    coeffs: Vec<f64>,
}

impl LogisticRegressor {
    /// Create a new logistic regressor with coefficients `coeffs`, where the first number is
    /// the y-intercept, and the second is the coefficient on some arbitrary predictor variables.
    pub fn new(coeffs: &[f64]) -> Self {
        LogisticRegressor {
            coeffs: coeffs.to_owned(),
        }
    }
    /// Prints the coefficients of the polynomial regressor.
    pub fn get_coeffs(&self) -> Vec<f64> {
        self.coeffs.clone()
    }
}

impl Default for LogisticRegressor {
    fn default() -> Self {
        LogisticRegressor::new(&[0., 1.])
    }
}

impl Predictor for LogisticRegressor {
    /// Update the coefficients of the logistic regressor.
    fn update(&mut self, params: &[f64]) -> &mut Self {
        self.coeffs = params.to_owned();
        self
    }
    /// Fit the logistic regressor to some observed data `y` with classes 0 and 1, given some
    /// explanatory variables `x` using the given optimizer.
    /// See [Optimizer](/compute/optimize/trait.Optimizer.html).
    fn fit<O>(&mut self, x: &[f64], y: &[f64], mut optimizer: O) -> &mut Self
    where
        O: Optimizer,
    {
        self.coeffs = optimizer.optimize(
            |evalat: &[f64], dim: usize| {
                partial(|params: &[f64]| mse(&predict(params, &x), &y), evalat, dim)
            },
            self.get_coeffs(),
        );
        self
    }
    /// Returns 1 / (1 + exp(-(b0 + b1 * x))), where `b0` is the intercept of the model, `x` is some
    /// vector of explanatory variables, and `b1` is the coefficient on those variables.
    fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|val| 1. / (1. + (-self.coeffs[0] - self.coeffs[1] * val).exp()))
            .collect::<Vec<_>>()
    }
}

fn predict(coeffs: &[f64], x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|val| 1. / (1. + (-coeffs[0] - coeffs[1] * val).exp()))
        .collect::<Vec<_>>()
}
