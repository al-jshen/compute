//! Various models for data fitting and forecasting.

mod logistic;
mod polynomial;
use crate::optimize::Optimizer;

/// A predictor for which the parameters can be optimized and updated.
pub trait Predictor {
    fn update(&mut self, params: &[f64]) -> &mut Self;
    fn fit_with_optimizer<O>(&mut self, x: &[f64], y: &[f64], optimizer: O) -> &mut Self
    where
        O: Optimizer;
    fn predict(&self, x: &[f64]) -> Vec<f64>;
}

pub use self::logistic::*;
pub use self::polynomial::*;
