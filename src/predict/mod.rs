//! Various models for data fitting and forecasting.

mod polynomial;
use crate::optimize::Optimizer;

pub trait Predictor {
    fn update(&mut self, params: &[f64]) -> &mut Self;
    fn fit<O>(&mut self, x: &[f64], y: &[f64], optimizer: O) -> &mut Self
    where
        O: Optimizer;
    fn predict(&self, x: &[f64]) -> Vec<f64>;
}

pub use self::polynomial::*;
