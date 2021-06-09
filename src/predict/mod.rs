//! Various statistical models for data fitting and prediction.

mod glms;
mod gps;
mod polynomial;
// use crate::optimize::optimizers::Optimizer;

// /// A predictor for which the parameters can be optimized and updated.
// pub trait Predictor {
//     fn update(&mut self, params: &[f64]) -> &mut Self;
//     // fn fit_with_optimizer<O>(
//     //     &mut self,
//     //     x: &[f64],
//     //     y: &[f64],
//     //     optimizer: O,
//     //     maxsteps: usize,
//     // ) -> &mut Self
//     // where
//     //     O: Optimizer;
//     fn predict(&self, x: &[f64]) -> Vec<f64>;
// }

pub use self::glms::*;
pub use self::gps::*;
pub use self::polynomial::*;
