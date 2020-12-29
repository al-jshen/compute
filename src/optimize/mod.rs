//! A module for optimization. Used in conjunction with [predict](/compute/predict) to fit models.

mod gradient;
mod loss;
mod optimizers;
mod sim_annealing;

pub use self::gradient::*;
pub use self::loss::*;
pub use self::optimizers::*;
pub use self::sim_annealing::*;
