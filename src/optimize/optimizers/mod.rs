//! Various optimization algorithms (eg. Adam, SGD, Levenberg-Marquardt).

use super::DiffFn;
use crate::linalg::Vector;
use autodiff::F1;

mod adam;
// mod lbfgs;
mod lm;
mod sgd;

pub trait Optimizer {
    type Output;
    fn optimize<F>(
        &self,
        f: F,
        parameters: &[f64],
        data: &[&[f64]],
        maxsteps: usize,
    ) -> Self::Output
    where
        F: DiffFn;
}

pub use self::adam::*;
// pub use self::lbfgs::*;
pub use self::lm::*;
pub use self::sgd::*;
