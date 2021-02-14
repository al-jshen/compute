//! Various optimization algorithms (eg. Adam, SGD, Levenberg-Marquardt).

use autodiff::F1;

mod adam;
mod lm;

pub trait Optimizer {
    fn optimize<F>(
        &self,
        xs: &[f64],
        ys: &[f64],
        f: F,
        parameters: &[f64],
        maxsteps: usize,
    ) -> Vec<f64>
    where
        F: Fn(&[F1]) -> F1 + Copy;
    fn grad_fn_type(&self) -> GradFn;
}

#[derive(Debug, Clone, Copy)]
pub enum GradFn {
    Predictive,
    Residual,
}

pub use self::adam::*;
pub use self::lm::*;
