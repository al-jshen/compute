//! Various optimization algorithms (eg. Adam, SGD, Levenberg-Marquardt).

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
        F: for<'a> Fn(&[Var<'a>], &[&[f64]]) -> Var<'a>;
}

pub use self::adam::*;
// pub use self::lbfgs::*;
pub use self::lm::*;
pub use self::sgd::*;
// re-export reverse
pub use reverse::*;
