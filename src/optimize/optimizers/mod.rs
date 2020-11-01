//! Various optimization algorithms (eg. Adam, SGD).

mod adam;

pub trait Optimizer {
    fn optimize<F>(&mut self, grad_fn: F, params: Vec<f64>) -> Vec<f64>
    where
        F: Fn(&[f64], usize) -> f64;
}

pub use self::adam::*;
