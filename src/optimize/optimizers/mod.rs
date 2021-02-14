//! Various optimization algorithms (eg. Adam, SGD).

mod adam;
// mod lm;

pub trait Optimizer {
    fn optimize<F>(&mut self, grad_fn: F, params: Vec<f64>, steps: usize) -> Vec<f64>
    where
        F: Fn(&[f64], usize) -> f64;
}

pub use self::adam::*;
// pub use self::lm::*;
