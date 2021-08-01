//! Functions for computing derivatives of functions.
use crate::linalg::Vector;
use autodiff::*;

pub trait DiffFn {
    fn eval(&self, params: &[F1], data: &[&[f64]]) -> (f64, Vector);
}

impl<T> DiffFn for T
where
    T: Fn(&[F1], &[&[f64]]) -> F1,
{
    fn eval(&self, params: &[F1], data: &[&[f64]]) -> (f64, Vector) {
        let mut pars = params.iter().map(|&x| F1::cst(x)).collect::<Vec<_>>();
        let mut val: f64 = 0.;
        let mut derivs = Vector::empty_n(pars.len());
        for i in 0..pars.len() {
            pars[i] = params[i];
            let res = self(&pars, data);
            val = res.x;
            derivs[i] = res.dx;
            pars[i] = F1::cst(params[i]);
        }
        (val, derivs)
    }
}
