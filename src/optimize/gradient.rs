//! Functions for computing derivatives of functions.
use autodiff::*;

/// Calculates the derivative of a 1D function.
pub fn der<G>(f: G, x: f64) -> f64
where
    G: Fn(F1) -> F1,
{
    diff(f, x)
}

/// Calculates the partial derivative of a function `f` with respect to the `i`th variable, where
/// `x` are the variables.
pub fn partial<G>(f: G, x: &[f64], i: usize) -> f64
where
    G: Fn(&[F1]) -> F1,
{
    let v = x
        .into_iter()
        .enumerate()
        .map(|(idx, val)| if idx == i { F::var(*val) } else { F::cst(*val) })
        .collect::<Vec<_>>();

    f(&v).deriv()
}

pub fn partials<G>(f: G, x: &[f64], dims: &[usize]) -> Vec<f64>
where
    G: Fn(&[F1]) -> F1 + Copy,
{
    let mut g: Vec<f64> = Vec::with_capacity(dims.len());
    for i in dims.iter() {
        g.push(partial(f, x, *i));
    }

    g
}

pub fn gradient<G>(f: G, x: &[f64]) -> Vec<f64>
where
    G: Fn(&[F1]) -> F1,
{
    grad(f, x)
}
