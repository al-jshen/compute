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
        .iter()
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

pub fn gradient<G>(f: G, at: &[f64], params: &[f64]) -> Vec<f64>
where
    G: Fn(&[F1]) -> F1,
{
    let at_grad = at.iter().map(|&x| F1::cst(x));
    let params_grad = params.iter().map(|&x| F1::cst(x));
    let mut vars_grad: Vec<F1> = at_grad.chain(params_grad).collect();
    let mut results = Vec::with_capacity(params.len());
    let offset = at.len();
    for i in offset..(params.len() + offset) {
        vars_grad[i] = F1::var(vars_grad[i]);
        results.push(f(&vars_grad).deriv());
        vars_grad[i] = F1::cst(vars_grad[i]);
    }
    results
}
