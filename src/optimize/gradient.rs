/// Calculates the step size `h` to use to compute the gradient.
fn calc_h(x: f64) -> f64 {
    if x != 0. {
        std::f64::EPSILON.sqrt() * x
    } else {
        std::f64::EPSILON.sqrt()
    }
}

/// Calculates the symmetric difference quotient `(f(x+h) - f(x-h)) / 2h`.
/// See https://en.wikipedia.org/wiki/Symmetric_derivative
///
/// ```rust
/// use compute::optimize::sym_der;
/// use approx_eq::assert_approx_eq;
///
/// assert_approx_eq!(2., sym_der(|x| x.powi(2), 1.));
/// assert_approx_eq!(12., sym_der(|x| x.powi(3), 2.));
/// assert_approx_eq!(0., sym_der(|_| 5., -2.));
/// assert_approx_eq!(5_f64.exp(), sym_der(|x| x.exp(), 5.));
/// assert_approx_eq!(0.5_f64.cos(), sym_der(|x| x.sin(), 0.5));
/// ```
pub fn sym_der<F>(f: F, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = calc_h(x);
    (f(x + h) - f(x - h)) / (2. * h)
}

/// Calculates the derivative from its mathematical definition.
///
/// ```rust
/// use compute::optimize::der;
/// use approx_eq::assert_approx_eq;
///
/// assert_approx_eq!(2., der(|x| x.powi(2), 1.));
/// assert_approx_eq!(12., der(|x| x.powi(3), 2.));
/// assert_approx_eq!(0., der(|_| 5., -2.));
/// assert_approx_eq!(5_f64.exp(), der(|x| x.exp(), 5.));
/// assert_approx_eq!(0.5_f64.cos(), der(|x| x.sin(), 0.5));
/// ```
pub fn der<F>(f: F, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = calc_h(x);
    (f(x + h) - f(x)) / h
}
