//! Functions for numerically computing derivatives of functions.

/// Calculates the step size `h` to use to compute the gradient.
fn calc_h(x: f64) -> f64 {
    if x != 0. {
        std::f64::EPSILON.sqrt() * x
    } else {
        std::f64::EPSILON.sqrt()
    }
}

/// Calculates the symmetric difference quotient `(f(x+h) - f(x-h)) / 2h`.
/// See <https://en.wikipedia.org/wiki/Symmetric_derivative>
pub fn sym_der<F>(f: F, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = calc_h(x);
    (f(x + h) - f(x - h)) / (2. * h)
}

/// Calculates the derivative from its mathematical definition.
pub fn der<F>(f: F, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = calc_h(x);
    (f(x + h) - f(x)) / h
}

/// Calculates the partial derivative of a function `f` with respect to the `i`th variable, where
/// `x` are the variables.
pub fn partial<F>(f: F, x: &[f64], i: usize) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let h = calc_h(x[i]);
    let mut xph = x.to_owned();
    xph[i] += h;
    let mut xmh = x.to_owned();
    xmh[i] -= h;
    (f(&xph) - f(&xmh)) / (2. * h)
}

/// Given a function, return its derivative (a function).
pub fn derivative<F>(f: F) -> impl Fn(f64) -> f64 + Copy
where
    F: Fn(f64) -> f64 + Copy,
{
    move |x: f64| sym_der(f, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_symder() {
        assert_approx_eq!(2., sym_der(|x| x.powi(2), 1.));
        assert_approx_eq!(12., sym_der(|x| x.powi(3), 2.));
        assert_approx_eq!(0., sym_der(|_| 5., -2.));
        assert_approx_eq!(5_f64.exp(), sym_der(|x| x.exp(), 5.));
        assert_approx_eq!(0.5_f64.cos(), sym_der(|x| x.sin(), 0.5));
    }

    #[test]
    fn test_der() {
        assert_approx_eq!(2., der(|x| x.powi(2), 1.));
        assert_approx_eq!(12., der(|x| x.powi(3), 2.));
        assert_approx_eq!(0., der(|_| 5., -2.));
        assert_approx_eq!(5_f64.exp(), der(|x| x.exp(), 5.));
        assert_approx_eq!(0.5_f64.cos(), der(|x| x.sin(), 0.5));
    }

    #[test]
    fn test_partial() {
        fn func1(vars: &[f64]) -> f64 {
            vars[0].powi(2) + vars[0] * vars[1] + vars[1].powi(2)
        }

        // partial of x^2 + xy + y^2 at (1, 1)
        assert_approx_eq!(partial(func1, &[1., 1.], 0), 3.); // wrt x
        assert_approx_eq!(partial(func1, &[1., 1.], 1), 3.); // wrt y

        fn func2(vars: &[f64]) -> f64 {
            vars[0].powi(2) * vars[1].powi(3)
        }

        // partial of x^2 * y^3
        assert_approx_eq!(partial(func2, &[5., 1.2], 0), 2. * 5. * 1.2_f64.powi(3));
        assert_approx_eq!(
            partial(func2, &[0.1, -2.], 1),
            3. * (-2. as f64).powi(2) * (0.1_f64).powi(2)
        );

        fn func3(vars: &[f64]) -> f64 {
            (vars[0].sin() / vars[1].exp()).powf(vars[1])
        }

        // partial of (sin(x) / e^y)^y with respect to x at (5, 5)
        assert_approx_eq!(partial(func3, &[5., 5.], 0), 1.665507727894749327e-11);

        // partial of a closure
        assert_approx_eq!(partial(|x: &[f64]| x[0] * x[0] + 2., &[0.], 0), 0.);
    }
}
