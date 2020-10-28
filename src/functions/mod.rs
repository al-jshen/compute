//! Various mathematical functions commonly used in statistics.

use std::f64::consts::PI;

const G: f64 = 4.7421875 + 1.;

/// Coefficients from [here](https://my.fit.edu/~gabdo/gamma.txt).
const GAMMA_COEFFS: [f64; 14] = [
    57.156235665862923517,
    -59.597960355475491248,
    14.136097974741747174,
    -0.49191381609762019978,
    0.33994649984811888699e-4,
    0.46523628927048575665e-4,
    -0.98374475304879564677e-4,
    0.15808870322491248884e-3,
    -0.21026444172410488319e-3,
    0.21743961811521264320e-3,
    -0.16431810653676389022e-3,
    0.84418223983852743293e-4,
    -0.26190838401581408670e-4,
    0.36899182659531622704e-5,
];

/// Calculates the [Gamma function](https://en.wikipedia.org/wiki/Gamma_function) using the [Lanczos
/// approximation](https://en.wikipedia.org/wiki/Lanczos_approximation). It obeys the equation
/// `gamma(x+1) = gamma(x) * x`. This approximation uses the reflection formula to extend the
/// calculation to the entire complex plane.
///
/// ```
/// use approx_eq::assert_approx_eq;
/// use compute::functions::gamma;
/// assert_approx_eq!(gamma(0.1), 9.513507698668731836292487);
/// assert_approx_eq!(gamma(0.5), 1.7724538509551602798167);
/// assert_approx_eq!(gamma(6.), 120.);
/// assert_approx_eq!(gamma(20.), 121645100408832000.);
/// assert_approx_eq!(gamma(-0.5), -3.54490770181103205459);
/// ```
pub fn gamma(z: f64) -> f64 {
    if z < 0.5 {
        PI / ((PI * z).sin() * gamma(1. - z))
    } else {
        let mut x = 0.99999999999999709182;
        for (idx, val) in GAMMA_COEFFS.iter().enumerate() {
            x += val / ((z - 1.) + (idx as f64) + 1.);
        }
        let t = (z - 1.) + G - 0.5;
        ((2. * PI) as f64).sqrt() * t.powf((z - 1.) + 0.5) * (-t).exp() * x
    }
}

/// Calculates the [beta function](https://en.wikipedia.org/wiki/Beta_function) using the
/// relationship between the beta function and the gamma function.
///
/// ```
/// use approx_eq::assert_approx_eq;
/// use compute::functions::beta;
/// use std::f64::consts::PI;
/// assert_approx_eq!(beta(1., 3.12345), 1. / 3.12345);
/// assert_approx_eq!(beta(2.1313, 1. - 2.1313), PI / (PI * 2.1313).sin());
/// assert_approx_eq!(
///     beta(7.2, 0.23) * beta(7.2 + 0.23, 1. - 0.23),
///     PI / (7.2 * (PI * 0.23).sin())
/// );
/// ```
pub fn beta(a: f64, b: f64) -> f64 {
    gamma(a) * gamma(b) / gamma(a + b)
}

/// Calculates the [digamma function](https://en.wikipedia.org/wiki/Digamma_function), which is the
/// logarithmic derivative of the gamma function. It obeys the equation `digamma(x+1) = digamma(x)
/// + 1/x`. The approximation works better for large values. If the value is small, this function
/// will shift it up using the digamma recurrence relation.
///
/// ```
/// use approx_eq::assert_approx_eq;
/// use compute::functions::digamma;
/// assert_approx_eq!(digamma(21. + 1.), digamma(21.) + 1. / 21.);
/// assert_approx_eq!(digamma(2. + 1.), digamma(2.) + 1. / 2.);
/// assert_approx_eq!(digamma(0.5), -1.96351002602142347944097633);
/// assert_approx_eq!(digamma(-0.5), 0.036489973978576520559023667);
/// assert_approx_eq!(digamma(1.), -0.57721566490153286060651209);
/// ```
pub fn digamma(x: f64) -> f64 {
    if x < 6. {
        digamma(x + 1.) - 1. / x
    } else {
        x.ln() - 1. / (2. * x) - 1. / (12. * x.powi(2)) + 1. / (120. * x.powi(4))
            - 1. / (252. * x.powi(6))
            + 1. / (240. * x.powi(8))
            - 5. / (660. * x.powi(10))
            + 691. / (32760. * x.powi(12))
            - 1. / (12. * x.powi(14))
    }
}

/// Calculates the standard [logistic function](https://en.wikipedia.org/wiki/Logistic_function)
///
/// ```
/// use approx_eq::assert_approx_eq;
/// use compute::functions::logistic;
/// use compute::distributions::{Exponential, Distribution};
/// let d = Exponential::new(5.).sample_iter(100 as usize);
/// d.iter().for_each(|x| {
///     assert_approx_eq!(logistic(*x) + logistic(-*x), 1.);
/// });
/// for i in 0..d.len() {
///     for j in i..d.len() {
///         if d[i] >= d[j] {
///             assert!(logistic(d[i]) >= logistic(d[j]));
///         }
///     }
/// }
/// assert_eq!(logistic(f64::NEG_INFINITY), 0.);
/// assert_eq!(logistic(0.), 0.5);
/// assert_eq!(logistic(f64::INFINITY), 1.);
/// ```
pub fn logistic(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

/// Calculates the [logit function](https://en.wikipedia.org/wiki/Logit)
/// ```
/// use approx_eq::assert_approx_eq;
/// use compute::functions::{logistic, logit};
/// use compute::distributions::{Uniform, Distribution};
/// let d = Uniform::new(0., 1.).sample_iter(100 as usize);
/// d.iter().for_each(|x| {
///     assert_approx_eq!(*x, logistic(logit(*x)));
///     assert_approx_eq!(*x, logit(logistic(*x)));
/// });
/// for i in 0..d.len() {
///     for j in (i+1)..d.len() {
///         assert_approx_eq!(
///             logit(d[i]) - logit(d[j]),
///             ((d[i] / (1. - d[i])) / (d[j] / (1. - d[j]))).ln()
///         );
///     }
/// }
/// assert_eq!(logit(0.), f64::NEG_INFINITY);
/// assert_eq!(logit(0.5), 0.);
/// assert_eq!(logit(1.), f64::INFINITY);
/// ```
pub fn logit(p: f64) -> f64 {
    if p < 0. || p > 1. {
        panic!("p must be in [0, 1]");
    }
    (p / (1. - p)).ln()
}
