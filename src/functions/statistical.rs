//! Various mathematical functions commonly used in statistics.

/// Calculates the standard [logistic function](https://en.wikipedia.org/wiki/Logistic_function)
pub fn logistic(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

/// Calculates the [logit function](https://en.wikipedia.org/wiki/Logit)
pub fn logit(p: f64) -> f64 {
    if !(0. ..=1.).contains(&p) {
        panic!("p must be in [0, 1]");
    }
    (p / (1. - p)).ln()
}

/// Calculates the one-parameter Box-Cox transformation with some power parameter `lambda`.
pub fn boxcox(x: f64, lambda: f64) -> f64 {
    assert!(x > 0., "x must be positive");
    if lambda == 0. {
        x.ln()
    } else {
        (x.powf(lambda) - 1.) / lambda
    }
}

/// Calculates the two-parameter Box-Cox transformation with some power parameter `lambda` and some
/// shift parameter `alpha`.
pub fn boxcox_shifted(x: f64, lambda: f64, alpha: f64) -> f64 {
    assert!(x > alpha, "x must larger than alpha");
    if lambda == 0. {
        (x + alpha).ln()
    } else {
        ((x + alpha).powf(lambda) - 1.) / lambda
    }
}

/// Calculates the softmax (the normalized exponential) function, which is a generalization of the
/// logistic function to multiple dimensions.
///
/// Takes in a vector of real numbers and normalizes it to a probability distribution such that
/// each of the components are in the interval (0, 1) and the components add up to 1. Larger input
/// components correspond to larger probabilities.
pub fn softmax(x: &[f64]) -> Vec<f64> {
    let sum_exp: f64 = x.iter().map(|i| i.exp()).sum();
    x.iter().map(|i| i.exp() / sum_exp).collect()
}

const ERF_P: f64 = 0.3275911;
const ERF_A1: f64 = 0.254829592;
const ERF_A2: f64 = -0.284496736;
const ERF_A3: f64 = 1.421413741;
const ERF_A4: f64 = -1.453152027;
const ERF_A5: f64 = 1.061405429;

/// Calculates the [error function](https://en.wikipedia.org/wiki/Error_function) erf(x).
///
/// # Remarks
/// Uses Equation 7.1.26 in Stegun in combination with Horner's Rule.
pub fn erf(x: f64) -> f64 {
    if x >= 0. {
        let t = 1. / (1. + ERF_P * x);
        1. - (((((ERF_A5 * t + ERF_A4) * t) + ERF_A3) * t + ERF_A2) * t + ERF_A1)
            * t
            * (-x * x).exp()
    } else {
        // erf is an odd function
        -erf(-x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{Distribution1D, Exponential, Uniform};
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_logistic() {
        let d = Exponential::new(5.).sample_n(100 as usize);
        d.iter().for_each(|x| {
            assert_approx_eq!(logistic(*x) + logistic(-*x), 1.);
        });
        for i in 0..d.len() {
            for j in i..d.len() {
                if d[i] >= d[j] {
                    assert!(logistic(d[i]) >= logistic(d[j]));
                }
            }
        }
        assert_eq!(logistic(f64::NEG_INFINITY), 0.);
        assert_eq!(logistic(0.), 0.5);
        assert_eq!(logistic(f64::INFINITY), 1.);
    }

    #[test]
    fn test_logit() {
        let d = Uniform::new(0., 1.).sample_n(100 as usize);
        d.iter().for_each(|x| {
            assert_approx_eq!(*x, logistic(logit(*x)));
            assert_approx_eq!(*x, logit(logistic(*x)));
        });
        for i in 0..d.len() {
            for j in (i + 1)..d.len() {
                assert_approx_eq!(
                    logit(d[i]) - logit(d[j]),
                    ((d[i] / (1. - d[i])) / (d[j] / (1. - d[j]))).ln()
                );
            }
        }
        assert_eq!(logit(0.), f64::NEG_INFINITY);
        assert_eq!(logit(0.5), 0.);
        assert_eq!(logit(1.), f64::INFINITY);
    }

    #[test]
    fn test_softmax() {
        let orig = vec![1., 2., 3., 4., 1., 2., 3.];
        let tfm = vec![
            0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813,
        ];
        let smv = softmax(&orig);
        for i in 0..smv.len() {
            assert_approx_eq!(smv[i], tfm[i]);
        }
        assert_approx_eq!(smv.iter().sum(), 1.);
    }

    #[test]
    fn test_erf() {
        assert_approx_eq!(erf(0.), 0., 1e-5);
        assert_approx_eq!(erf(0.02), 0.022564575, 1e-5);
        assert_approx_eq!(erf(0.04), 0.045111106, 1e-5);
        assert_approx_eq!(erf(0.06), 0.067621594, 1e-5);
        assert_approx_eq!(erf(0.08), 0.090078126, 1e-5);
        assert_approx_eq!(erf(0.1), 0.112462916, 1e-5);
        assert_approx_eq!(erf(0.2), 0.222702589, 1e-5);
        assert_approx_eq!(erf(0.3), 0.328626759, 1e-5);
        assert_approx_eq!(erf(0.4), 0.428392355, 1e-5);
        assert_approx_eq!(erf(0.5), 0.520499878, 1e-5);
        assert_approx_eq!(erf(0.6), 0.603856091, 1e-5);
        assert_approx_eq!(erf(0.7), 0.677801194, 1e-5);
        assert_approx_eq!(erf(0.8), 0.742100965, 1e-5);
        assert_approx_eq!(erf(0.9), 0.796908212, 1e-5);
        assert_approx_eq!(erf(1.), 0.842700793, 1e-5);
        assert_approx_eq!(erf(1.1), 0.88020507, 1e-5);
        assert_approx_eq!(erf(1.2), 0.910313978, 1e-5);
        assert_approx_eq!(erf(1.3), 0.934007945, 1e-5);
        assert_approx_eq!(erf(1.4), 0.95228512, 1e-5);
        assert_approx_eq!(erf(1.5), 0.966105146, 1e-5);
        assert_approx_eq!(erf(1.6), 0.976348383, 1e-5);
        assert_approx_eq!(erf(1.7), 0.983790459, 1e-5);
        assert_approx_eq!(erf(1.8), 0.989090502, 1e-5);
        assert_approx_eq!(erf(1.9), 0.992790429, 1e-5);
        assert_approx_eq!(erf(2.), 0.995322265, 1e-5);
        assert_approx_eq!(erf(2.1), 0.997020533, 1e-5);
        assert_approx_eq!(erf(2.2), 0.998137154, 1e-5);
        assert_approx_eq!(erf(2.3), 0.998856823, 1e-5);
        assert_approx_eq!(erf(2.4), 0.999311486, 1e-5);
        assert_approx_eq!(erf(2.5), 0.999593048, 1e-5);
        assert_approx_eq!(erf(3.), 0.99997791, 1e-5);
        assert_approx_eq!(erf(3.5), 0.999999257, 1e-5);
    }
}
