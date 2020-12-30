#![allow(unused_variables)]
use approx_eq::rel_diff;

/// Integrate a function `f` from `a` to `b` using the [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) with `n` partitions.
pub fn trapz<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let dx: f64 = (b - a) / (n as f64);
    dx * ((0..n).map(|k| f(a + k as f64 * dx)).sum::<f64>() + (f(b) + f(a)) / 2.)
}

/// Integrate a function `f` from `a` to `b` using the [Romberg method](https://en.wikipedia.org/wiki/Romberg%27s_method),
/// stopping after either sequential estimates are less than `eps` or `n` steps have been taken.
pub fn romberg<F>(f: F, a: f64, b: f64, eps: f64, nmax: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut r: Vec<Vec<f64>> = vec![vec![0.; nmax]; nmax];

    r[0][0] = (b - a) / 2. * (f(a) + f(b));

    for n in 1..nmax {
        let hn = (b - a) / 2_f64.powi(n as i32);
        let s: f64 = (1..=2_u32.pow((n - 1) as u32))
            .map(|k| f(a + (2 * k - 1) as f64 * hn))
            .sum();
        r[n][0] = 0.5 * r[n - 1][0] + hn * s;
    }

    for n in 1..nmax {
        for m in 1..=n {
            r[n][m] = r[n][m - 1] + (r[n][m - 1] - r[n - 1][m - 1]) / (4_f64.powi(m as i32) - 1.);
        }
        if n > 1
            && (rel_diff(r[n][n], r[n - 1][n - 1]) < eps || (r[n][n] - r[n - 1][n - 1]).abs() < eps)
        {
            return r[n][n];
        }
    }

    r[nmax - 1][nmax - 1]
}

/// Given upper and lower limits of integration, this function calculates the nodes `x` and weights
/// `w` for the n-point Gauss-Legendre quadrature.
pub fn gau_leg_weights(a: f64, b: f64, n: u32) -> (Vec<f64>, Vec<f64>) {
    unimplemented!();
}

///
/// Given upper and lower limits of integration, this function calculates the nodes `x` and weights
/// `w` for the n-point Gauss-Laguerre quadrature.
pub fn gau_lag_weights(a: f64, b: f64, n: u32) -> (Vec<f64>, Vec<f64>) {
    unimplemented!();
}

/// Given upper and lower limits of integration, this function calculates the nodes `x` and weights
/// `w` for the n-point Gauss-Jacobi quadrature.
pub fn gau_jac_weights(a: f64, b: f64, n: u32) -> (Vec<f64>, Vec<f64>) {
    unimplemented!();
}

/// Given upper and lower limits of integration, this function calculates the nodes `x` and weights
/// `w` for the n-point Gauss-Hermite quadrature.
pub fn gau_her_weights(a: f64, b: f64, n: u32) -> (Vec<f64>, Vec<f64>) {
    unimplemented!();
}

/// Integrate a function `f` from `a` to `b` using the [Gauss-Legendre quadrature
/// method](https://en.wikipedia.org/wiki/Gaussian_quadrature) with 5 points (allows for
/// exact integration of polynomials up to degree 9).
pub fn quad5<F>(f: F, a: f64, b: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let xm = 0.5 * (b + a);
    let xr = 0.5 * (b - a);
    (0..5)
        .map(|i| {
            let dx = xr * GAUSS_QUAD_NODES[i];
            GAUSS_QUAD_WEIGHTS[i] * (f(xm + dx) + f(xm - dx))
        })
        .sum::<f64>()
        * xr
}

const GAUSS_QUAD_NODES: [f64; 5] = [
    0.1488743389816312,
    0.4333953941292472,
    0.6794095682990244,
    0.8650633666889845,
    0.9739065285171717,
];
const GAUSS_QUAD_WEIGHTS: [f64; 5] = [
    0.2955242247147529,
    0.2692667193099963,
    0.2190863625159821,
    0.1494513491505806,
    0.0666713443086881,
];

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;
    use std::f64::consts::PI;

    #[test]
    pub fn test_integrators() {
        let f1 = |x: f64| x * (1. + 2. * x).sqrt();
        assert_approx_eq!(trapz(f1, 4., 0., 1000), -298. / 15., 1e-2);
        assert_approx_eq!(romberg(f1, 4., 0., 1e-8, 20), -298. / 15.);

        let f2 = |x: f64| x.sin().powi(2) * x.cos().powi(2);
        assert_approx_eq!(trapz(f2, -2., 2., 1000), (8. - 8_f64.sin()) / 16., 1e-2);
        assert_approx_eq!(romberg(f2, -2., 2., 1e-8, 20), (8. - 8_f64.sin()) / 16.);

        let f3 = |x: f64| x.ln() / x;
        assert_approx_eq!(
            trapz(f3, 3., 6., 1000),
            0.5 * 2_f64.ln() * 18_f64.ln(),
            1e-2
        );
        assert_approx_eq!(
            romberg(f3, 3., 6., 1e-8, 10),
            0.5 * 2_f64.ln() * 18_f64.ln()
        );

        let f4 = |x: f64| x.sin().powi(3) * x.cos();
        assert_approx_eq!(trapz(f4, 0., PI / 3., 1000), 9. / 64., 1e-2);
        assert_approx_eq!(romberg(f4, 0., PI / 3., 1e-8, 20), 9. / 64.);

        let f5 = |x: f64| 1. / (3. * x - 7.).powi(2);
        assert_approx_eq!(trapz(f5, 3., 4., 1000), 0.1, 1e-2);
        assert_approx_eq!(romberg(f5, 3., 4., 1e-8, 20), 0.1);
    }
}
