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
