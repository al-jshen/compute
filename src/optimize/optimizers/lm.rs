use crate::linalg::{diag, matmul, norm, solve, xtx};
use crate::optimize::gradient::gradient;
use crate::statistics::max;
use autodiff::F1;

/// Implements a [Levenberg-Marquardt optimizer](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
/// for solving (non-linear) least squares problems.
#[derive(Debug, Clone, Copy)]
pub struct LM {
    pub lambda: f64, // damping factor
}

impl Default for LM {
    fn default() -> Self {
        LM { lambda: 1e-4 }
    }
}

impl LM {
    /// Create a new Levenberg-Marquardt optimizer.
    ///
    /// lambda: damping factor
    pub fn new(lambda: f64) -> Self {
        LM { lambda }
    }

    pub fn optimize<F>(&self, xs: &[f64], ys: &[f64], f: F, params: &mut [f64], steps: usize)
    where
        F: Fn(&[F1]) -> F1 + Copy,
    {
        let eps1 = 1e-10;
        let eps2 = 1e-10;

        assert_eq!(xs.len(), ys.len());
        let n = xs.len();
        let param_len = params.len();

        let mut const_params: Vec<F1> = params.iter().map(|&x| F1::cst(x)).collect();
        let mut residuals: Vec<f64> = (0..n)
            .map(|i| ys[i] - f(&[&[F1::cst(xs[i])], const_params.as_slice()].concat()).value())
            .collect();

        let mut jacobian = Vec::with_capacity(n * param_len);

        for i in 0..n {
            let j_i = gradient(f, &[xs[i]], &params);
            jacobian.extend(j_i);
        }

        let mut jtj = xtx(&jacobian, n);
        let mut jtr = matmul(&jacobian, &residuals, n, n, true, false);

        let mut found = norm(&jtr) <= eps1;
        let mut step = 0;

        let mut mu = 10. * max(&diag(&jtj));
        let mut nu = 2.;

        while step <= steps && (!found) {
            step += 1;

            let mut damped = jtj.clone();
            for i in 0..param_len {
                damped[i * param_len + i] += mu * jtj[i * param_len + i];
            }
            let delta = solve(&damped, &jtr);

            if norm(&delta) <= eps2 * (norm(&params) + eps2) {
                break;
            } else {
                let new_params: Vec<f64> = (0..param_len).map(|i| params[i] + delta[i]).collect();
                let new_const_params: Vec<F1> = new_params.iter().map(|&x| F1::cst(x)).collect();
                let new_residuals: Vec<f64> = (0..n)
                    .map(|i| {
                        ys[i]
                            - f(&[&[F1::cst(xs[i])], new_const_params.as_slice()].concat()).value()
                    })
                    .collect();

                let res_norm = norm(&residuals);
                let new_res_norm = norm(&new_residuals);

                let ldelta = (0..param_len)
                    .map(|i| mu * delta[i] + jtr[i])
                    .collect::<Vec<_>>();
                let rho_denom = matmul(&delta, &ldelta, param_len, param_len, true, false);
                assert_eq!(rho_denom.len(), 1);

                let rho = (res_norm.powi(2) - new_res_norm.powi(2)) / (0.5 * rho_denom[0]);

                if rho > 0. {
                    for i in 0..param_len {
                        params[i] = new_params[i];
                    }

                    let mut jacobian = Vec::with_capacity(n * param_len);

                    for i in 0..n {
                        let j_i = gradient(f, &[xs[i]], &params);
                        jacobian.extend(j_i);
                    }
                    jtj = xtx(&jacobian, n);
                    jtr = matmul(&jacobian, &residuals, n, n, true, false);

                    found = norm(&jtr) <= eps1;

                    mu *= f64::max(1. / 3., 1. - (2. * rho - 1.).powi(3));
                    nu = 2.;
                } else {
                    mu *= nu;
                    nu *= 2.;
                }
            }
        }
    }
}
