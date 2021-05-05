use crate::linalg::{diag, dot, inf_norm, matmul, norm, solve, svmul, vadd, xtx};
use crate::optimize::gradient::gradient;
use crate::statistics::max;
use autodiff::F1;

use super::{GradFn, Optimizer};

/// Implements a [Levenberg-Marquardt optimizer](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
/// for solving (non-linear) least squares problems.
///
/// # Example
///
/// ```ignore
/// use compute::optimize::{LM, Optimizer};
/// use compute::prelude::F1;
///
/// // pairs of points (x_i, y_i)
/// let x = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
/// let y = vec![11., 22., 33., 44., 55., 66., 77., 88., 99.];
///
/// // define a function to optimize:
/// // function must have signature Fn(&[F1]) -> F1,
/// // with f(x, params).
/// //
/// // so the first argument in the (function input) list must be x
/// // and the rest of the arguments in the list are parameters to optimize
/// // the output of the function is f(x, params)
/// let eq_line = |x: &[F1]| x[0] * x[2] + x[1]; // x * b + a
///
/// // create an instance of the optimizer
/// let lm = LM::default();
///
/// // initial parameters (guess)
/// let params = [1., 2.];
///
/// // run for max of 50 steps and find the best parameters
/// let opt = lm.optimize(&x, &y, eq_line, &params, 50);
/// println!("{:?}", opt);
///
/// assert!((opt[0] - 0.).abs() < 0.01);
/// assert!((opt[1] - 11.).abs() < 0.01);
/// ```

#[derive(Debug, Clone, Copy)]
pub struct LM {
    pub eps1: f64, // tolerance for norm of residuals
    pub eps2: f64, // tolerance for change in parameters
    pub tau: f64,  // initial scaling for damping factor
    gradfn: GradFn,
}

impl Default for LM {
    fn default() -> Self {
        LM {
            eps1: 1e-6,
            eps2: 1e-6,
            tau: 1e-2,
            gradfn: GradFn::Predictive,
        }
    }
}

impl LM {
    /// Create a new Levenberg-Marquardt optimizer.
    pub fn new(eps1: f64, eps2: f64, tau: f64) -> Self {
        LM {
            eps1,
            eps2,
            tau,
            gradfn: GradFn::Predictive,
        }
    }
}

impl Optimizer for LM {
    fn optimize<F>(
        &self,
        xs: &[f64],
        ys: &[f64],
        f: F,
        parameters: &[f64],
        maxsteps: usize,
    ) -> Vec<f64>
    where
        F: Fn(&[F1]) -> F1 + Copy,
    {
        assert_eq!(xs.len(), ys.len());
        let mut params = parameters.to_vec();

        let n = xs.len();
        let param_len = params.len();

        let const_params: Vec<F1> = params.iter().map(|&x| F1::cst(x)).collect();
        let mut residuals: Vec<f64> = (0..n)
            .map(|i| ys[i] - f(&[&[F1::cst(xs[i])], const_params.as_slice()].concat()).value())
            .collect();

        let mut jacobian = Vec::with_capacity(n * param_len);
        for x in xs {
            let j_i = gradient(f, &[*x], &params);
            jacobian.extend(j_i);
        }

        let mut jtj = xtx(&jacobian, n);
        let mut jtr = matmul(&jacobian, &residuals, n, n, true, false);

        let mut step = 0;
        let mut mu = self.tau * max(&diag(&jtj));
        let mut nu = 2.;

        let mut stop = inf_norm(&jtr, param_len) <= self.eps1;

        loop {
            step += 1;
            if step > maxsteps || stop {
                break;
            }

            // apply adaptive damping parameter
            let mut damped = jtj.clone();
            for i in 0..param_len {
                damped[i * param_len + i] += mu * jtj[i * param_len + i];
            }
            let delta = solve(&damped, &jtr);

            stop = norm(&delta) <= self.eps2 * (norm(&params) + self.eps2);
            if stop {
                break;
            }

            // calculations using new proposed parameters
            // let new_params: Vec<f64> = (0..param_len).map(|i| params[i] + delta[i]).collect();
            let new_params = vadd(&params, &delta);
            let new_const_params: Vec<F1> = new_params.iter().map(|&x| F1::cst(x)).collect();
            let new_residuals: Vec<f64> = (0..n)
                .map(|i| {
                    ys[i] - f(&[&[F1::cst(xs[i])], new_const_params.as_slice()].concat()).value()
                })
                .collect();

            let res_norm = dot(&residuals, &residuals);
            let new_res_norm = dot(&new_residuals, &new_residuals);
            let pred_reduction = matmul(
                &delta,
                &vadd(&svmul(mu, &delta), &jtr),
                param_len,
                param_len,
                true,
                false,
            );

            assert_eq!(pred_reduction.len(), 1);

            // calculate the gain ratio (actual reduction in error over predicted reduction)
            let rho = (res_norm - new_res_norm) / (0.5 * pred_reduction[0]);

            if rho > 0. {
                // good step, accept the new parameters and update all variables
                params[..param_len].clone_from_slice(&new_params[..param_len]);

                jacobian.clear();

                for x in xs {
                    let j_i = gradient(f, &[*x], &new_params);
                    jacobian.extend(j_i);
                }
                jtj = xtx(&jacobian, n);
                jtr = matmul(&jacobian, &new_residuals, n, n, true, false);
                residuals = new_residuals;
                stop = inf_norm(&jtr, param_len) <= self.eps1;
                if stop {
                    break;
                }
                // adjust damping factor
                mu = f64::max(1. / 3., 1. - (2. * rho - 1.).powi(3));
                nu = 2.;
            } else {
                // increase damping factor and try again with same parameters
                mu *= nu;
                nu *= 2.;
            }
        }

        params

        // TODO: implement uncertainty calculation. requires inverse cdf for t distribution.
    }
    fn grad_fn_type(&self) -> GradFn {
        self.gradfn
    }
}
