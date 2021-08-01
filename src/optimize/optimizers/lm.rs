use super::DiffFn;
use crate::linalg::{
    diag, dot, inf_norm, matmul, norm, solve, svmul, vadd, xtx, Dot, Matrix, Solve, Vector,
};
use crate::statistics::max;
use autodiff::F1;

use super::Optimizer;

/// Implements a [Levenberg-Marquardt optimizer](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
/// for solving (non-linear) least squares problems.
///
/// # Example
///
/// ```rust
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
///
/// ```

#[derive(Debug, Clone, Copy)]
pub struct LM {
    pub eps1: f64, // tolerance for norm of residuals
    pub eps2: f64, // tolerance for change in parameters
    pub tau: f64,  // initial scaling for damping factor
}

impl Default for LM {
    fn default() -> Self {
        LM {
            eps1: 1e-6,
            eps2: 1e-6,
            tau: 1e-2,
        }
    }
}

impl LM {
    /// Create a new Levenberg-Marquardt optimizer.
    pub fn new(eps1: f64, eps2: f64, tau: f64) -> Self {
        LM { eps1, eps2, tau }
    }
}

impl Optimizer for LM {
    type Output = (Vector, Matrix);
    fn optimize<F>(
        &self,
        f: F,
        parameters: &[f64],
        data: &[&[f64]],
        maxsteps: usize,
    ) -> (Vector, Matrix)
    where
        F: DiffFn,
    {
        let mut params = parameters.iter().map(|&x| F1::var(x)).collect::<Vec<_>>();
        let params_vec = Vector::from(parameters);
        let param_len = params.len();
        assert!(data.len() == 2, "data must contain two slices (x and y)");
        let (xs, ys) = (data[0], data[1]);
        assert_eq!(xs.len(), ys.len(), "x and y must have the same length");
        let n = xs.len();

        let (mut res, grad): (Vector, Vec<Vector>) = xs
            .iter()
            .zip(ys)
            .map(|(&x, y)| {
                let (val, grad) = f.eval(&params, &[&[x]]);
                (y - val, grad)
            })
            .unzip();

        let mut jacobian = Matrix::new(
            grad.into_iter().flatten().collect::<Vector>(),
            n as i32,
            param_len as i32,
        );

        let mut jtj = jacobian.t_dot(&jacobian);
        let mut jtr = jacobian.t_dot(&res).to_matrix();

        let mut step = 0;
        let mut mu = self.tau * jtj.diag().max();
        let mut nu = 2.;

        let mut stop = jtr.inf_norm() <= self.eps1;

        loop {
            step += 1;
            if step > maxsteps || stop {
                break;
            }

            // apply adaptive damping parameter
            let mut damped = jtj.clone();
            for i in 0..param_len {
                damped[[i, i]] += mu * jtj[[i, i]];
            }

            let delta = damped.solve(jtr.data());

            stop = delta.norm()
                <= self.eps2 * (params.iter().map(|x| x.x).collect::<Vector>().norm() + self.eps2);
            if stop {
                break;
            }

            // calculations using new proposed parameters
            // let new_params: Vec<f64> = (0..param_len).map(|i| params[i] + delta[i]).collect();
            let new_params = params
                .iter()
                .zip(&delta)
                .map(|(x, d)| *x + F1::cst(*d))
                .collect::<Vec<_>>();

            let new_res: Vector = xs
                .iter()
                .zip(ys)
                .map(|(&x, y)| {
                    let (val, _) = f.eval(&new_params, &[&[x]]);
                    y - val
                })
                .collect();

            let res_norm_sq = res.dot(&res);
            let new_res_norm_sq = new_res.dot(&new_res);

            let pred_reduction = delta.t_dot(mu * &delta + jtr.data());

            // calculate the gain ratio (actual reduction in error over predicted reduction)
            let rho = (res_norm_sq - new_res_norm_sq) / (0.5 * pred_reduction);

            if rho > 0. {
                // good step, accept the new parameters and update all variables
                params.copy_from_slice(&new_params);

                let new_grad: Vec<Vector> = xs
                    .iter()
                    .zip(ys)
                    .map(|(&x, y)| {
                        let (_, grad) = f.eval(&new_params, &[&[x]]);
                        grad
                    })
                    .collect();

                jacobian = Matrix::new(
                    new_grad.into_iter().flatten().collect::<Vector>(),
                    n as i32,
                    param_len as i32,
                );

                jtj = jacobian.t_dot(&jacobian);
                jtr = jacobian.t_dot(&new_res).to_matrix();
                res = new_res;
                stop = jtr.inf_norm() <= self.eps1;
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

        (
            Vector::new(params.iter().map(|x| x.x).collect::<Vec<_>>()),
            res.t_dot(&res) / (n - param_len) as f64 * jtj.inv(),
        )
    }
}
