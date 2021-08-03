use super::Optimizer;
use crate::linalg::{Dot, Matrix, Solve, Vector};
use reverse::*;

/// Implements a [Levenberg-Marquardt optimizer](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
/// for solving (non-linear) least squares problems.
///
/// # Example
///
/// ```rust
/// use compute::prelude::{LM, Optimizer, Vector};
/// use reverse::*;
///
/// // pairs of points (x_i, y_i)
/// let x = Vector::from([1., 2., 3., 4., 5., 6., 7., 8., 9.]);
/// let y = Vector::from([11., 22., 33., 44., 55., 66., 77., 88., 99.]);
///
/// // define a function to optimize:
/// // f(parameters, data) where parameters are parameters to optimize
/// // and data are data to be used in the function
///
/// fn equation_line<'a>(params: &[Var<'a>], data: &[&[f64]]) -> Var<'a> {
///     assert!(data.len() == 1);
///     assert!(data[0].len() == 1);
///     assert!(params.len() == 2);
///
///     return data[0][0] * params[0] + params[1];
/// }
///
/// // create an instance of the optimizer
/// let lm = LM::default();
///
/// // initial parameters (guess)
/// let params = [1., 2.];
///
/// // run for max of 50 steps and find the best parameters
/// // and the associated estimated covariance matrix.
/// // the standard deviations of the parameters can be obtained from the
/// // square root of the diagonal elements of the covariance matrix.
/// let (popt, pcov) = lm.optimize(equation_line, &params, &[&x, &y], 50);
/// let perr = pcov.diag().sqrt();
///
/// println!("{}", popt);
/// println!("{}", pcov);
///
/// assert!((popt[0] - 11.).abs() < 0.01);
/// assert!((popt[1] - 0.).abs() < 0.01);
/// ```

#[derive(Debug, Clone)]
pub struct LM {
    pub eps1: f64, // tolerance for norm of residuals
    pub eps2: f64, // tolerance for change in parameters
    pub tau: f64,  // initial scaling for damping factor
    tape: Tape,    // tape for computing gradients
}

impl Default for LM {
    fn default() -> Self {
        LM {
            eps1: 1e-6,
            eps2: 1e-6,
            tau: 1e-2,
            tape: Tape::new(),
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
            tape: Tape::new(),
        }
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
        F: for<'a> Fn(&[Var<'a>], &[&[f64]]) -> Var<'a>,
    {
        self.tape.clear();
        let mut params = parameters
            .into_iter()
            .copied()
            .map(|x| self.tape.add_var(x))
            .collect::<Vec<_>>();

        let param_len = params.len();
        assert!(data.len() == 2, "data must contain two slices (x and y)");
        let (xs, ys) = (data[0], data[1]);
        assert_eq!(xs.len(), ys.len(), "x and y must have the same length");
        let n = xs.len();

        let (mut res, grad): (Vector, Vec<Vector>) = xs
            .iter()
            .zip(ys)
            .map(|(&x, &y)| {
                let val = f(&params, &[&[x]]);
                ((y - val).val(), Vector::from(val.grad().wrt(&params)))
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
                <= self.eps2
                    * (params.iter().map(|x| x.val()).collect::<Vector>().norm() + self.eps2);
            if stop {
                break;
            }

            // calculations using new proposed parameters
            // let new_params: Vec<f64> = (0..param_len).map(|i| params[i] + delta[i]).collect();
            let new_params = params
                .iter()
                .zip(&delta)
                .map(|(&x, &d)| x + d)
                .collect::<Vec<_>>();

            let new_res: Vector = xs
                .iter()
                .zip(ys)
                .map(|(&x, y)| {
                    let val = f(&new_params, &[&[x]]).val();
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

                let new_grad = xs
                    .iter()
                    .map(|&x| {
                        let res = f(&new_params, &[&[x]]);
                        Vector::from(res.grad().wrt(&new_params))
                    })
                    .flatten()
                    .collect::<Vector>();

                jacobian = Matrix::new(new_grad, n as i32, param_len as i32);

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

            // clear gradients and intermediate variables
            self.tape.clear();
            params = params
                .iter()
                .map(|x| self.tape.add_var(x.val()))
                .collect::<Vec<_>>();
        }

        (
            Vector::new(params.iter().map(|x| x.val()).collect::<Vec<_>>()),
            res.t_dot(&res) / (n - param_len) as f64 * jtj.inv(),
        )
    }
}
