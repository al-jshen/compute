use super::Optimizer;
use crate::linalg::Vector;
use approx_eq::rel_diff;
use reverse::*;

/// Implements the Stochastic Gradient Descent optimizer with (Nesterov) momentum.
///
/// # Examples
///
/// ```rust
/// // optimize the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function)
/// // with fixed parameters `a = 1` and `b = 100`.
/// // the minimum value is at (1, 1), which is what we will try to recover
///
/// use compute::optimize::*;
/// use approx_eq::assert_approx_eq;
/// use reverse::Var;
///
/// fn rosenbrock<'a>(p: &[Var<'a>], d: &[&[f64]]) -> Var<'a> {
///     assert_eq!(p.len(), 2);
///     assert_eq!(d.len(), 1);
///     assert_eq!(d[0].len(), 2);
///
///     let (x, y) = (p[0], p[1]);
///     let (a, b) = (d[0][0], d[0][1]);
///
///     (a - x).powi(2) + b * (y - x.powi(2)).powi(2)
/// }
///
/// let init = [0., 0.];
/// let optim = SGD::new(1e-3, 0.9, true);
/// let popt = optim.optimize(rosenbrock, &init, &[&[1., 100.]], 10000);
///
/// assert_approx_eq!(popt[0], 1.);
/// assert_approx_eq!(popt[1], 1.);
/// ```
#[derive(Debug, Clone)]
pub struct SGD {
    stepsize: f64,  // step size
    momentum: f64,  // momentum
    nesterov: bool, // whether to use Nesterov accelerated gradient
    tape: Tape,     // tape for computing gradients
}

impl SGD {
    /// Create a new SGD optimizer.
    ///
    /// stepsize: step size
    /// momentum: momentum factor
    /// nesterov: whether to use Nesterov momentum
    pub fn new(stepsize: f64, momentum: f64, nesterov: bool) -> Self {
        Self {
            stepsize,
            momentum,
            nesterov,
            tape: Tape::new(),
        }
    }
    pub fn set_stepsize(&mut self, stepsize: f64) {
        assert!(stepsize > 0., "stepsize must be positive");
        self.stepsize = stepsize;
    }
}

impl Default for SGD {
    fn default() -> Self {
        Self {
            stepsize: 1e-5,
            momentum: 0.9,
            nesterov: true,
            tape: Tape::new(),
        }
    }
}

impl Optimizer for SGD {
    type Output = Vector;
    /// Run the optimization algorithm, given a vector of parameters to optimize and a function which calculates the residuals.
    fn optimize<F>(
        &self,
        f: F,
        parameters: &[f64],
        data: &[&[f64]],
        maxsteps: usize,
    ) -> Self::Output
    where
        F: for<'a> Fn(&[Var<'a>], &[&[f64]]) -> Var<'a>,
    {
        self.tape.clear();
        let param_len = parameters.len();
        let mut params = parameters
            .iter()
            .map(|&x| self.tape.add_var(x))
            .collect::<Vec<_>>();
        let mut update_vec = Vector::zeros(param_len);

        let mut t: usize = 0;
        let mut converged = false;

        while t < maxsteps && !converged {
            t += 1;
            let prev_params = params.clone();

            let grad = if self.nesterov {
                let future_params = params
                    .iter()
                    .zip(&update_vec)
                    .map(|(p, u)| *p - self.momentum * u)
                    .collect::<Vec<_>>();
                let res = f(&future_params, data);
                eprintln!("t = {:?}, res = {}", t, res.val());
                res.grad().wrt(&future_params)
            } else {
                f(&params, data).grad().wrt(&params)
            };

            for p in 0..param_len {
                update_vec[p] = self.momentum * update_vec[p] + self.stepsize * grad[p];
                params[p] = params[p] - update_vec[p]
            }
            // println!("{:?}", params);

            if crate::statistics::max(
                &(0..param_len)
                    .map(|i| rel_diff(params[i].val(), prev_params[i].val()))
                    .collect::<Vec<_>>(),
            ) < f64::EPSILON
            {
                converged = true;
            }

            // clear gradients and intermediate variables
            self.tape.clear();
            params = params
                .iter()
                .map(|&x| self.tape.add_var(x.val()))
                .collect::<Vec<_>>();
        }
        Vector::from(params.iter().map(|x| x.val()).collect::<Vec<_>>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_sgd_slr() {
        let x = vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35.,
            36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52.,
            53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65., 66., 67., 68., 69.,
            70., 71., 72., 73., 74., 75., 76., 77., 78., 79., 80., 81., 82., 83., 84., 85., 86.,
            87., 88., 89., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99.,
        ];
        // actual coefficients
        let coeffs = vec![5., -2.5];
        let y = [
            5., 2.5, 0., -2.5, -5., -7.5, -10., -12.5, -15., -17.5, -20., -22.5, -25., -27.5, -30.,
            -32.5, -35., -37.5, -40., -42.5, -45., -47.5, -50., -52.5, -55., -57.5, -60., -62.5,
            -65., -67.5, -70., -72.5, -75., -77.5, -80., -82.5, -85., -87.5, -90., -92.5, -95.,
            -97.5, -100., -102.5, -105., -107.5, -110., -112.5, -115., -117.5, -120., -122.5,
            -125., -127.5, -130., -132.5, -135., -137.5, -140., -142.5, -145., -147.5, -150.,
            -152.5, -155., -157.5, -160., -162.5, -165., -167.5, -170., -172.5, -175., -177.5,
            -180., -182.5, -185., -187.5, -190., -192.5, -195., -197.5, -200., -202.5, -205.,
            -207.5, -210., -212.5, -215., -217.5, -220., -222.5, -225., -227.5, -230., -232.5,
            -235., -237.5, -240., -242.5,
        ];

        fn fn_resid<'a>(params: &[Var<'a>], data: &[&[f64]]) -> Var<'a> {
            let (x, y) = (data[0], data[1]);
            x.iter()
                .zip(y)
                .map(|(&xv, &yv)| ((params[0] + xv * params[1]) - yv).powi(2))
                .sum()
        }

        let optim = SGD::new(2e-6, 0.9, true);
        let est_params = optim.optimize(fn_resid, &[1., 1.], &[&x, &y], 10000);

        for i in 0..2 {
            assert_approx_eq!(est_params[i], coeffs[i], 0.01);
        }
    }
}
