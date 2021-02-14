use super::GradFn;
use super::Optimizer;
use crate::optimize::gradient::gradient;
use crate::validation::shuffle_two;
use approx_eq::rel_diff;
use autodiff::F1;

/// Implements the Adam optimizer. See [Kingma and Ba 2014](https://arxiv.org/abs/1412.6980) for
/// details about the algorithm.
#[derive(Debug, Clone, Copy)]
pub struct Adam {
    stepsize: f64,    // step size
    beta1: f64,       // exponential decay rate for first moment
    beta2: f64,       // exponential decay rate for second moment
    epsilon: f64,     // small number to prevent division by zero
    batchsize: usize, // batch size to use for gradients
    gradfn: GradFn,   // gradient function type
}

impl Adam {
    /// Create a new Adam optimizer.
    /// stepsize: step size
    /// beta1: exponential decay rate for first moment
    /// beta2: exponential decay rate for second moment
    /// epsilon: small number to prevent division by zero
    pub fn new(stepsize: f64, beta1: f64, beta2: f64, epsilon: f64, batchsize: usize) -> Self {
        Adam {
            stepsize,
            beta1,
            beta2,
            epsilon,
            batchsize,
            gradfn: GradFn::Residual,
        }
    }
}

impl Default for Adam {
    /// Uses the defaults recommended by Kingma and Ba 2014
    fn default() -> Self {
        Adam {
            stepsize: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            batchsize: 16,
            gradfn: GradFn::Residual,
        }
    }
}

impl Optimizer for Adam {
    /// Run the optimization algorithm, given a vector of parameters to optimize and a function which calculates the residuals.
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
        let n = xs.len();
        let mut params = parameters.to_vec();
        let param_len = params.len();

        let mut t: usize = 0;
        let mut m = vec![0.; param_len];
        let mut v = vec![0.; param_len];
        let mut converged = false;

        while t < maxsteps && !converged {
            t += 1;
            let prev_params = params.clone();

            let (xshuf, yshuf) = shuffle_two(xs, ys);

            for batch in (0..n).collect::<Vec<_>>().chunks(self.batchsize) {
                let mut grad = vec![0.; param_len];
                for i in 0..batch.len() {
                    let g_i = gradient(f, &[yshuf[i], xshuf[i]], &params);
                    for k in 0..param_len {
                        grad[k] += g_i[k];
                    }
                }
                // println!("{:?}", grad);
                for p in 0..param_len {
                    m[p] = self.beta1 * m[p] + (1. - self.beta1) * grad[p]; // biased first moment estimate
                    v[p] = self.beta2 * v[p] + (1. - self.beta2) * grad[p].powi(2); // biased second moment estimate
                    let mhat = m[p] / (1. - self.beta1.powi(t as i32)); // bias-corrected first moment estimate
                    let vhat = v[p] / (1. - self.beta2.powi(t as i32)); // bias-corrected second moment estimate
                    params[p] -= self.stepsize * mhat / (vhat.sqrt() + self.epsilon);
                }

                // println!("{:?}", params);
            }

            if crate::statistics::max(
                &(0..param_len)
                    .map(|i| rel_diff(params[i], prev_params[i]))
                    .collect::<Vec<_>>(),
            ) < f64::EPSILON
            {
                converged = true;
            }
        }
        params
    }
    fn grad_fn_type(&self) -> GradFn {
        self.gradfn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;
    use autodiff::Float;

    #[test]
    fn test_adam_slr() {
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

        let fres = |x: &[F1]| (x[0] - (x[1] * x[3] + x[2])).powi(2);

        let optim = Adam::new(8e-4, 0.99, 0.999, 1e-6, 2);
        let est_params = optim.optimize(&x, &y, fres, &[1., 1.], 1000);

        for i in 0..2 {
            assert_approx_eq!(est_params[i], coeffs[i], 0.015);
        }
    }
}
