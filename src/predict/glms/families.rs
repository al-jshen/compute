use crate::linalg::{norm, svsub, vexp, vmul, vrecip, vsmul, vssub, vsub};

/// An enum to represent the [exponential
/// family](https://en.wikipedia.org/wiki/Exponential_family) set of distributions. These are
/// intended for use with [GLM](../predict/struct.GLM.html).
#[derive(Debug, Clone, Copy)]
pub enum ExponentialFamily {
    Gaussian,
    Bernoulli,
    QuasiPoisson,
    Poisson,
    Gamma,
    Exponential,
}

impl ExponentialFamily {
    pub fn has_dispersion(&self) -> bool {
        match self {
            ExponentialFamily::Gaussian => true,
            ExponentialFamily::Bernoulli => false,
            ExponentialFamily::QuasiPoisson => true,
            ExponentialFamily::Poisson => false,
            ExponentialFamily::Gamma => true,
            ExponentialFamily::Exponential => false,
        }
    }

    pub fn variance(&self, mu: &[f64]) -> Vec<f64> {
        match self {
            ExponentialFamily::Gaussian => vec![1.; mu.len()],
            ExponentialFamily::Bernoulli => vmul(mu, &svsub(1., mu)),
            ExponentialFamily::QuasiPoisson => mu.to_vec(),
            ExponentialFamily::Poisson => mu.to_vec(),
            ExponentialFamily::Gamma => vmul(&mu, &mu),
            ExponentialFamily::Exponential => vmul(&mu, &mu),
        }
    }

    pub fn inv_link(&self, eta: &[f64]) -> Vec<f64> {
        match self {
            ExponentialFamily::Gaussian => eta.to_vec(),
            ExponentialFamily::Bernoulli => eta.iter().map(|x| 1. / (1. + (-x).exp())).collect(),
            ExponentialFamily::QuasiPoisson => vexp(eta), //eta.iter().map(|x| x.exp()).collect(),
            ExponentialFamily::Poisson => vexp(eta),      //eta.iter().map(|x| x.exp()).collect(),
            ExponentialFamily::Gamma => vexp(eta),        //eta.iter().map(|x| x.exp()).collect(),
            ExponentialFamily::Exponential => vexp(eta),  //eta.iter().map(|x| x.exp()).collect(),
        }
    }

    pub fn d_inv_link(&self, eta: &[f64], mu: &[f64]) -> Vec<f64> {
        match self {
            ExponentialFamily::Gaussian => vec![1.; eta.len()],
            ExponentialFamily::Bernoulli => vmul(&mu, &svsub(1., &mu)),
            ExponentialFamily::QuasiPoisson => mu.to_vec(),
            ExponentialFamily::Poisson => mu.to_vec(),
            ExponentialFamily::Gamma => mu.to_vec(),
            ExponentialFamily::Exponential => mu.to_vec(),
        }
    }

    pub fn deviance(&self, y: &[f64], mu: &[f64]) -> f64 {
        let n = y.len();
        assert_eq!(n, mu.len());
        match self {
            ExponentialFamily::Gaussian => norm(&vsub(y, mu)),
            ExponentialFamily::Bernoulli => {
                (0..n)
                    .map(|i| y[i] * mu[i].ln() + (1. - y[i]) * (1. - mu[i]).ln())
                    .sum::<f64>()
                    * -2.
            }
            ExponentialFamily::QuasiPoisson => {
                let ylogy = y
                    .iter()
                    .map(|x| if *x == 0. { 0. } else { x * x.ln() })
                    .collect::<Vec<_>>();
                2. * (0..y.len())
                    .map(|i| mu[i] - y[i] - y[i] * mu[i].ln() + ylogy[i])
                    .sum::<f64>()
            }
            ExponentialFamily::Poisson => {
                let ylogy = y
                    .iter()
                    .map(|x| if *x == 0. { 0. } else { x * x.ln() })
                    .collect::<Vec<_>>();
                2. * (0..y.len())
                    .map(|i| mu[i] - y[i] - y[i] * mu[i].ln() + ylogy[i])
                    .sum::<f64>()
            }
            ExponentialFamily::Gamma => {
                2. * (y
                    .iter()
                    .zip(mu)
                    .map(|(yv, muv)| (yv - muv) / (muv) - (yv / muv).ln())
                    .sum::<f64>())
            }
            ExponentialFamily::Exponential => {
                2. * (y
                    .iter()
                    .zip(mu)
                    .map(|(yv, muv)| (yv - muv) / (muv) - (yv / muv).ln())
                    .sum::<f64>())
            }
        }
    }

    pub fn initial_working_response(&self, y: &[f64]) -> Option<Vec<f64>> {
        match self {
            ExponentialFamily::Gaussian => Some(y.to_vec()),
            ExponentialFamily::Bernoulli => Some(vssub(&vssub(y, 0.5), 0.25)),
            ExponentialFamily::QuasiPoisson => None,
            ExponentialFamily::Poisson => None,
            ExponentialFamily::Gamma => None,
            ExponentialFamily::Exponential => None,
        }
    }
    pub fn initial_working_weights(&self, y: &[f64]) -> Option<Vec<f64>> {
        match self {
            ExponentialFamily::Gaussian => Some(vrecip(&vec![y.len() as f64; y.len()])),
            ExponentialFamily::Bernoulli => {
                Some(vsmul(&vrecip(&vec![y.len() as f64; y.len()]), 0.25))
            }
            ExponentialFamily::QuasiPoisson => None,
            ExponentialFamily::Poisson => None,
            ExponentialFamily::Gamma => None,
            ExponentialFamily::Exponential => None,
        }
    }

    pub fn penalized_deviance(&self, y: &[f64], mu: &[f64], alpha: f64, coef: &[f64]) -> f64 {
        self.deviance(y, mu) + alpha * norm(&coef[1..])
    }
}
