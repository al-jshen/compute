use crate::linalg::{norm, svsub, vmul, vsdiv, vsmul, vssub, vsub};

// pub trait HasVariance {
//     fn variance(&self, mu: &[f64]) -> Vec<f64>;
// }

// pub trait HasInvLink {
//     fn inv_link(&self, nu: &[f64]) -> Vec<f64>;
//     fn d_inv_link(&self, nu: &[f64], mu: &[f64]) -> Vec<f64>;
// }

// pub trait HasDeviance {
//     fn deviance(&self, y: &[f64], mu: &[f64]) -> f64;
// }

// pub trait HasPenalizedDeviance: HasDeviance {
//     fn penalized_deviance(&self, y: &[f64], mu: &[f64], alpha: f64, coef: &[f64]) -> f64 {
//         self.deviance(&y, mu) + alpha * norm(&coef[1..])
//     }
// }

// pub trait HasDispersion {
//     fn has_dispersion(&self) -> bool;
// }

// pub trait HasInitialValues {
//     fn initial_working_response(&self, y: &[f64]) -> Option<Vec<f64>>;
//     fn initial_working_weights(&self, y: &[f64]) -> Option<Vec<f64>>;
// }

pub enum ExponentialFamily {
    Gaussian,
    Bernoulli,
    QuasiPoisson,
    Poisson,
    Gamma,
    Exponential,
}

impl ExponentialFamily {
    fn has_dispersion(&self) -> bool {
        match self {
            ExponentialFamily::Gaussian => true,
            ExponentialFamily::Bernoulli => false,
            ExponentialFamily::QuasiPoisson => true,
            ExponentialFamily::Poisson => false,
            ExponentialFamily::Gamma => true,
            ExponentialFamily::Exponential => false,
        }
    }

    fn variance(&self, mu: &[f64]) -> Vec<f64> {
        match self {
            ExponentialFamily::Gaussian => vec![1.; mu.len()],
            ExponentialFamily::Bernoulli => vmul(&mu, &svsub(1., &mu)),
            ExponentialFamily::QuasiPoisson => mu.to_vec(),
            ExponentialFamily::Poisson => mu.to_vec(),
            ExponentialFamily::Gamma => vmul(&mu, &mu),
            ExponentialFamily::Exponential => vmul(&mu, &mu),
        }
    }

    fn inv_link(&self, nu: &[f64]) -> Vec<f64> {
        match self {
            ExponentialFamily::Gaussian => nu.to_vec(),
            ExponentialFamily::Bernoulli => nu.iter().map(|x| 1. / (1. + (-x).exp())).collect(),
            ExponentialFamily::QuasiPoisson => nu.iter().map(|x| x.exp()).collect(),
            ExponentialFamily::Poisson => nu.iter().map(|x| x.exp()).collect(),
            ExponentialFamily::Gamma => nu.iter().map(|x| x.exp()).collect(),
            ExponentialFamily::Exponential => nu.iter().map(|x| x.exp()).collect(),
        }
    }

    fn d_inv_link(&self, nu: &[f64], mu: &[f64]) -> Vec<f64> {
        match self {
            ExponentialFamily::Gaussian => vec![1.; nu.len()],
            ExponentialFamily::Bernoulli => nu.iter().map(|x| x.exp()).collect(),
            ExponentialFamily::QuasiPoisson => mu.to_vec(),
            ExponentialFamily::Poisson => mu.to_vec(),
            ExponentialFamily::Gamma => mu.to_vec(),
            ExponentialFamily::Exponential => mu.to_vec(),
        }
    }

    fn deviance(&self, y: &[f64], mu: &[f64]) -> f64 {
        match self {
            ExponentialFamily::Gaussian => norm(&vsub(y, mu)),
            ExponentialFamily::Bernoulli => {
                -2. * (y
                    .iter()
                    .zip(mu)
                    .map(|(yv, muv)| (yv * muv.ln() * (1. - yv) * (1. - muv.ln())))
                    .sum::<f64>())
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

    fn initial_working_response(&self, y: &[f64]) -> Option<Vec<f64>> {
        match self {
            ExponentialFamily::Gaussian => Some(y.to_vec()),
            ExponentialFamily::Bernoulli => Some(vssub(&vssub(y, 0.5), 0.25)),
            ExponentialFamily::QuasiPoisson => None,
            ExponentialFamily::Poisson => None,
            ExponentialFamily::Gamma => None,
            ExponentialFamily::Exponential => None,
        }
    }
    fn initial_working_weights(&self, y: &[f64]) -> Option<Vec<f64>> {
        match self {
            ExponentialFamily::Gaussian => Some(vsdiv(&vec![1.; y.len()], y.len() as f64)),
            ExponentialFamily::Bernoulli => {
                Some(vsmul(&vsdiv(&vec![1.; y.len()], y.len() as f64), 0.25))
            }
            ExponentialFamily::QuasiPoisson => None,
            ExponentialFamily::Poisson => None,
            ExponentialFamily::Gamma => None,
            ExponentialFamily::Exponential => None,
        }
    }
}
