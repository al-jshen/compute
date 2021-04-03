use crate::linalg::{norm, svsub, vmul};

pub trait HasVariance {
    fn variance(&self, mu: &[f64]) -> Vec<f64>;
}

pub trait HasInvLink {
    fn inv_link(&self, nu: &[f64]) -> Vec<f64>;
    fn d_inv_link(&self, nu: &[f64], mu: &[f64]) -> Vec<f64>;
}

pub trait HasDeviance {
    fn deviance(&self, y: &[f64], mu: &[f64]) -> f64;
}

pub trait HasPenalizedDeviance: HasDeviance {
    fn penalized_deviance(&self, y: &[f64], mu: &[f64], alpha: f64, coef: &[f64]) -> f64 {
        self.deviance(&y, mu) + alpha * norm(&coef[1..])
    }
}

pub trait HasDispersion {
    fn has_dispersion(&self) -> bool;
}

pub enum ExponentialFamily {
    Gaussian,
    Bernoulli,
    QuasiPoisson,
    Poisson,
    Gamma,
    Exponential,
}

impl HasDispersion for ExponentialFamily {
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
}

impl HasVariance for ExponentialFamily {
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
}
