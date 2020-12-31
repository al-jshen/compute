extern crate lapack;
use super::acf;
use crate::summary::mean;
use crate::utils::*;
use std::fmt::{Display, Formatter, Result};

#[derive(Debug)]
pub struct AR {
    p: usize,
    coeffs: Vec<f64>,
    intercept: f64,
}

/// Implements an [autoregressive model](https://en.wikipedia.org/wiki/Autoregressive_model) of
/// order p.
impl AR {
    pub fn new(p: usize) -> Self {
        assert!(p > 0, "p must be greater than 0");
        AR {
            p,
            coeffs: vec![1.; p],
            intercept: 0.,
        }
    }

    /// Fit the AR(p) model to the data using the Yule-Walker equations.
    pub fn fit(&mut self, data: &[f64]) -> &mut Self {
        let autocorrelations: Vec<f64> = (0..=self.p).map(|t| acf(data, t as i32)).collect();
        let r = &autocorrelations[1..];
        let n = r.len() as i32;
        let r_matrix = invert_matrix(&toeplitz_even_square(&autocorrelations[..n as usize]));
        let coeffs = matmul(&r_matrix, &r, n, n, false, false);
        self.coeffs = coeffs;
        self.coeffs.reverse();
        self.intercept = mean(&data);
        self
    }

    /// Given some data, predict the value for a single timestep ahead.
    pub fn predict_one(&self, data: &[f64]) -> f64 {
        let n = data.len();
        let coeff_len = self.coeffs.len();
        if n >= coeff_len {
            return dot(&data[n - coeff_len..], &self.coeffs) + self.intercept;
        } else {
            return dot(data, &self.coeffs[..n]) + self.intercept;
        }
    }

    /// Predict n values ahead. For forecasts after the first forecast, uses previous forecasts as
    /// "data" to create subsequent forecasts.
    pub fn predict(&self, data: &[f64], n: usize) -> Vec<f64> {
        let forecasts = vec![0.; n];
        let mut d: Vec<f64> = data[data.len() - self.coeffs.len()..].to_vec();
        d.extend(forecasts);
        for i in self.coeffs.len()..d.len() {
            d[i] = self.predict_one(&d[..i]);
        }
        d[d.len() - n..].to_vec()
    }
}

impl Display for AR {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "AR({}) model", self.p)?;
        for (p, coeff) in self.coeffs.iter().rev().enumerate() {
            writeln!(f, "p{}={}", p, coeff)?;
        }
        writeln!(f, "intercept={}", self.intercept)?;
        Ok(())
    }
}
