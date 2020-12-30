extern crate lapack;
use super::acf;
use crate::utils::*;

pub struct AR {
    pub p: usize,
    pub intercept: f64,
    pub coeffs: Vec<f64>,
}

impl AR {
    pub fn new(p: usize) -> Self {
        AR {
            p,
            intercept: 0.,
            coeffs: vec![1.; p],
        }
    }

    pub fn fit(&mut self, data: &[f64]) -> &mut Self {
        let autocorrelations: Vec<f64> = (0..=self.p).map(|t| acf(data, t as i32)).collect();
        let r = &autocorrelations[1..];
        let n = r.len() as i32;
        let r_matrix = invert_matrix(&toeplitz_even_square(&autocorrelations[..n as usize]));
        let coeffs = matmul(&r_matrix, &r, n, n, false, false);
        self.coeffs = coeffs;
        self
    }
}
