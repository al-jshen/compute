extern crate lapack;
use super::acf;
use crate::statistics::mean;
use crate::utils::*;
use std::fmt::{Display, Formatter, Result};

#[derive(Debug)]
pub struct AR {
    pub p: usize,
    pub coeffs: Vec<f64>,
    pub intercept: f64,
}

/// Implements an [autoregressive model](https://en.wikipedia.org/wiki/Autoregressive_model) of
/// order p.
#[cfg(all(feature = "blas", feature = "lapack"))]
impl AR {
    pub fn new(p: usize) -> Self {
        assert!(p > 0, "p must be greater than 0");
        AR {
            p,
            coeffs: vec![0.; p],
            intercept: 0.,
        }
    }

    /// Fit the AR(p) model to the data using the Yule-Walker equations.
    pub fn fit(&mut self, data: &[f64]) -> &mut Self {
        self.intercept = mean(&data);
        let adjusted = data
            .iter()
            .map(|x| x - self.intercept)
            .collect::<Vec<f64>>();
        let autocorrelations: Vec<f64> = (0..=self.p).map(|t| acf(&adjusted, t as i32)).collect();
        let r = &autocorrelations[1..];
        let n = r.len();
        let r_matrix = invert_matrix(&toeplitz(&autocorrelations[..n]));
        let coeffs = matmul(&r_matrix, &r, n, n, false, false);
        self.coeffs = coeffs;
        self.coeffs.reverse();
        self
    }

    /// Given some data, predict the value for a single timestep ahead.
    pub fn predict_one(&self, data: &[f64]) -> f64 {
        let n = data.len();
        let coeff_len = self.coeffs.len();
        if n >= coeff_len {
            return dot(&data[n - coeff_len..], &self.coeffs);
        } else {
            // maybe panic instead? or return NA
            // return std::f64::NAN;
            return dot(&data, &self.coeffs[..n]);
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
        d[d.len() - n..]
            .to_vec()
            .iter()
            .map(|x| x + self.intercept)
            .collect()
    }
}

impl Display for AR {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "AR({}) model", self.p)?;
        for (p, coeff) in self.coeffs.iter().rev().enumerate() {
            writeln!(f, "p{:.4} = {:.4}", p + 1, coeff)?;
        }
        writeln!(f, "intercept = {:.4}", self.intercept)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::mean;
    use approx_eq::{assert_approx_eq, rel_diff};

    #[test]
    fn test_ar_model() {
        let data = vec![
            0.21038,
            1.131721,
            3.28641,
            2.338077,
            1.499455,
            1.19406,
            -0.6015611,
            -1.287033,
            -3.051659,
            -2.630405,
            0.1041386,
            2.933628,
            3.872648,
            3.519838,
            1.81834,
            -1.454362,
            -2.431581,
            -3.986453,
            -3.122605,
            -1.141113,
            -0.07377645,
            -0.5474213,
            0.2350843,
            -1.247623,
            -1.788729,
            -0.1836658,
            -0.6114766,
            -0.0003512522,
            1.27916,
            -0.2754683,
            -1.792122,
            -0.1902297,
            -1.64871,
            -1.227125,
            -1.666066,
            -2.217532,
            0.3182005,
            0.839974,
            1.883632,
            2.562701,
            2.064571,
            1.347031,
            0.5822702,
            -0.2100001,
            -0.9831178,
            -2.022402,
            -0.2950079,
            2.435764,
            0.1554406,
            1.180818,
            0.9291775,
            -1.096983,
            -0.3009598,
            1.009731,
            -1.003446,
            -1.346068,
            0.6554112,
            0.3273469,
            0.0252534,
            0.1289094,
            0.4402104,
            -1.071554,
            -1.768173,
            -0.01722473,
            -1.309611,
            -1.140079,
            1.76984,
            1.784674,
            1.269765,
            0.4825738,
            -1.461408,
            -1.727341,
            -1.477258,
            1.036593,
            1.520819,
            0.2923091,
            0.7511532,
            1.356483,
            -1.149694,
            -3.703727,
            -2.837313,
            -2.164919,
            -0.9490226,
            1.258048,
            4.173029,
            5.098197,
            3.297466,
            1.711004,
            0.5347419,
            -2.626136,
            -3.520617,
            -2.993732,
            -1.993039,
            -1.283884,
            2.713336,
            3.42282,
            2.94359,
            2.0757,
            0.13544,
            -2.641659,
        ];

        let mut ar = AR::new(4);
        ar.fit(&data);

        let coeffs: Vec<f64> = ar.coeffs.iter().rev().copied().collect();
        let coeffs_from_r = vec![0.7976, -0.3638, 0.2437, -0.4929];

        for i in 0..4 {
            assert!(
                rel_diff(coeffs[i], coeffs_from_r[i]) < 0.05
                    || (coeffs[i] - coeffs_from_r[i]).abs() < 0.1,
                "{} {}",
                coeffs[i],
                coeffs_from_r[i]
            );
        }

        let pred = ar.predict(&data, 25);
        let pred_from_r = vec![
            -3.085083,
            -2.474237,
            -1.547886,
            0.2270113,
            1.67267,
            2.105194,
            1.900999,
            1.059832,
            -0.1426515,
            -1.05828,
            -1.455626,
            -1.318618,
            -0.6962956,
            0.1038774,
            0.7445638,
            1.048708,
            0.9470198,
            0.51771,
            -0.02880782,
            -0.482939,
            -0.7009118,
            -0.6315113,
            -0.3386402,
            0.04001635,
            0.3596356,
        ];

        for i in 0..25 {
            assert!(
                rel_diff(pred[i], pred_from_r[i]) < 0.05 || (pred[i] - pred_from_r[i]).abs() < 0.1,
                "{} {}",
                pred[i],
                pred_from_r[i]
            );
        }

        let far_pred = ar.predict(&data, 1000).pop().unwrap();
        assert_approx_eq!(far_pred, mean(&data), 1e-4);
    }
}
