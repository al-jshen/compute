use std::f64::consts::PI;

use super::{Continuous, Distribution, Distribution1D, DistributionND, Mean, Normal, Variance};
use crate::prelude::{Dot, Matrix, Vector};

/// [Multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution).
#[derive(Debug, Clone)]
pub struct MVN {
    mean: Vector,
    covariance_matrix: Matrix,
    inverse_covariance_matrix: Matrix,
    covariance_determinant: f64,
    decomposed_covariance_matrix: Matrix,
}

pub type MultivariateNormal = MVN;

impl MVN {
    pub fn new<V, M>(mean: V, covariance_matrix: M) -> Self
    where
        V: Into<Vector>,
        M: Into<Matrix>,
    {
        let m = mean.into();
        let c = covariance_matrix.into();

        assert!(c.is_symmetric(), "covariance matrix must be symmetric");
        assert_eq!(
            m.len(),
            c.ncols,
            "mean vector and covariance matrix must have the same dimensions"
        );

        // don't really want to compute these if not necessary but if you make these option<..> and
        // compute only when necessary it gets kind of nasty because you need &mut self for e.g.
        // the pdf method which requires the trait definition to be changed. will just eat the cost
        // for now and figure something else out. still better than not caching.
        let l = (&c).cholesky();
        let cinv = (&c).inv();
        let cdet = (&c).det();

        Self {
            mean: m,
            covariance_matrix: c,
            inverse_covariance_matrix: cinv,
            covariance_determinant: cdet,
            decomposed_covariance_matrix: l,
        }
    }
}

impl Distribution for MVN {
    type Output = Vector;
    fn sample(&self) -> Vector {
        let z = Normal::default().sample_n(self.mean.len());
        &self.mean + self.decomposed_covariance_matrix.dot(z)
    }
}

impl DistributionND for MVN {
    fn get_dim(&self) -> usize {
        self.mean.len()
    }
}

impl<'a> Continuous for &'a MVN {
    type PDFType = &'a [f64];
    fn pdf(&self, x: Self::PDFType) -> f64 {
        assert!(self.covariance_matrix.is_positive_definite());
        assert_eq!(x.len(), self.mean.len());

        let x_minus_mu: Vector = x
            .iter()
            .enumerate()
            .map(|(i, v)| v - self.mean[i])
            .collect();

        let numerator =
            (-0.5 * &x_minus_mu.t_dot(&self.inverse_covariance_matrix.dot(&x_minus_mu))).exp();
        let denominator = ((2. * PI).powi(x.len() as i32) * self.covariance_determinant).sqrt();

        numerator / denominator
    }

    fn ln_pdf(&self, x: Self::PDFType) -> f64 {
        assert!(self.covariance_matrix.is_positive_definite());
        assert_eq!(x.len(), self.mean.len());

        let x_minus_mu: Vector = x
            .iter()
            .enumerate()
            .map(|(i, v)| v - self.mean[i])
            .collect();

        -0.5 * (self.covariance_determinant.ln()
            + &x_minus_mu.t_dot(&self.inverse_covariance_matrix.dot(&x_minus_mu))
            + x.len() as f64 * (2. * PI).ln())
    }
}

impl<'a> Mean for &'a MVN {
    type MeanType = &'a [f64];
    fn mean(&self) -> Self::MeanType {
        &self.mean
    }
}

impl<'a> Variance for &'a MVN {
    type VarianceType = &'a Matrix;
    fn var(&self) -> Self::VarianceType {
        &self.covariance_matrix
    }
}

#[cfg(test)]
mod tests {
    use super::Continuous;
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_mvn_pdf() {
        let mu = Vector::new([
            0.6971976638355714,
            -0.6676833280983583,
            -2.0192124253834733,
            -1.5335621337312673,
        ]);

        let cov = Matrix::new(
            [
                2.247288887309859,
                0.2995972155043716,
                0.5845592696474896,
                -0.13434631148751136,
                0.2995972155043716,
                1.3959897541030757,
                -0.1601386729230161,
                2.2253865738659315,
                0.5845592696474896,
                -0.1601386729230161,
                3.977276244924999,
                -1.977313729867125,
                -0.13434631148751136,
                2.2253865738659315,
                -1.977313729867125,
                8.06177161880807,
            ],
            4,
            4,
        );

        let mvn = MVN::new(mu, cov);

        let x1 = Vector::new([
            0.050102652382139026,
            -0.0521232079055611,
            0.6617157383972537,
            -0.8086304981120899,
        ]);
        let x2 = Vector::new([
            0.8416518707855118,
            -1.1531229014478865,
            1.7008635367302818,
            -0.6559951109477243,
        ]);
        let x3 = Vector::new([
            0.6545389674230797,
            1.739584646246535,
            -0.4158677788241667,
            1.2753434275913207,
        ]);

        assert_approx_eq!(0.0008500500589160902, (&mvn).pdf(&x1));
        assert_approx_eq!(0.0001612231592518467, (&mvn).pdf(&x2));
        assert_approx_eq!(0.00025701999301292773, (&mvn).pdf(&x3));
    }
}
