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

impl Continuous for MVN {
    fn pdf(&self, x: Self::Output) -> f64 {
        assert!(self.covariance_matrix.is_positive_definite());

        let x_minus_mu = &x - &self.mean;
        let numerator =
            (-0.5 * &x_minus_mu.t_dot(&self.inverse_covariance_matrix.dot(&x_minus_mu))).exp();
        let denominator = ((2. * PI).powi(x.len() as i32) * self.covariance_determinant).sqrt();
        numerator / denominator
    }
}

impl Mean for MVN {
    type MeanType = Vector;
    fn mean(&self) -> Self::MeanType {
        self.mean.clone()
    }
}

impl Variance for MVN {
    type VarianceType = Matrix;
    fn var(&self) -> Self::VarianceType {
        self.covariance_matrix.clone()
    }
}
