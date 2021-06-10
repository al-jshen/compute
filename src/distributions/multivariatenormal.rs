use super::{Continuous, Distribution, Normal};
use crate::prelude::{Dot, Matrix, Vector};

/// [Multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution).
pub struct MVN {
    mean: Vector,
    decomposed_covariance_matrix: Matrix,
}

pub type MultivariateNormal = MVN;

impl MVN {
    fn new<V, M>(mean: V, covariance_matrix: M) -> Self
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

        Self {
            mean: m,
            decomposed_covariance_matrix: c.cholesky(),
        }
    }
}

// yikes, need to rewrite all distribution stuff with generics.
//
// impl Distribution for MVN {
//     fn sample(&self) -> Vector {
//         let z = Normal::default().sample_vec(self.mean.len());
//         &self.mean + self.decomposed_covariance_matrix.dot(z)
//     }
// }
