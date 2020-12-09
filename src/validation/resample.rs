//! Algorithms for data resampling.

use crate::distributions::{DiscreteUniform, Distribution};

/// Given an array of data, returns `n_bootstrap` vectors, where each has elements that are drawn
/// from the original array with replacement, and the length of each vector is the same as the
/// length of the original array.
pub fn bootstrap(data: &[f64], n_bootstrap: usize) -> Vec<Vec<f64>> {
    let mut resamples: Vec<Vec<f64>> = Vec::with_capacity(n_bootstrap);

    let resamp_gen = DiscreteUniform::new(0, (data.len() - 1) as i64);

    for _ in 0..n_bootstrap {
        let idxs = resamp_gen.sample_vec(data.len());
        resamples.push(idxs.into_iter().map(|i| data[i as usize]).collect());
    }

    resamples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{Distribution, Normal};
    use crate::summary::{max, mean, min, std};
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_bootstrap_range() {
        let x: Vec<f64> = (0..25).into_iter().map(|x| x as f64).collect();
        let samps = bootstrap(&x, 50);
        for i in samps {
            assert!(min(&i) >= min(&x));
            assert!(max(&i) <= max(&x))
        }
    }

    #[test]
    fn test_bootstrap_moments() {
        let x = Normal::new(4., 2.).sample_vec(1e3 as usize);
        let samps = bootstrap(&x, 50);
        let means: Vec<f64> = (&samps).into_iter().map(|samp| mean(samp)).collect();
        let stds: Vec<f64> = (&samps).into_iter().map(|samp| std(samp)).collect();

        // check that the sampling mean and sampling std agree with
        // the "true" mean and std within 2.5%
        assert_approx_eq!(mean(&x), mean(&means), 0.025);
        assert_approx_eq!(std(&x), mean(&stds), 0.025);
    }
}
