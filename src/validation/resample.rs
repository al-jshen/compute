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

/// Given a length-n array of data, returns all leave-one-out length n-1 vectors. See
/// <https://en.wikipedia.org/wiki/Jackknife_resampling>
pub fn jackknife(data: &[f64]) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut resamples: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let (front, back) = data.split_at(i);
        let (_, rest) = back.split_first().unwrap();
        let mut v = front.to_vec();
        v.extend_from_slice(rest);
        resamples.push(v);
    }
    resamples
}

/// Shuffle an array in-place.
pub fn shuffle(data: &mut [f64]) {
    let randomizer = DiscreteUniform::new(0, data.len() as i64 - 1);
    for _ in 0..data.len() * 2 {
        let (a, b) = (randomizer.sample(), randomizer.sample());
        data.swap(a as usize, b as usize);
    }
}

/// Shuffle two array in-place. The same shuffling is applied to both arrays. That is, a pair (x_i,
/// y_i) will still be paired together as (x_j, y_j) after shuffling.
pub fn shuffle_two(arr1: &mut [f64], arr2: &mut [f64]) {
    assert_eq!(arr1.len(), arr2.len());
    let randomizer = DiscreteUniform::new(0, arr1.len() as i64 - 1);
    for _ in 0..arr1.len() * 2 {
        let (a, b) = (randomizer.sample(), randomizer.sample());
        arr1.swap(a as usize, b as usize);
        arr2.swap(a as usize, b as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{Distribution, Normal};
    use crate::statistics::{max, mean, min, std};
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

    #[test]
    fn test_jackknife_size() {
        let x = Normal::default().sample_vec(50);
        let jk_samples = jackknife(&x);
        for s in jk_samples {
            assert_eq!(s.len(), 49);
        }
    }

    #[test]
    fn test_jackknife_mean() {
        let x = Normal::default().sample_vec(100);
        let jk_samples = jackknife(&x);
        let jk_means = jk_samples.iter().map(|s| mean(s)).collect::<Vec<_>>();
        assert_approx_eq!(mean(&x), mean(&jk_means));
    }
}
