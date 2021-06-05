//! A module for computing statistical moments and related values. In particular, this includes
//! means and variances.

use crate::linalg::sum;

/// An implementation of Welford's online algorithm, which is used for calculating statistics in a
/// recurrent and stable manner.
/// See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance for the reference
/// implementation of the Welford update algorithm.
fn welford_update(existing_aggregate: (usize, f64, f64), new_val: &f64) -> (usize, f64, f64) {
    // existing aggregate consists of (count, mean, M2)
    let (mut count, mut mean, mut m2) = existing_aggregate;
    count += 1;
    let delta = new_val - mean;
    mean += delta / count as f64;
    let delta2 = new_val - mean;
    m2 += delta * delta2;
    (count, mean, m2)
}

/// Uses the Welford online algorithm to calculate the count, mean, and m2 of an array of data
/// points. This is the driver for the `mean`, `variance`, and `sample_variance` functions.
fn welford_statistics(data: &[f64]) -> (usize, f64, f64) {
    let mut aggregate = (0_usize, 0., 0.);
    for i in data {
        aggregate = welford_update(aggregate, i);
    }
    aggregate
}

// /// Calulates the sum of an array of data points.
// pub fn sum(data: &[f64]) -> f64 {
//     data.iter().sum::<f64>()
// }

/// Calculates the mean of an array of data points.
pub fn mean(data: &[f64]) -> f64 {
    sum(&data) / data.len() as f64
}

/// Calculates the mean of an array of data points using the Welford algorithm.
pub fn welford_mean(data: &[f64]) -> f64 {
    let (_, mean, _) = welford_statistics(data);
    mean
}

/// Calculates the population variance from an array of data points in a numerically stable manner
/// using the Welford algorithm.
pub fn var(data: &[f64]) -> f64 {
    let (count, _, m2) = welford_statistics(data);
    m2 / count as f64
}

/// Calculates the sample variance from an array of data points in a numerically stable manner
/// using the Welford algorithm.
pub fn sample_var(data: &[f64]) -> f64 {
    let (count, _, m2) = welford_statistics(data);
    m2 / (count - 1) as f64
}

/// Calculates the standard deviation of an array of data points. This is the square root of the
/// variance.
pub fn std(data: &[f64]) -> f64 {
    var(data).sqrt()
}

/// Calculates the sample standard deviation of an array of data points. This is the square root of the
/// sample variance.
pub fn sample_std(data: &[f64]) -> f64 {
    sample_var(data).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_mean() {
        let data1: Vec<f64> = vec![
            -0.2711336,
            1.20002575,
            0.69102151,
            -0.56390913,
            -1.62661382,
            -0.0613969,
            0.39876752,
            -0.99619281,
            1.12860854,
            -0.61163405,
        ];
        assert_approx_eq!(mean(&data1), -0.071245699);

        let data2: Vec<f64> = vec![
            -1.35521905,
            0.70316493,
            -0.24386284,
            0.20382644,
            1.28818114,
            -0.90003795,
            -0.73912347,
            1.48550753,
            1.02038191,
            0.18684426,
        ];
        assert_approx_eq!(mean(&data2), 0.16496629);
    }
    #[test]
    fn test_welford_mean() {
        let data1: Vec<f64> = vec![
            -0.2711336,
            1.20002575,
            0.69102151,
            -0.56390913,
            -1.62661382,
            -0.0613969,
            0.39876752,
            -0.99619281,
            1.12860854,
            -0.61163405,
        ];
        assert_approx_eq!(welford_mean(&data1), -0.071245699);

        let data2: Vec<f64> = vec![
            -1.35521905,
            0.70316493,
            -0.24386284,
            0.20382644,
            1.28818114,
            -0.90003795,
            -0.73912347,
            1.48550753,
            1.02038191,
            0.18684426,
        ];
        assert_approx_eq!(welford_mean(&data2), 0.16496629);
    }
    #[test]
    fn test_var() {
        let data1: Vec<f64> = vec![
            -0.2711336,
            1.20002575,
            0.69102151,
            -0.56390913,
            -1.62661382,
            -0.0613969,
            0.39876752,
            -0.99619281,
            1.12860854,
            -0.61163405,
        ];
        assert_approx_eq!(var(&data1), 0.7707231173572182);

        let data2: Vec<f64> = vec![
            -1.35521905,
            0.70316493,
            -0.24386284,
            0.20382644,
            1.28818114,
            -0.90003795,
            -0.73912347,
            1.48550753,
            1.02038191,
            0.18684426,
        ];
        assert_approx_eq!(var(&data2), 0.8458540238604941);
    }
    #[test]
    fn test_sample_var() {
        let data1: Vec<f64> = vec![
            -0.2711336,
            1.20002575,
            0.69102151,
            -0.56390913,
            -1.62661382,
            -0.0613969,
            0.39876752,
            -0.99619281,
            1.12860854,
            -0.61163405,
        ];
        assert_approx_eq!(sample_var(&data1), 0.8563590181955176);

        let data2: Vec<f64> = vec![
            -1.35521905,
            0.70316493,
            -0.24386284,
            0.20382644,
            1.28818114,
            -0.90003795,
            -0.73912347,
            1.48550753,
            1.02038191,
            0.18684426,
        ];
        assert_approx_eq!(sample_var(&data2), 0.939837803612305);
    }
    #[test]
    fn test_std() {
        let data1: Vec<f64> = vec![
            -0.2711336,
            1.20002575,
            0.69102151,
            -0.56390913,
            -1.62661382,
            -0.0613969,
            0.39876752,
            -0.99619281,
            1.12860854,
            -0.61163405,
        ];
        assert_approx_eq!(std(&data1), 0.8779083758433825);

        let data2: Vec<f64> = vec![
            -1.35521905,
            0.70316493,
            -0.24386284,
            0.20382644,
            1.28818114,
            -0.90003795,
            -0.73912347,
            1.48550753,
            1.02038191,
            0.18684426,
        ];
        assert_approx_eq!(std(&data2), 0.9197032256391593);
    }
}
