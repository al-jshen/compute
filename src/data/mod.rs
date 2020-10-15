/// An implementation of Welford's online algorithm, which is used for calculating statistics in a
/// recurrent and stable manner.
/// See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance for the reference
/// implementation of the Welford update algorithm.
pub fn welford_update(existing_aggregate: (usize, f64, f64), new_val: &f64) -> (usize, f64, f64) {
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
pub fn welford_statistics(data: &[f64]) -> (usize, f64, f64) {
    let mut aggregate = (0 as usize, 0., 0.);
    for i in data {
        aggregate = welford_update(aggregate, i);
    }
    aggregate
}

/// Calculates the mean of an array of data points in a numerically stable manner
/// using the Welford algorithm.
/// ```
/// use statistics::data::mean;
/// use approx_eq::assert_approx_eq;
///
/// let data1: Vec<f64> = vec![-0.2711336 ,  1.20002575,  0.69102151, -0.56390913, -1.62661382, -0.0613969 ,  0.39876752, -0.99619281,  1.12860854, -0.61163405];
/// assert_approx_eq!(mean(&data1), -0.071245699);
/// println!("{}", mean(&data1));
///
/// let data2: Vec<f64> = vec![-1.35521905,  0.70316493, -0.24386284,  0.20382644,  1.28818114, -0.90003795, -0.73912347,  1.48550753,  1.02038191,  0.18684426];
/// assert_approx_eq!(mean(&data2), 0.16496629);
/// ```
pub fn mean(data: &[f64]) -> f64 {
    let (_, mean, _) = welford_statistics(data);
    mean
}

/// Calculates the population variance from an array of data points in a numerically stable manner
/// using the Welford algorithm.
/// ```
/// use statistics::data::variance;
/// use approx_eq::assert_approx_eq;
///
/// let data1: Vec<f64> = vec![-0.2711336 ,  1.20002575,  0.69102151, -0.56390913, -1.62661382, -0.0613969 ,  0.39876752, -0.99619281,  1.12860854, -0.61163405];
/// assert_approx_eq!(variance(&data1), 0.7707231173572182);
///
/// let data2: Vec<f64> = vec![-1.35521905,  0.70316493, -0.24386284,  0.20382644,  1.28818114, -0.90003795, -0.73912347,  1.48550753,  1.02038191,  0.18684426];
/// assert_approx_eq!(variance(&data2), 0.8458540238604941);
/// ```
pub fn variance(data: &[f64]) -> f64 {
    let (count, _, m2) = welford_statistics(data);
    m2 / count as f64
}

/// Calculates the sample variance from an array of data points in a numerically stable manner
/// using the Welford algorithm.
/// ```
/// use statistics::data::sample_variance;
/// use approx_eq::assert_approx_eq;
///
/// let data1: Vec<f64> = vec![-0.2711336 ,  1.20002575,  0.69102151, -0.56390913, -1.62661382, -0.0613969 ,  0.39876752, -0.99619281,  1.12860854, -0.61163405];
/// assert_approx_eq!(sample_variance(&data1), 0.8563590181955176);
///
/// let data2: Vec<f64> = vec![-1.35521905,  0.70316493, -0.24386284,  0.20382644,  1.28818114, -0.90003795, -0.73912347,  1.48550753,  1.02038191,  0.18684426];
/// assert_approx_eq!(sample_variance(&data2), 0.939837803612305);
/// ```
pub fn sample_variance(data: &[f64]) -> f64 {
    let (count, _, m2) = welford_statistics(data);
    m2 / (count - 1) as f64
}
