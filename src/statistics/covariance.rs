use crate::statistics::mean;

/// Calculates the covariance between two vectors x and y. This is a two-pass algorithm which
/// centers the data before computing the covariance, which improves stability but does not
/// change the result as covariance is invariant with respect to shifts.
pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mean_x = mean(&x);
    let mean_y = mean(&y);
    let n = x.len();

    (0..n)
        .into_iter()
        .map(|i| (x[i] - mean_x) * (y[i] - mean_y))
        .sum::<f64>()
        / n as f64
}
///
/// Calculates the sample covariance between two vectors x and y. This is a two-pass algorithm which
/// centers the data before computing the covariance, which improves stability but does not
/// change the result as covariance is invariant with respect to shifts.
pub fn sample_covariance(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mean_x = mean(&x);
    let mean_y = mean(&y);
    let n = x.len();

    (0..n)
        .into_iter()
        .map(|i| (x[i] - mean_x) * (y[i] - mean_y))
        .sum::<f64>()
        / (n - 1) as f64
}

/// Calculates the covariance between two vectors x and y. This is a one-pass algorithm which
/// shifts the data by the first element in each vector before computing the covariance,
/// which improves stability but does not change the result as covariance is invariant with respect to shifts.
pub fn sample_covariance_onepass(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    (0..n)
        .into_iter()
        .map(|i| (x[i] - x[0]) * (y[i] - y[0]))
        .sum::<f64>()
        / (n - 1) as f64
}

/// Calculates the covariance between two vectors x and y. This is a stable one-pass online algorithm.
/// See <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Covariance>
pub fn sample_covariance_online(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut meanx = 0.;
    let mut meany = 0.;
    let mut c = 0.;
    let mut n = 0.;

    for (i, j) in x.iter().zip(y.iter()) {
        n += 1.;
        let dx = i - meanx;
        let dy = j - meany;
        meanx += dx / n;
        meany += dy / n;
        c += dx * dy;
    }

    c / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_covariance() {
        let x1: Vec<f64> = vec![-2.1, -1., 4.3];
        let y1: Vec<f64> = vec![3., 1.1, 0.12];
        assert_approx_eq!(covariance(&x1, &y1), -2.8573333);
        assert_approx_eq!(sample_covariance(&x1, &y1), -4.286);

        let x2: Vec<f64> = vec![1.1, 1.7, 2.1, 1.4, 0.2];
        let y2: Vec<f64> = vec![3.0, 4.2, 4.9, 4.1, 2.5];
        assert_approx_eq!(covariance(&x2, &y2), 0.532);
        assert_approx_eq!(sample_covariance(&x2, &y2), 0.665);
    }
}
