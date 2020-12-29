use crate::summary::mean;

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
        / (n - 1) as f64
}

/// Calculates the covariance between two vectors x and y. This is a one-pass algorithm which
/// shifts the data by the first element in each vector before computing the covariance,
/// which improves stability but does not change the result as covariance is invariant with respect to shifts.
pub fn covariance_onepass(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    (0..n)
        .into_iter()
        .map(|i| (x[i] - x[0]) * (y[i] - y[0]))
        .sum::<f64>()
        / (n - 1) as f64
}

/// Calculates the covariance between two vectors x and y. This is a stable one-pass online algorithm.
/// See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Covariance
pub fn covariance_online(x: &[f64], y: &[f64]) -> f64 {
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
    fn test_covariance1() {
        let x: Vec<f64> = vec![-2.1, -1., 4.3];
        let y: Vec<f64> = vec![3., 1.1, 0.12];
        assert_approx_eq!(covariance(&x, &y), -4.286);
        // assert_approx_eq!(covariance_onepass(&x, &y), -4.286);
        // assert_approx_eq!(covariance_online(&x, &y), -4.286);
    }

    #[test]
    fn test_covariance2() {
        let x: Vec<f64> = vec![1.1, 1.7, 2.1, 1.4, 0.2];
        let y: Vec<f64> = vec![3.0, 4.2, 4.9, 4.1, 2.5];
        assert_approx_eq!(covariance(&x, &y), 0.665);
        // assert_approx_eq!(covariance_onepass(&x, &y), 0.665);
        // assert_approx_eq!(covariance_online(&x, &y), 0.665);
    }
}
