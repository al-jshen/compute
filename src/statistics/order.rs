//! A module for computing order statistics. This includes medians, quantiles, and extrema.

/// Returns the smallest element in the array.
pub fn min(data: &[f64]) -> f64 {
    data.iter().fold(f64::NAN, |acc, i| f64::min(acc, *i))
}

/// Returns the largest element in the array.
pub fn max(data: &[f64]) -> f64 {
    data.iter().fold(f64::NAN, |acc, i| f64::max(acc, *i))
}

/// Returns the index of the smallest element in the array.
pub fn argmin(data: &[f64]) -> usize {
    data.iter()
        .enumerate()
        .fold(
            (0, f64::MAX),
            |acc, (i, j)| if acc.1 > *j { (i, *j) } else { acc },
        )
        .0
}

/// Returns the index of the largest element in the array.
pub fn argmax(data: &[f64]) -> usize {
    data.iter()
        .enumerate()
        .fold(
            (0, f64::MIN),
            |acc, (i, j)| if acc.1 < *j { (i, *j) } else { acc },
        )
        .0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_min() {
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
        assert_approx_eq!(min(&data1), -1.62661382);

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
        assert_approx_eq!(min(&data2), -1.35521905);
    }

    #[test]
    fn test_argmin() {
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
        assert_eq!(argmin(&data1), 4);

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
        assert_eq!(argmin(&data2), 0);
    }

    #[test]
    fn test_max() {
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
        assert_approx_eq!(max(&data1), 1.20002575);

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
        assert_approx_eq!(max(&data2), 1.48550753);
    }

    #[test]
    fn test_argmax() {
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
        assert_eq!(argmax(&data1), 1);

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
        assert_eq!(argmax(&data2), 7);
    }
}
