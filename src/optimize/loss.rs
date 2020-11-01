//! Various functions for computing loss/cost functions.

use crate::summary::mean;

/// Calculates the mean squared error of a vector of predictions.
/// See https://en.wikipedia.org/wiki/Mean_squared_error
pub fn mse(predicted: &[f64], observed: &[f64]) -> f64 {
    mean(
        &(0..predicted.len())
            .into_iter()
            .map(|i| (predicted[i] - observed[i]).powi(2))
            .collect::<Vec<_>>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predict::*;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_mse() {
        let a1 = vec![
            6.37389029, 2.97393581, 7.41407052, 8.27988358, 8.02743405, 2.47355454, 3.80182034,
            6.41040071, 6.13588181, 7.35956359,
        ];
        let a2 = vec![
            5.66858702, 4.13987754, 6.00940185, 4.37266015, 2.73021164, 5.5139846, 5.11833522,
            4.19841612, 3.65598817, 5.95169814,
        ];
        assert_approx_eq!(mse(&a1, &a2), 7.115918683570368);

        let x = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let y = vec![
            5.802896, 7.802896, 9.802896, 11.802896, 13.802896, 15.802896, 17.802896, 19.802896,
            21.802896, 23.802896,
        ];
        let slr = PolynomialRegressor::new(&[5., 2.]);
        assert_approx_eq!(mse(&slr.predict(&x), &y), 0.6446419891202162);
    }
}
