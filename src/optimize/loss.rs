use crate::summary::mean;

/// Calculates the mean squared error of a vector of predictions.
/// See https://en.wikipedia.org/wiki/Mean_squared_error
pub fn mse(predicted: Vec<f64>, observed: Vec<f64>) -> f64 {
    mean(
        &(0..predicted.len())
            .into_iter()
            .map(|i| (predicted[i] - observed[i]).powi(2))
            .collect::<Vec<_>>(),
    )
}
