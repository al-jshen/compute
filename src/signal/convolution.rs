use crate::linalg::dot;

/// Performs a discrete linear
/// [convolution](https://en.wikipedia.org/wiki/Convolution)
/// of two 1D arrays. Swaps arrays `f` and `w` if `w` is longer than `f`.
/// The resulting convolution has length `n+m-1`, where `n` is the length
/// of `f` and `m` is the length of `w`.
pub fn convolve(f: &[f64], w: &[f64], dt: f64) -> Vec<f64> {
    let (mut a, mut weight) = if w.len() > f.len() {
        (w.to_vec(), f.to_vec())
    } else {
        (f.to_vec(), w.to_vec())
    };

    let m = w.len();

    let mut signal = vec![0.; m - 1];
    a.extend(&signal);
    signal.extend(&a);

    weight.reverse();

    (0..(f.len() + m - 1))
        .into_iter()
        .map(|i| dot(&signal[i..(i + m)], &weight) * dt)
        .collect()
}
