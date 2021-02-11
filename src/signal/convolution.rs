use crate::utils::dot;

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
    // weight.iter_mut().for_each(|x| *x *= dt);

    (0..(f.len() + m - 1))
        .into_iter()
        .map(|i| dot(&signal[i..(i + m)], &weight) * dt)
        .collect()
}
