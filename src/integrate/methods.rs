pub fn trapz<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let dx: f64 = (b - a) / (n as f64);
    dx * ((0..n).map(|k| f(a + k as f64 * dx)).sum::<f64>() + (f(b) + f(a)) / 2.)
}
