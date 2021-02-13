/// Implements the discretized [Heaviside step function](https://en.wikipedia.org/wiki/Heaviside_step_function).
pub fn heaviside(n: usize) -> Vec<f64> {
    let mut h = vec![1.; n];
    h[0] = 0.5;
    h
}

/// Implements the discretized [Delta
/// function](https://en.wikipedia.org/wiki/Dirac_delta_function).
pub fn delta(n: usize, dt: f64) -> Vec<f64> {
    let mut d = vec![0.; n];
    d[0] = 1. / dt;
    d
}
