pub fn heaviside(n: usize) -> Vec<f64> {
    let mut h = vec![1.; n];
    h[0] = 0.5;
    h
}

pub fn delta(n: usize, dt: f64) -> Vec<f64> {
    let mut d = vec![0.; n];
    d[0] = 1. / dt;
    d
}
