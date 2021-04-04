use crate::linalg::{dot, is_square};

/// Solve a matrix equation of the form Lx=b, where L is a lower triangular matrix.
/// See the [Wikipedia page](https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution).
pub fn forward_substitution(l: &[f64], b: &[f64]) -> Vec<f64> {
    let n = is_square(l).unwrap();
    assert_eq!(b.len(), n);
    let mut x = vec![0.; n];
    for i in 0..n {
        x[i] = (b[i] - dot(&l[(i * n)..(i * n + i)], &x[..i])) / l[i * n + i];
    }
    x
}

/// Solve a matrix equation of the form Ux=b, where U is an upper triangular matrix.
/// See the [Wikipedia page](https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution).
pub fn backward_substitution(u: &[f64], b: &[f64]) -> Vec<f64> {
    let n = is_square(u).unwrap();
    assert_eq!(b.len(), n);
    let mut x = vec![0.; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - dot(&u[(i * n + i + 1)..(i * n + n)], &x[i + 1..])) / u[i * n + i];
    }
    x
}
