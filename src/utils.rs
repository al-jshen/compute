extern crate openblas_src;
use lapack::{dgetrf, dgetri};

pub fn invert_matrix(matrix: &[f64], n: i32, m: i32) -> Vec<f64> {
    assert_eq!(matrix.len() as i32, n * m);
    let mut a = matrix.to_vec();
    let mut ipiv = vec![0; 3];
    let mut info: i32 = 0;
    unsafe { dgetrf(m, n, &mut a, n, &mut ipiv, &mut info) }
    let lwork: i32 = n.pow(2);
    let mut work = vec![0.; lwork as usize];
    unsafe { dgetri(n, &mut a, n, &ipiv, &mut work, lwork, &mut info) }
    a
}
