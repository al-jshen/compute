extern crate blas;
extern crate lapack;
extern crate openblas_src;
use blas::dgemm;
use lapack::{dgetrf, dgetri};

/// Given an n by n matrix, invert it. The resulting matrix is returned as a flattened array.
pub fn invert_matrix(matrix: &[f64]) -> Vec<f64> {
    let n = (matrix.len() as f64).sqrt() as i32; // should divide into it perfectly
    let mut a = matrix.to_vec();
    let mut ipiv = vec![0; n as usize];
    let mut info: i32 = 0;
    let lwork: i32 = 64 * n; // optimal size as given by lwork=1
    let mut work = vec![0.; lwork as usize];
    // Matrix inversion
    unsafe {
        dgetrf(n, n, &mut a, n, &mut ipiv, &mut info);
        assert_eq!(info, 0, "dgetrf failed");
    }
    unsafe {
        dgetri(n, &mut a, n, &mut ipiv, &mut work, lwork, &mut info);
        assert_eq!(info, 0, "dgetri failed");
    }
    a
}

/// Given a matrix X with k rows, return X transpose times X, which is a symmetric matrix.
pub fn xtx(x: &[f64], k: i32) -> Vec<f64> {
    let n = x.len() as i32 / k; // should divide into it perfectly
    let mut result = vec![0.; (n * n) as usize];
    unsafe {
        dgemm(b'T', b'N', n, n, k, 1., x, k, x, k, 0., &mut result, n);
    }
    result
}

/// Multiply two matrices together, optionally transposing one or both of them.
pub fn matmul(
    a: &[f64],
    b: &[f64],
    rows_a: i32,
    rows_b: i32,
    transpose_a: bool,
    transpose_b: bool,
) -> Vec<f64> {
    let cols_a = a.len() as i32 / rows_a;
    let cols_b = b.len() as i32 / rows_b;
    let trans_a = if transpose_a { b'T' } else { b'N' };
    let trans_b = if transpose_b { b'T' } else { b'N' };
    let m = if transpose_a { cols_a } else { rows_a };
    let n = if transpose_b { rows_b } else { cols_b };
    let k = if transpose_a { rows_a } else { cols_a };
    let alpha = 1.;
    let beta = 0.;
    let lda = rows_a;
    let ldb = rows_b;
    let ldc = m;
    if transpose_a {
        assert!(lda >= k, "lda={} must be at least as large as k={}", lda, k);
    } else {
        assert!(lda >= m, "lda={} must be at least as large as m={}", lda, m);
    }
    if transpose_b {
        assert!(ldb >= n, "ldb={} must be at least as large as n={}", ldb, n);
    } else {
        assert!(ldb >= k, "ldb={} must be at least as large as k={}", ldb, k);
    }
    let mut c = vec![0.; (ldc * n) as usize];
    unsafe {
        dgemm(
            trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, &mut c, ldc,
        );
    }
    c
}

/// Create a design matrix from a given matrix.
pub fn design(x: &[f64], rows: i32) -> Vec<f64> {
    let mut ones = vec![1.; rows as usize];
    ones.extend_from_slice(x);
    ones
}

/// Given a vector of length n, creates n stacked duplicates, resulting in a square [Toeplitz
/// matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix). This function also assumes evenness.
/// That is, x_i = x_{-i}.
pub fn toeplitz_even_square(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut v = vec![0.; n * n];
    for i in 0..n as i32 {
        for j in 0..n as i32 {
            v[(i * n as i32 + j) as usize] = x[(i - j).abs() as usize];
        }
    }
    v
}
