#[cfg(feature = "blas")]
extern crate blas;
#[cfg(feature = "lapack")]
extern crate lapack;

#[cfg(all(target_os = "macos", feature = "blas", feature = "lapack"))]
extern crate accelerate_src;
#[cfg(all(not(target_os = "macos"), feature = "blas", feature = "lapack"))]
extern crate openblas_src;

#[cfg(feature = "blas")]
use blas::{ddot, dgemm};
#[cfg(feature = "lapack")]
use lapack::{dgesv, dgetrf, dgetri};

#[cfg(not(feature = "lapack"))]
use super::lu;

/// Check if matrix is square.
pub fn is_square(m: &[f64]) -> Result<usize, String> {
    let n = (m.len() as f32).sqrt();
    if n % 1. == 0. {
        Ok(n as usize)
    } else {
        Err("Matrix not square".to_string())
    }
}

/// Transpose a matrix.
pub fn transpose(a: &[f64], nrows: usize) -> Vec<f64> {
    assert!((a.len() / nrows) % 1 == 0, "shape not correct for a matrix");
    let ncols = a.len() / nrows;
    let mut at: Vec<f64> = Vec::with_capacity(a.len());
    for i in 0..nrows {
        for j in 0..ncols {
            at.push(a[j * nrows + i]);
        }
    }
    at
}

/// Given an n by n matrix, invert it. The resulting matrix is returned as a flattened array.
pub fn invert_matrix(matrix: &[f64]) -> Vec<f64> {
    let n = is_square(&matrix).unwrap();
    #[cfg(feature = "lapack")]
    {
        let n = n as i32;
        let mut a = matrix.to_vec();
        let mut ipiv = vec![0; n as usize];
        let mut info: i32 = 0;
        let lwork: i32 = 64 * n; // optimal size as given by lwork=-1
        let mut work = vec![0.; lwork as usize];
        unsafe {
            dgetrf(n, n, &mut a, n, &mut ipiv, &mut info);
            assert_eq!(info, 0, "dgetrf failed");
        }
        unsafe {
            dgetri(n, &mut a, n, &mut ipiv, &mut work, lwork, &mut info);
            assert_eq!(info, 0, "dgetri failed");
        }
        return a;
    }
    // maybe a little dumber than just implementing a solve function that solves all equations at
    // the same time but this is still 5x faster than lapack's inverse for 20x20 matrices
    #[cfg(not(feature = "lapack"))]
    {
        let mut ones = vec![0.; n];
        let mut inverse = vec![0.; n * n];
        let lup = lu(&matrix);

        for i in 0..n {
            ones[i] = 1.;
            let sol = lu_solve(&lup, &ones);
            assert_eq!(sol.len(), n);
            for j in 0..n {
                inverse[j * n + i] = sol[j];
            }
            ones[i] = 0.;
        }
        inverse
    }
}

/// Given a matrix X with k rows, return X transpose times X, which is a symmetric matrix.
pub fn xtx(x: &[f64], k: usize) -> Vec<f64> {
    #[cfg(feature = "blas")]
    {
        let k = k as i32;
        let n = x.len() as i32 / k; // should divide into it perfectly
        let mut result = vec![0.; (n * n) as usize];
        unsafe {
            dgemm(b'T', b'N', n, n, k, 1., x, k, x, k, 0., &mut result, n);
        }
        assert!(is_square(&result).is_ok());
        return result;
    }
    #[cfg(not(feature = "blas"))]
    matmul(x, x, k, k, true, false)
}

/// Solve the linear system Ax = b with LU decomposition.
pub fn solve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    assert!(a.len() == n * n);

    // this is slower than the non-lapack version
    #[cfg(feature = "lapack")]
    {
        let mut lu = transpose(&a, n);
        let mut ipiv = vec![0; n as usize];
        let mut result = b.to_vec();
        let mut info = 0;
        unsafe {
            dgesv(
                n as i32,
                1,
                &mut lu,
                n as i32,
                &mut ipiv,
                &mut result,
                n as i32,
                &mut info,
            );
            assert_eq!(info, 0, "dgesv failed");
        }
        return result;
    }

    #[cfg(not(feature = "lapack"))]
    lu_solve(&lu(&a), b)
}

/// Solve the linear system Ax = b given a LU decomposed matrix A. The first argument should be a
/// tuple, where the first element is the LU decomposed matrix and the second element is the pivots
/// P.
pub fn lu_solve(lup: &(Vec<f64>, Vec<i32>), b: &[f64]) -> Vec<f64> {
    let (lu, pivots) = lup;
    let n = b.len();
    assert!(lu.len() == n * n);

    let mut x = vec![0.; n];
    for i in 0..pivots.len() {
        x[i] = b[pivots[i] as usize];
    }

    for k in 0..n {
        for i in (k + 1)..n {
            x[i] -= x[k] * lu[i * n + k];
        }
    }

    for k in (0..n).rev() {
        x[k] /= lu[k * n + k];
        for i in 0..k {
            x[i] -= x[k] * lu[i * n + k];
        }
    }

    return x;
}

/// Multiply two matrices together, optionally transposing one or both of them. Note that the
/// matrices must be in column-major ordering.
pub fn matmul(
    a: &[f64],
    b: &[f64],
    rows_a: usize,
    rows_b: usize,
    transpose_a: bool,
    transpose_b: bool,
) -> Vec<f64> {
    #[cfg(feature = "blas")]
    {
        let rows_a = rows_a as i32;
        let rows_b = rows_b as i32;
        let cols_a = a.len() as i32 / rows_a;
        let cols_b = b.len() as i32 / rows_b;
        let trans_a = if transpose_a { b'T' } else { b'N' };
        let trans_b = if transpose_b { b'T' } else { b'N' };
        let m = if transpose_a { cols_a } else { rows_a as i32 };
        let n = if transpose_b { rows_b as i32 } else { cols_b };
        let k = if transpose_a { rows_a as i32 } else { cols_a };
        let alpha = 1.;
        let beta = 0.;
        let lda = rows_a as i32;
        let ldb = rows_b as i32;
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
        return c;
    }

    #[cfg(not(feature = "blas"))]
    {
        let cols_a = a.len() / rows_a;
        let cols_b = b.len() / rows_b;
        let m = if transpose_a { cols_a } else { rows_a };
        let l = if transpose_a { rows_a } else { cols_a };
        let n = if transpose_b { rows_b } else { cols_b };
        let mut c = vec![0.; m * n];

        // this is kind of dumb. TODO: figure out the indexing for transpose
        let a = if transpose_a {
            transpose(&a, rows_a)
        } else {
            a.to_vec()
        };
        let b = if transpose_b {
            transpose(&b, rows_b)
        } else {
            b.to_vec()
        };

        for i in 0..m {
            for k in 0..l {
                let temp = a[k * m + i];
                for j in 0..n {
                    c[j * m + i] += temp * b[j * l + k];
                }
            }
        }

        return c;
    }
}

/// Create a design matrix from a given matrix. Note that this follows column-major ordering, so
/// the resulting vector simply has some 1s appended to the front.
pub fn design(x: &[f64], rows: i32) -> Vec<f64> {
    let mut ones = vec![1.; rows as usize];
    ones.extend_from_slice(x);
    ones
}

/// Given some length m data x, create an nth order
/// [Vandermonde matrix](https://en.wikipedia.org/wiki/Vandermonde_matrix).
/// Note that this returns a vector with column-major ordering.
pub fn vandermonde(x: &[f64], n: usize) -> Vec<f64> {
    let mut vm = Vec::with_capacity(x.len() * n);

    for i in 0..n {
        for v in x.iter() {
            vm.push(v.powi(i as i32));
        }
    }

    vm
}

/// Given a vector of length n, creates n stacked duplicates, resulting in a square [Toeplitz
/// matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix). This function also assumes evenness.
/// That is, x_i = x_{-i}.
pub fn toeplitz(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut v = vec![0.; n * n];
    for i in 0..n as i32 {
        for j in 0..n as i32 {
            v[(i * n as i32 + j) as usize] = x[(i - j).abs() as usize];
        }
    }
    v
}

/// Calculates the dot product of two equal-length vectors. When the feature "blas" is enabled,
/// uses `ddot` from BLAS. Otherwise, uses a length-8 unrolled loop.
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());

    #[cfg(feature = "blas")]
    {
        unsafe {
            return ddot(x.len() as i32, x, 1, y, 1);
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        let n = x.len();
        let chunks = (n - (n % 8)) / 8;
        let mut s = 0.;

        // unroll as many as possible
        for i in 0..chunks {
            let idx = i * 8;
            assert!(n > idx + 7);
            s += x[idx] * y[idx]
                + x[idx + 1] * y[idx + 1]
                + x[idx + 2] * y[idx + 2]
                + x[idx + 3] * y[idx + 3]
                + x[idx + 4] * y[idx + 4]
                + x[idx + 5] * y[idx + 5]
                + x[idx + 6] * y[idx + 6]
                + x[idx + 7] * y[idx + 7];
        }

        // do the rest
        for j in (chunks * 8)..n {
            s += x[j] * y[j];
        }

        return s;
    }
}

/// Calculates the norm of a vector.
pub fn norm(x: &[f64]) -> f64 {
    dot(&x, &x).sqrt()
}
