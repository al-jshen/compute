//! Provides general linear algebra utilities like dot products, matrix multiplication, matrix
//! inversion, and the creation of Vandermonde matrices. Matrix operations are generally performed on 1D
//! arrays with the number of rows specified, whether the matrix is in row-major ordering. If the
//! `blas` and/or `lapack` features are enabled, then those will be used. Otherwise, native Rust
//! alternatives are available as defaults.

#[cfg(feature = "blas")]
extern crate blas;
#[cfg(feature = "lapack")]
extern crate lapack;

#[cfg(all(target_os = "macos", any(feature = "blas", feature = "lapack")))]
extern crate accelerate_src;
#[cfg(all(not(target_os = "macos"), any(feature = "blas", feature = "lapack")))]
extern crate openblas_src;

#[cfg(feature = "blas")]
use blas::{ddot, dgemm};
#[cfg(feature = "lapack")]
use lapack::{dgesv, dgetrf, dgetri};

#[cfg(not(feature = "lapack"))]
use crate::linalg::decomposition::lu::*;
use crate::prelude::max;

/// Generates evenly spaced values within a given interval. Values generated in the half-open
/// interval [start, stop). That is, the stop point is not included.
pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = (stop - start) / step + 1.;
    (0..n as usize)
        .map(|i| start as f64 + i as f64 * step)
        .collect::<Vec<_>>()
}

/// Generates evenly spaced values within a given interval, with a set number of points. Both the
/// start and stop points are included.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let width = (stop - start) / (num - 1) as f64;
    (0..num)
        .map(|i| start + i as f64 * width)
        .collect::<Vec<_>>()
}

/// Checks whether a 1D array is a valid square matrix.
pub fn is_square(m: &[f64]) -> Result<usize, String> {
    let n = (m.len() as f32).sqrt();
    if n % 1. == 0. {
        Ok(n as usize)
    } else {
        Err("Matrix not square".to_string())
    }
}

/// Checks whether a 1D array is a valid matrix representation given the number of rows.
pub fn is_matrix(m: &[f64], nrows: usize) -> Result<usize, String> {
    let ncols = m.len() / nrows;
    if nrows * ncols == m.len() {
        Ok(ncols)
    } else {
        Err("Not a matrix".to_string())
    }
}

/// Checks whether a 1D array is a valid design matrix, given its number of rows.
pub fn is_design(m: &[f64], nrows: usize) -> bool {
    let mut is_design = true;
    let ncols = is_matrix(m, nrows).unwrap();
    for i in 0..nrows {
        if m[i * ncols] != 1. {
            is_design = false;
        }
    }
    is_design
}

/// Convert a 1D matrix from row-major ordering into column-major ordering.
pub fn row_to_col_major(a: &[f64], nrows: usize) -> Vec<f64> {
    let ncols = is_matrix(a, nrows).unwrap();
    let mut x = a.to_vec();
    for i in 0..nrows {
        for j in 0..ncols {
            x[j * nrows + i] = a[i * ncols + j];
        }
    }
    x
}

/// Convert a 1D matrix from column-major ordering into row-major ordering.
pub fn col_to_row_major(a: &[f64], nrows: usize) -> Vec<f64> {
    let ncols = is_matrix(a, nrows).unwrap();
    let mut x = a.to_vec();
    for i in 0..nrows {
        for j in 0..ncols {
            x[i * ncols + j] = a[j * nrows + i];
        }
    }
    x
}

/// Transpose a matrix.
pub fn transpose(a: &[f64], nrows: usize) -> Vec<f64> {
    let ncols = is_matrix(&a, nrows).unwrap();

    let mut at: Vec<f64> = Vec::with_capacity(a.len());

    for j in 0..ncols {
        for i in 0..nrows {
            at.push(a[i * ncols + j]);
        }
    }

    at
}

/// Extract the diagonal elements of a matrix.
pub fn diag(a: &[f64]) -> Vec<f64> {
    let n = is_square(a).unwrap();
    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        results.push(a[i * n + i]);
    }
    results
}

/// Create a diagonal matrix with the given elements along the elements.
pub fn diag_matrix(a: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut new = vec![0.; n * n];
    for i in 0..n {
        new[i * n + i] = a[i];
    }
    new
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
        let mut work = vec![0.; 1];
        unsafe {
            dgetri(n, &mut a, n, &ipiv, &mut work, -1, &mut info);
            assert_eq!(info, 0, "dgetri failed");
        }
        let lwork = work[0] as usize;
        work.extend_from_slice(&vec![0.; lwork - 1]);
        unsafe {
            dgetrf(n, n, &mut a, n, &mut ipiv, &mut info);
            assert_eq!(info, 0, "dgetrf failed");
        }
        unsafe {
            dgetri(n, &mut a, n, &ipiv, &mut work, lwork as i32, &mut info);
            assert_eq!(info, 0, "dgetri failed");
        }
        a
    }

    #[cfg(not(feature = "lapack"))]
    {
        let mut ones = vec![0.; n];
        let mut inverse = vec![0.; n * n];
        let (lu, piv) = lu(&matrix);

        for i in 0..n {
            ones[i] = 1.;
            let sol = lu_solve(&lu, &piv, &ones);
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
    matmul(x, x, k, k, true, false)
}

/// Solves a system of linear scalar equations. `a` must represent a square matrix.
pub fn solve_sys(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = is_square(a).unwrap();
    let nsys = is_matrix(b, n).unwrap();

    #[cfg(feature = "lapack")]
    {
        let mut A = row_to_col_major(&a, n);
        let mut B = row_to_col_major(&b, n);
        let mut ipiv = vec![0; n as usize];
        let mut info = 0;
        unsafe {
            dgesv(
                n as i32,
                nsys as i32,
                &mut A,
                n as i32,
                &mut ipiv,
                &mut B,
                n as i32,
                &mut info,
            );
            assert_eq!(info, 0, "dgesv failed");
        }
        col_to_row_major(&B, n)
    }

    #[cfg(not(feature = "lapack"))]
    {
        let mut solutions = Vec::with_capacity(b.len());
        let (lu, piv) = lu(a);
        let B = row_to_col_major(&b, n);

        for i in 0..nsys {
            let sol = lu_solve(&lu, &piv, &B[(i * n)..((i + 1) * n)]);
            assert_eq!(sol.len(), n);
            solutions.extend_from_slice(&sol);
        }
        col_to_row_major(&solutions, n)
    }
}

/// Solve the linear system Ax = b with LU decomposition.
pub fn solve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    assert!(a.len() == n * n);

    #[cfg(feature = "lapack")]
    {
        let mut lu = row_to_col_major(&a, n);
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
        result
    }

    #[cfg(not(feature = "lapack"))]
    {
        let (lu, piv) = lu(&a);
        lu_solve(&lu, &piv, b)
    }
}

/// Performs blocked matrix multiplication with block size `bsize`. See the API for the
/// [matmul](crate::linalg::matmul).
pub fn matmul_blocked(
    a: &[f64],
    b: &[f64],
    rows_a: usize,
    rows_b: usize,
    transpose_a: bool,
    transpose_b: bool,
    bsize: usize,
) -> Vec<f64> {
    let cols_a = is_matrix(a, rows_a).unwrap();
    let cols_b = is_matrix(b, rows_b).unwrap();

    let m = if transpose_a { cols_a } else { rows_a };
    let l = if transpose_a { rows_a } else { cols_a };
    let n = if transpose_b { rows_b } else { cols_b };

    let mut c = vec![0.; m * n];

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

    // https://courses.engr.illinois.edu/cs232/sp2009/lectures/X18.pdf
    for jj in 0..(n / bsize + 1) {
        for kk in 0..(l / bsize + 1) {
            for i in 0..m {
                for k in (kk * bsize)..std::cmp::min((kk * bsize) + bsize, l) {
                    let temp = a[i * l + k];
                    for j in (jj * bsize)..std::cmp::min((jj * bsize) + bsize, n) {
                        c[i * n + j] += temp * b[k * n + j];
                    }
                }
            }
        }
    }

    c
}

/// Multiply two matrices together, optionally transposing one or both of them.
pub fn matmul(
    a: &[f64],
    b: &[f64],
    rows_a: usize,
    rows_b: usize,
    transpose_a: bool,
    transpose_b: bool,
) -> Vec<f64> {
    let cols_a = is_matrix(a, rows_a).unwrap();
    let cols_b = is_matrix(b, rows_b).unwrap();

    #[cfg(feature = "blas")]
    {
        // some swapping to use row-major ordering
        let (cols_a, rows_a) = (rows_a, cols_a);
        let (cols_b, rows_b) = (rows_b, cols_b);

        let (transpose_a, transpose_b) = (!transpose_a, !transpose_b);

        let m = if transpose_a { cols_a } else { rows_a };
        let n = if transpose_b { rows_b } else { cols_b };
        let k = if transpose_a { rows_a } else { cols_a };

        let trans_a = if transpose_a { b'T' } else { b'N' };
        let trans_b = if transpose_b { b'T' } else { b'N' };

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
                trans_a, trans_b, m as i32, n as i32, k as i32, alpha, a, lda as i32, b,
                ldb as i32, beta, &mut c, ldc as i32,
            );
        }

        // this is expensive?
        transpose(&c, n)
    }

    #[cfg(not(feature = "blas"))]
    {
        let m = if transpose_a { cols_a } else { rows_a };
        let l = if transpose_a { rows_a } else { cols_a };
        let n = if transpose_b { rows_b } else { cols_b };

        let mut c = vec![0.; m * n];

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
                let temp = a[i * l + k];
                for j in 0..n {
                    c[i * n + j] += temp * b[k * n + j];
                }
            }
        }

        c
    }
}

/// Create a design matrix from a given matrix.
pub fn design(x: &[f64], rows: usize) -> Vec<f64> {
    let mut ones = vec![1.; rows];
    ones.extend_from_slice(x);
    col_to_row_major(&ones, rows)
}

/// Given some length m data x, create an nth order
/// [Vandermonde matrix](https://en.wikipedia.org/wiki/Vandermonde_matrix).
pub fn vandermonde(x: &[f64], n: usize) -> Vec<f64> {
    let mut vm = Vec::with_capacity(x.len() * n);

    for v in x.iter() {
        for i in 0..n {
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
        unsafe { ddot(x.len() as i32, x, 1, y, 1) }
    }

    #[cfg(not(feature = "blas"))]
    {
        let n = x.len();
        let chunks = (n - (n % 8)) / 8;
        let mut s = 0.;

        // unroll
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

        s
    }
}

/// Vector-vector operations.
#[macro_export]
macro_rules! impl_vops_binary {
    ($opname: ident, $op:tt) => {
        pub fn $opname(v1: &[f64], v2: &[f64]) -> Vec<f64> {
            assert_eq!(v1.len(), v2.len());
            let n = v1.len();

            let mut v = vec![0.; n];
            let chunks = (n - (n % 8)) / 8;

            // unroll
            for i in 0..chunks {
                let idx = i * 8;
                assert!(n > idx + 7);
                v[idx] = v1[idx] $op v2[idx];
                v[idx + 1] = v1[idx + 1] $op v2[idx + 1];
                v[idx + 2] = v1[idx + 2] $op v2[idx + 2];
                v[idx + 3] = v1[idx + 3] $op v2[idx + 3];
                v[idx + 4] = v1[idx + 4] $op v2[idx + 4];
                v[idx + 5] = v1[idx + 5] $op v2[idx + 5];
                v[idx + 6] = v1[idx + 6] $op v2[idx + 6];
                v[idx + 7] = v1[idx + 7] $op v2[idx + 7];
            }

            // do the rest
            for j in (chunks * 8)..n {
                v[j] = v1[j] $op v2[j];
            }

            v
        }
    }
}

impl_vops_binary!(vadd, +);
impl_vops_binary!(vsub, -);
impl_vops_binary!(vmul, *);
impl_vops_binary!(vdiv, /);

/// Single vector operations.
#[macro_export]
macro_rules! impl_vops_unary {
    ($opname: ident, $op:ident) => {
        pub fn $opname(v1: &[f64]) -> Vec<f64> {
            let n = v1.len();

            let mut v = vec![0.; n];
            let chunks = (n - (n % 8)) / 8;

            // unroll
            for i in 0..chunks {
                let idx = i * 8;
                assert!(n > idx + 7);
                v[idx] = v1[idx].$op();
                v[idx + 1] = v1[idx + 1].$op();
                v[idx + 2] = v1[idx + 2].$op();
                v[idx + 3] = v1[idx + 3].$op();
                v[idx + 4] = v1[idx + 4].$op();
                v[idx + 5] = v1[idx + 5].$op();
                v[idx + 6] = v1[idx + 6].$op();
                v[idx + 7] = v1[idx + 7].$op();
            }

            // do the rest
            for j in (chunks * 8)..n {
                v[j] = v1[j].$op();
            }

            v
        }
    };
}

impl_vops_unary!(vln, ln);
impl_vops_unary!(vln1p, ln_1p);
impl_vops_unary!(vlog10, log10);
impl_vops_unary!(vlog2, log2);
impl_vops_unary!(vexp, exp);
impl_vops_unary!(vexp2, exp2);
impl_vops_unary!(vsin, sin);
impl_vops_unary!(vcos, cos);
impl_vops_unary!(vtan, tan);
impl_vops_unary!(vsinh, sinh);
impl_vops_unary!(vcosh, cosh);
impl_vops_unary!(vtanh, tanh);
impl_vops_unary!(vasin, asin);
impl_vops_unary!(vacos, acos);
impl_vops_unary!(vatan, atan);
impl_vops_unary!(vasinh, asinh);
impl_vops_unary!(vacosh, acosh);
impl_vops_unary!(vatanh, atanh);
impl_vops_unary!(vsqrt, sqrt);
impl_vops_unary!(vcbrt, cbrt);
impl_vops_unary!(vabs, abs);
impl_vops_unary!(vfloor, floor);
impl_vops_unary!(vceil, ceil);
impl_vops_unary!(vtoradians, to_radians);
impl_vops_unary!(vtodegrees, to_degrees);
impl_vops_unary!(vrecip, recip);

/// Vector-scalar operations.
#[macro_export]
macro_rules! impl_vsops {
        ($opname: ident, $op:tt) => {
            pub fn $opname(v1: &[f64], scalar: f64) -> Vec<f64> {
                let n = v1.len();

                let mut v = vec![0.; n];
                let chunks = (n - (n % 8)) / 8;

                // unroll
                for i in 0..chunks {
                    let idx = i * 8;
                    assert!(n > idx + 7);
                    v[idx] = v1[idx] $op scalar;
                    v[idx + 1] = v1[idx + 1] $op scalar;
                    v[idx + 2] = v1[idx + 2] $op scalar;
                    v[idx + 3] = v1[idx + 3] $op scalar;
                    v[idx + 4] = v1[idx + 4] $op scalar;
                    v[idx + 5] = v1[idx + 5] $op scalar;
                    v[idx + 6] = v1[idx + 6] $op scalar;
                    v[idx + 7] = v1[idx + 7] $op scalar;
                }

                // do the rest
                for j in (chunks * 8)..n {
                    v[j] = v1[j] $op scalar;
                }

                v
            }
        }
    }

impl_vsops!(vsadd, +);
impl_vsops!(vssub, -);
impl_vsops!(vsmul, *);
impl_vsops!(vsdiv, /);

/// Scalar-vector operations.
#[macro_export]
macro_rules! impl_svops {
    ($opname: ident, $op:tt) => {
        pub fn $opname(scalar: f64, v1: &[f64]) -> Vec<f64> {
            let n = v1.len();

            let mut v = vec![0.; n];
            let chunks = (n - (n % 8)) / 8;

            // unroll
            for i in 0..chunks {
                let idx = i * 8;
                assert!(n > idx + 7);
                v[idx] = scalar $op v1[idx];
                v[idx + 1] = scalar $op v1[idx + 1];
                v[idx + 2] = scalar $op v1[idx + 2];
                v[idx + 3] = scalar $op v1[idx + 3];
                v[idx + 4] = scalar $op v1[idx + 4];
                v[idx + 5] = scalar $op v1[idx + 5];
                v[idx + 6] = scalar $op v1[idx + 6];
                v[idx + 7] = scalar $op v1[idx + 7];
            }

            // do the rest
            for j in (chunks * 8)..n {
                v[j] = scalar $op v1[j];
            }

            v
        }
    }
}

impl_svops!(svadd, +);
impl_svops!(svsub, -);
impl_svops!(svmul, *);
impl_svops!(svdiv, /);

/// Calculates the norm of a vector.
pub fn norm(x: &[f64]) -> f64 {
    dot(&x, &x).sqrt()
}

/// Calculates the infinity norm of a matrix. That is, it sums the absolute values along each row,
/// and then returns the largest of these values.
pub fn inf_norm(x: &[f64], nrows: usize) -> f64 {
    let ncols = is_matrix(x, nrows).unwrap();
    let mut abs_row_sums = Vec::with_capacity(nrows);
    for i in 0..nrows {
        let mut s = 0.;
        for j in 0..ncols {
            s += x[i * ncols + j].abs();
        }
        abs_row_sums.push(s);
    }
    max(&abs_row_sums)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_invert() {
        let x = [
            -0.46519316,
            -3.1042875,
            -5.01766541,
            -1.86300107,
            2.7692825,
            2.3097699,
            -12.3854289,
            -8.70520295,
            6.02201052,
            -6.71212792,
            -1.74683781,
            -6.08893455,
            -2.53731118,
            2.72112893,
            4.70204472,
            -1.03387848,
        ];
        let inv = invert_matrix(&x);

        let inv_ref = [
            -0.25572126,
            0.03156201,
            0.06146028,
            -0.16691749,
            -0.16856104,
            0.07197315,
            -0.0498292,
            -0.00880639,
            -0.05192178,
            -0.033113,
            0.0482877,
            0.08798427,
            -0.0522019,
            -0.03862469,
            -0.06237155,
            -0.18061673,
        ];

        for i in 0..x.len() {
            assert_approx_eq!(inv[i], inv_ref[i]);
        }
    }

    #[test]
    fn test_matmul() {
        let x = [
            7., 2., 6., 5., 5., 5., 3., 9., 2., 2., 3., 9., 7., 9., 7., 8., 2., 7., 4., 5.,
        ];
        let nrows = 4;
        let ncols = is_matrix(&x, nrows).unwrap();
        assert_eq!(ncols, 5);

        let xtx1 = matmul(&x, &x, 4, 4, true, false);
        let xtx2 = matmul(&transpose(&x, 4), &x, 5, 4, false, false);
        let xtx1b = matmul_blocked(&x, &x, 4, 4, true, false, 4);
        let xtx2b = matmul_blocked(&transpose(&x, 4), &x, 5, 4, false, false, 4);

        let y = vec![5., 5., 4., 6., 8., 5., 6., 4., 3., 6.];
        let xty1 = matmul(&x, &y, 4, 5, false, false);
        let xty2 = matmul(&y, &x, 2, 4, false, true);
        let xty1b = matmul_blocked(&x, &y, 4, 5, false, false, 4);
        let xty2b = matmul_blocked(&y, &x, 2, 4, false, true, 4);

        assert_eq!(
            xtx1,
            vec![
                147., 72., 164., 104., 106., 72., 98., 116., 105., 89., 164., 116., 215., 139.,
                132., 104., 105., 139., 126., 112., 106., 89., 132., 112., 103.
            ]
        );
        assert_eq!(xtx1, xtx2);
        assert_eq!(xtx1, xtx1b);
        assert_eq!(xtx2, xtx2b);
        assert_eq!(xtx1b, xtx2b);
        assert_eq!(xty1, vec![136., 127., 127., 108., 182., 182., 143., 133.]);
        assert_eq!(xty2, vec![139., 104., 198., 142., 116., 97., 166., 122.]);
        assert_eq!(xty1, xty1b);
        assert_eq!(xty2, xty2b);
    }

    #[test]
    fn test_solve() {
        let A = vec![
            -0.46519316,
            -3.1042875,
            -5.01766541,
            -1.86300107,
            2.7692825,
            2.3097699,
            -12.3854289,
            -8.70520295,
            6.02201052,
            -6.71212792,
            -1.74683781,
            -6.08893455,
            -2.53731118,
            2.72112893,
            4.70204472,
            -1.03387848,
        ];
        let b = vec![-4.13075599, -1.28124453, 4.65406058, 3.69106842];

        let x = solve(&A, &b);
        let x_ref = vec![0.68581948, 0.33965616, 0.8063919, -0.69182874];

        for i in 0..4 {
            assert_approx_eq!(x[i], x_ref[i]);
        }
    }

    #[test]
    fn test_linspace() {
        let r1 = vec![2., 2.4, 2.8, 3.2, 3.6, 4.];
        let e1 = linspace(2., 4., 6);
        for i in 0..6 {
            assert_approx_eq!(r1[i], e1[i]);
        }

        let r2 = vec![
            -50.,
            -48.47457627,
            -46.94915254,
            -45.42372881,
            -43.89830508,
            -42.37288136,
            -40.84745763,
            -39.3220339,
            -37.79661017,
            -36.27118644,
            -34.74576271,
            -33.22033898,
            -31.69491525,
            -30.16949153,
            -28.6440678,
            -27.11864407,
            -25.59322034,
            -24.06779661,
            -22.54237288,
            -21.01694915,
            -19.49152542,
            -17.96610169,
            -16.44067797,
            -14.91525424,
            -13.38983051,
            -11.86440678,
            -10.33898305,
            -8.81355932,
            -7.28813559,
            -5.76271186,
            -4.23728814,
            -2.71186441,
            -1.18644068,
            0.33898305,
            1.86440678,
            3.38983051,
            4.91525424,
            6.44067797,
            7.96610169,
            9.49152542,
            11.01694915,
            12.54237288,
            14.06779661,
            15.59322034,
            17.11864407,
            18.6440678,
            20.16949153,
            21.69491525,
            23.22033898,
            24.74576271,
            26.27118644,
            27.79661017,
            29.3220339,
            30.84745763,
            32.37288136,
            33.89830508,
            35.42372881,
            36.94915254,
            38.47457627,
            40.,
        ];
        let e2 = linspace(-50., 40., 60);
        for i in 0..60 {
            assert_approx_eq!(r2[i], e2[i]);
        }
    }
}
