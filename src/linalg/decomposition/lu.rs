//! Implements [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition) and system solving with the decomposition.

use std::cmp;

// #[cfg(feature = "lapack")]
// use lapack::dgetrf;

use crate::linalg::is_square;

/// Computes the pivoted LU decomposition of a square matrix. For some matrix A, this decomposition
/// is A = PLU. The resulting matrix has U in its upper triangle and L in its lower triangle.
/// The unit diagonal elements of L are not stored. The pivot indices representing the permutation
/// matrix P is also returned.
pub fn lu(matrix: &[f64]) -> (Vec<f64>, Vec<i32>) {
    let n = is_square(matrix).unwrap();
    let mut lu = matrix.to_vec();

    let mut pivots: Vec<i32> = (0..n).map(|x| x as i32).collect();

    for j in 0..n {
        for i in 0..n {
            let mut s = 0.;
            for k in 0..cmp::min(i, j) {
                s += lu[i * n + k] * lu[k * n + j];
            }
            lu[i * n + j] -= s;
        }

        let mut p = j;
        for i in (j + 1)..n {
            if lu[i * n + j].abs() > lu[p * n + j].abs() {
                p = i;
            }
        }

        if p != j {
            for k in 0..n {
                lu.swap(p * n + k, j * n + k)
            }
            pivots.swap(p, j);
        }

        if j < n && lu[j * n + j] != 0. {
            for i in (j + 1)..n {
                lu[i * n + j] /= lu[j * n + j];
            }
        }
    }

    (lu, pivots)
}

/// Solve the linear system Ax = b given a LU decomposed matrix A. The first argument should be a
/// tuple, where the first element is the LU decomposed matrix and the second element is the pivots
/// P.
pub fn lu_solve(lu: &[f64], pivots: &[i32], b: &[f64]) -> Vec<f64> {
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

    x
}

#[cfg(test)]
mod tests {
    use super::super::lu_solve;
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_lu() {
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

        let (lu, piv) = lu(&A);
        let x = lu_solve(&lu, &piv, &b);

        let lu_ref = vec![
            6.02201052,
            -6.71212792,
            -1.74683781,
            -6.08893455,
            0.45986012,
            5.39640987,
            -11.58212785,
            -5.90514476,
            -0.07724881,
            -0.67133363,
            -12.92807843,
            -6.29768626,
            -0.42133955,
            -0.01981984,
            -0.28902028,
            -5.53658552,
        ];
        let x_ref = vec![0.68581948, 0.33965616, 0.8063919, -0.69182874];

        for i in 0..16 {
            assert_approx_eq!(lu[i], lu_ref[i]);
        }

        for i in 0..4 {
            assert_approx_eq!(x[i], x_ref[i]);
        }
    }
}
