use std::cmp;

#[cfg(feature = "lapack")]
use super::transpose;
#[cfg(feature = "lapack")]
use lapack::dgetrf;

use super::is_square;

/// Computes the pivoted LU decomposition of a square matrix. For some matrix A, this decomposition
/// is A = PLU. The resulting matrix has U in its upper triangle and L in its lower triangle.
/// The unit diagonal elements of L are not stored. The pivot indices representing the permutation
/// matrix P is also returned.
pub fn lu(matrix: &[f64]) -> (Vec<f64>, Vec<i32>) {
    let mut lu = matrix.to_vec();

    let n = is_square(matrix).unwrap();

    // this is slower than the non-lapack version, even without the double transpose
    // #[cfg(feature = "lapack")]
    // {
    //     let mut ipiv: Vec<i32> = vec![0; n];
    //     let mut info = 0;
    //     let mut lut = transpose(&lu, n);
    //     unsafe {
    //         dgetrf(n as i32, n as i32, &mut lut, n as i32, &mut ipiv, &mut info);
    //         assert_eq!(info, 0, "dgetrf failed");
    //     }
    //     (lut, ipiv)
    // }

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
                let t = lu[p * n + k];
                lu[p * n + k] = lu[j * n + k];
                lu[j * n + k] = t;
            }
            let k = pivots[p];
            pivots[p] = pivots[j];
            pivots[j] = k;
        }

        if j < n && lu[j * n + j] != 0. {
            for i in (j + 1)..n {
                lu[i * n + j] /= lu[j * n + j];
            }
        }
    }

    (lu, pivots)
}
