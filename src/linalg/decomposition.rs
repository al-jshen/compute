use std::cmp;

pub fn lu(matrix: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let mut lu = matrix.to_vec();

    let n = (matrix.len() as f32).sqrt();
    assert!(n % 1. == 0.);
    let n = n as usize;

    let mut pivots: Vec<usize> = (0..n).collect();

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

    (pivots, lu)
}
