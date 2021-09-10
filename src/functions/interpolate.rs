use crate::prelude::Vector;

pub enum ExtrapolationMode {
    Panic,
    Fill(f64, f64),
    Extrapolate,
}

pub fn interp1d_linear(
    x: &[f64],
    y: &[f64],
    tgt: &[f64],
    extrapolate: ExtrapolationMode,
) -> Vector {
    // Performs linear interpolation on an array of values, given some (x, y) pairs.
    // Assumes that x is sorted (in ascending order).
    //
    // Inputs:
    //   x --- 1d array of x values
    //   y --- 1d array of y values, which must be the same size as x
    //   tgt - x values to apply interpolation to
    //   extrapolate -- bool indicating whether to extrapolate for points outside the range of x
    //
    // Outputs:
    //   interp - interpolated (extrapolated) y values corresponding to the input tgt values

    assert_eq!(x.len(), y.len(), "x and y must have the same size");

    let n = x.len();

    // for now, x must be in ascending order (TODO: relax this when there is argsort implementation)
    for i in 0..n - 1 {
        if x[i + 1] - x[i] < 0. {
            panic!("x must be sorted in ascending order");
        }
    }

    interp1d_linear_unchecked(x, y, tgt, extrapolate)
}

pub fn interp1d_linear_unchecked(
    x: &[f64],
    y: &[f64],
    tgt: &[f64],
    extrapolate: ExtrapolationMode,
) -> Vector {
    // Performs linear interpolation on an array of values, given some (x, y) pairs.
    // Assumes, and does not check, that x is sorted (in ascending order).
    //
    // Inputs:
    //   x --- 1d array of x values
    //   y --- 1d array of y values, which must be the same size as x
    //   tgt - x values to apply interpolation to
    //   extrapolate -- bool indicating whether to extrapolate for points outside the range of x
    //
    // Outputs:
    //   interp - interpolated (extrapolated) y values corresponding to the input tgt values

    assert_eq!(x.len(), y.len(), "x and y must have the same size");

    let n = x.len();

    let k = tgt.len();

    // // interpolated values
    let mut interp = Vector::with_capacity(k);

    for i in 0..k {
        // find the closest supplied x (lower and upper)
        let mut idx = 0;
        for j in 0..n - 1 {
            if x[j] > tgt[i] {
                break;
            }
            idx += 1;
        }

        // out of bounds, optionally extrapolate
        if idx == 0 || idx > n {
            match extrapolate {
                ExtrapolationMode::Panic => panic!(
                    "Target out of bounds, need to extrapolate, but extrapolation mode is panic!"
                ),
                ExtrapolationMode::Fill(left, right) => {
                    if idx == 0 {
                        interp.push(left);
                    } else if idx > n {
                        interp.push(right);
                    }
                }
                ExtrapolationMode::Extrapolate => {
                    // extrapolate left
                    if idx == 0 {
                        /* print("extrapolating left ", tgt[i]); */
                        let slope = (y[1] - y[0]) / (x[1] - x[0]);
                        interp.push(-slope * (x[0] - tgt[i]) + y[0]);
                    }
                    // extrapolate right
                    else if idx > n {
                        /* print("extrapolating right ", tgt[i]); */
                        let slope = (y[n] - y[n - 1]) / (x[n] - x[n - 1]);
                        interp.push(slope * (tgt[i] - x[n]) + y[n]);
                    }
                }
            }
        }
        // within bounds, do normal interpolation
        else {
            // how close is target to the closest lower x and upper x?
            let ratio = (tgt[i] - x[idx - 1]) / (x[idx] - x[idx - 1]);

            // interpolate y value based on ratio
            interp.push(ratio * y[idx] + (1. - ratio) * y[idx - 1]);
        }
    }

    return interp;
}

#[cfg(test)]
mod test {
    use crate::prelude::arange;

    use super::*;

    #[test]
    fn test_interp1d_linear_1() {
        let x = Vector::from([0., 0.5, 1., 1.5, 2., 2.5, 3.]);
        let y = x.exp().cos() + x.sin();
        let xnew = arange(-2., 6., 0.2);
        let ynew = interp1d_linear(&x, &y, &xnew, ExtrapolationMode::Extrapolate);

        let ynew_true = Vector::from([
            1.09519, 1.0397, 0.984215, 0.928726, 0.873237, 0.817748, 0.762259, 0.70677, 0.651281,
            0.595791, 0.540302, 0.484813, 0.429324, 0.307211, 0.118474, -0.0702629, 0.265377,
            0.601016, 0.8866, 1.12213, 1.35765, 1.42487, 1.49208, 1.3145, 0.892106, 0.469715,
            0.0473239, -0.375067, -0.797458, -1.21985, -1.64224, -2.06463, -2.48702, -2.90941,
            -3.3318, -3.75419, -4.17658, -4.59898, -5.02137, -5.44376,
        ]);

        assert!(ynew.close_to(&ynew_true, 1e-3));
    }

    #[test]
    fn test_interp1d_linear_2() {
        let x = arange(-1., 4., 0.2);
        let y = (-x.exp()).cos();
        let xnew = arange(-2., 6., 0.2);
        let ynew = interp1d_linear(&x, &y, &xnew, ExtrapolationMode::Extrapolate);

        let ynew_true = Vector::from([
            1.09485857,
            1.06250527,
            1.03015197,
            0.99779867,
            0.96544537,
            0.93309208,
            0.90073878,
            0.85314506,
            0.78362288,
            0.68314866,
            0.54030231,
            0.34232808,
            0.07888957,
            -0.2486851,
            -0.60895668,
            -0.91173391,
            -0.98410682,
            -0.61089378,
            0.23832758,
            0.97285375,
            0.44835624,
            -0.92115269,
            0.02759859,
            0.62366998,
            -0.74070077,
            0.32859476,
            0.82521644,
            0.11868939,
            0.452814,
            0.75253893,
            1.05226386,
            1.35198879,
            1.65171372,
            1.95143865,
            2.25116358,
            2.55088852,
            2.85061345,
            3.15033838,
            3.45006331,
            3.74978824,
        ]);

        assert!(ynew.close_to(&ynew_true, 1e-3));
    }
}
