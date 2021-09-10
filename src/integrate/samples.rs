use crate::linalg::Vector;

/// Integrate samples from a function
pub fn trapezoid(y: &[f64], x: Option<&[f64]>, dx: Option<f64>) -> f64 {
    let diff_x = if let Some(xarr) = x {
        assert_eq!(y.len(), xarr.len(), "x and y must have the same length.");
        assert!(dx.is_none(), "Since x was passed, dx must be None");
        (1..xarr.len())
            .map(|i| xarr[i] - xarr[i - 1])
            .collect::<Vector>()
    } else {
        Vector::ones(y.len() - 1) * if let Some(diff) = dx { diff } else { 1. }
    };

    (1..y.len())
        .map(|i| (y[i] + y[i - 1]) / 2. * diff_x[i - 1])
        .sum()
}

#[cfg(test)]
mod test {
    use approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_trapezoid() {
        let x = vec![1., 2., 3., 5., 6., 7.];
        let y = vec![2., 4., 5., 6., 5., 2.];
        let int = trapezoid(&y, Some(&x), None);
        assert_approx_eq!(int, 27.5);

        let x = vec![
            -1.2492823978867575,
            -1.0827721123898908,
            -0.9406043223301596,
            -0.7680308246853681,
            -0.4229503089687044,
            0.015055841196579461,
            0.2583590719224359,
            0.6164607574036753,
            0.8938966780618812,
            1.6356468475989316,
        ];

        let y = vec![
            -1.03464106,
            0.51438613,
            0.1316389,
            -0.53705463,
            -1.18204451,
            -1.12118844,
            1.08181116,
            -1.0660365,
            -1.5044052,
            -0.58971003,
        ];

        let int = trapezoid(&y, Some(&x), None);
        assert_approx_eq!(int, -1.9685902719500574);
    }
}
