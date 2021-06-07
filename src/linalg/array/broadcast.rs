use super::{matmatadd, matmatdiv, matmatmul, matmatsub, Matrix};

#[derive(Debug, Clone, Copy)]
pub enum Broadcast {
    Hstack(usize),
    Vstack(usize),
    IsScalar,
    None,
    Invalid,
}

fn calc_broadcast_shape(m1: &Matrix, m2: &Matrix) -> [Broadcast; 2] {
    if m1.shape() == m2.shape() {
        [Broadcast::None, Broadcast::None]
    } else if m1.shape().contains(&1) {
        if m1.nrows == 1 {
            assert!(
                m1.ncols == m2.ncols // single vstack broadcast for m1
                || m2.ncols == 1 // vstack broadcast m1 and hstack broadcast m2
                || m1.ncols == 1 // m1 is just a single element matrix, vstack and hstack it.
            );
            if m1.ncols == m2.ncols {
                [Broadcast::Vstack(m2.nrows), Broadcast::None]
            } else if m2.ncols == 1 {
                [Broadcast::Vstack(m2.nrows), Broadcast::Hstack(m1.ncols)]
            } else if m1.ncols == 1 {
                [Broadcast::IsScalar, Broadcast::None]
            } else {
                [Broadcast::Invalid, Broadcast::Invalid]
            }
        } else {
            // m1.ncols == 1
            assert!(
                m1.nrows == m2.nrows // single hstack broadcast for m1
                || m2.nrows == 1 // hstack broadcast m1 and vstack broadcast m2
                || m1.nrows == 1 // m1 is just a single element matrix
            );
            if m1.nrows == m2.nrows {
                [Broadcast::Hstack(m2.ncols), Broadcast::None]
            } else if m2.nrows == 1 {
                [Broadcast::Hstack(m2.ncols), Broadcast::Vstack(m1.nrows)]
            } else if m1.nrows == 1 {
                [Broadcast::IsScalar, Broadcast::None]
            } else {
                [Broadcast::Invalid, Broadcast::Invalid]
            }
        }
    } else if m2.shape().contains(&1) {
        let [b1, b2] = calc_broadcast_shape(m2, m1);
        [b2, b1]
    } else {
        [Broadcast::Invalid, Broadcast::Invalid]
    }
}

macro_rules! broadcast_op {
    ($op: tt, $fnname: ident, $matmatfn: ident) => {
        pub fn $fnname(m1: &Matrix, m2: &Matrix) -> Matrix {
            let b = calc_broadcast_shape(m1, m2);
            match b {
                [Broadcast::None, Broadcast::None] => {
                    assert_eq!(m1.shape(), m2.shape());
                    // easy, do nothing special
                    $matmatfn(m1, m2)
                }
                [Broadcast::Hstack(hstack), Broadcast::None] => {
                    assert_eq!(hstack, m2.ncols);
                    let mut new = m2.clone();
                    // each element in m1 gets added to each row in m2
                    // .    ....
                    // .    ....
                    // .    ....
                    // .    ....
                    for i in 0..m1.nrows {
                        new.apply_along_row(i, |x| m1[i][0] $op x)
                    }
                    new
                }
                [Broadcast::Vstack(vstack), Broadcast::None] => {
                    assert_eq!(vstack, m2.nrows);
                    let mut new = m2.clone();
                    // each element in m1 gets added to each column in m2
                    // ....     ....
                    //          ....
                    //          ....
                    //          ....
                    for i in 0..new.nrows {
                        new[i].iter_mut().zip(&m1[0]).for_each(|(x, y)| *x = y $op *x);
                    }
                    new
                }
                [Broadcast::None, Broadcast::Hstack(hstack)] => {
                    assert_eq!(hstack, m1.ncols);
                    let mut new = m1.clone();
                    // each element in m1 gets added to each row in m2
                    // ....   .
                    // ....   .
                    // ....   .
                    // ....   .
                    for i in 0..m2.nrows {
                        new.apply_along_row(i, |x| x $op m2[i][0])
                    }
                    new
                },
                [Broadcast::None, Broadcast::Vstack(vstack)] => {
                    assert_eq!(vstack, m1.nrows);
                    let mut new = m1.clone();
                    // each element in m1 gets added to each column in m2
                    // ....   ....
                    // ....
                    // ....
                    // ....
                    for i in 0..new.nrows {
                        new[i].iter_mut().zip(&m2[0]).for_each(|(x, y)| *x = *x $op y);
                    }
                    new
                },
                [Broadcast::Hstack(hstack), Broadcast::Vstack(vstack)] => {
                    assert_eq!(m2.ncols, hstack);
                    assert_eq!(m1.nrows, vstack);
                    assert_eq!(m2.nrows, 1);
                    assert_eq!(m1.ncols, 1);
                    // .  .  .  .
                    // .
                    // .
                    // .
                    let mut new = Matrix::zeros(m1.nrows, m2.ncols);
                    for i in 0..new.nrows {
                        for j in 0..new.ncols {
                            new[i][j] = m1[i][0] $op m2[0][j]
                        }
                    }
                    new
                }
                [Broadcast::Vstack(vstack), Broadcast::Hstack(hstack)] => {
                    assert_eq!(m1.ncols, hstack);
                    assert_eq!(m2.nrows, vstack);
                    assert_eq!(m1.nrows, 1);
                    assert_eq!(m2.ncols, 1);
                    // .  .  .  .
                    //          .
                    //          .
                    //          .
                    let mut new = Matrix::zeros(m2.nrows, m1.ncols);
                    for i in 0..new.nrows {
                        for j in 0..new.ncols {
                            new[i][j] = m1[0][j] $op m2[i][0]
                        }
                    }
                    new
                }
                [Broadcast::IsScalar, _] => {
                    // 2nd broadcast should always be Broadcast::None
                    assert!(m1.nrows == 1 && m1.ncols == 1);
                    m1[0][0] $op m2
                }
                [_, Broadcast::IsScalar] => {
                    // 1st broadcast should always be Broadcast::None
                    assert!(m2.nrows == 1 && m2.ncols == 1);
                    m1 $op m2[0][0]
                }
                _ => {
                    // one of them is invalid, or have [hstack, hstack] or [vstack, vstack], neither
                    // of which are possible
                    panic!("invalid broadcast shape")
                }
            }
        }
    };
}

broadcast_op!(+, broadcast_add, matmatadd);
broadcast_op!(-, broadcast_sub, matmatsub);
broadcast_op!(*, broadcast_mul, matmatmul);
broadcast_op!(/, broadcast_div, matmatdiv);

#[cfg(test)]
mod tests {
    use super::super::super::arange;
    use super::super::Vector;
    use super::*;

    #[test]
    fn test_broadcast_1() {
        let mut a = Matrix::new([8., 9., 2., 5., 4., 9., 1., 6., 3.], 3, 3);
        let mut b = Vector::new([1., 2., 3.]).to_matrix(); // 1x3 matrix
        let c = broadcast_add(&a, &b);
        assert_eq!(c, Matrix::new([9., 11., 5., 6., 6., 12., 2., 8., 6.], 3, 3));
        let d = broadcast_mul(&a, &b);
        assert_eq!(
            d,
            Matrix::new([8., 18., 6., 5., 8., 27., 1., 12., 9.], 3, 3)
        );
        b.t_mut(); // 3x1
        let e = broadcast_sub(&b, &a);
        assert_eq!(
            e,
            Matrix::new([-7., -8., -1., -3., -2., -7., 2., -3., 0.], 3, 3)
        );
        a.reshape_mut(1, -1); // flatten to 9x1
        let f = broadcast_div(&a, &b);
        assert_eq!(
            f,
            Matrix::new(
                vec![
                    8.,
                    9.,
                    2.,
                    5.,
                    4.,
                    9.,
                    1.,
                    6.,
                    3.,
                    4.,
                    4.5,
                    1.,
                    2.5,
                    2.,
                    4.5,
                    0.5,
                    3.,
                    1.5,
                    2. + 2. / 3.,
                    3.,
                    2. / 3.,
                    1. + 2. / 3.,
                    1. + 1. / 3.,
                    3.,
                    1. / 3.,
                    2.,
                    1.
                ],
                3,
                9
            )
        );
    }

    #[test]
    fn test_broadcast_2() {
        let a = Matrix::new(
            [
                -0.699, -1.031, 1.235, 0.328, 0.026, 0.046, 1.501, 0.438, 1.304, 0.728, 1., -0.417,
                -0.265, 0.091, 0.422, 0.602,
            ],
            4,
            4,
        );
        let b = Matrix::new([0.896, 0.488, 0.577, 0.316], 4, 1);
        let c = broadcast_sub(&a, &b);
        assert_eq!(
            c,
            Matrix::new(
                [
                    -1.595, -1.927, 0.339, -0.568, -0.462, -0.442, 1.013, -0.05, 0.727, 0.151,
                    0.423, -0.994, -0.581, -0.225, 0.106, 0.286
                ],
                4,
                4
            )
        );
        let d = broadcast_sub(&a, &b.t());
        assert_eq!(
            d,
            Matrix::new(
                [
                    -1.595, -1.519, 0.658, 0.012, -0.87, -0.442, 0.924, 0.122, 0.408, 0.24, 0.423,
                    -0.733, -1.161, -0.397, -0.155, 0.286
                ],
                4,
                4
            )
        );
        let e = broadcast_div(&b, &a);
        assert_eq!(
            e,
            Matrix::new(
                [
                    -1.2818311874105868,
                    -0.86905916585839,
                    0.7255060728744939,
                    2.7317073170731705,
                    18.76923076923077,
                    10.608695652173912,
                    0.3251165889407062,
                    1.1141552511415524,
                    0.44248466257668706,
                    0.7925824175824175,
                    0.577,
                    -1.3836930455635492,
                    -1.1924528301886792,
                    3.4725274725274726,
                    0.7488151658767773,
                    0.5249169435215947
                ],
                4,
                4
            )
        );
        let f = broadcast_div(&a.t(), &b);
        assert_eq!(
            f,
            Matrix::new(
                [
                    -0.7801339285714285,
                    0.02901785714285714,
                    1.4553571428571428,
                    -0.2957589285714286,
                    -2.1127049180327866,
                    0.0942622950819672,
                    1.4918032786885247,
                    0.1864754098360656,
                    2.1403812824956674,
                    2.601386481802426,
                    1.733102253032929,
                    0.7313691507798961,
                    1.0379746835443038,
                    1.3860759493670887,
                    -1.3196202531645569,
                    1.9050632911392404
                ],
                4,
                4
            )
        );
    }

    #[test]
    fn test_broadcast_3() {
        let a = Matrix::new([1., 2., 3., 4.], 2, 2);
        let b = Matrix::new([3., 4., 1., 1.], 2, 2);
        let c = broadcast_add(&a, &b);
        assert_eq!(c, Matrix::new([4., 6., 4., 5.], 2, 2));
        let d = broadcast_sub(&a, &b.t());
        assert_eq!(d, Matrix::new([-2., 1., -1., 3.], 2, 2));
        let e = broadcast_div(&a.t(), &b);
        assert_eq!(e, Matrix::new([1. / 3., 0.75, 2., 4.], 2, 2));
    }

    #[test]
    fn test_broadcast_4() {
        let a = arange(0., 4., 1.).to_matrix().reshape(1, 4);
        let b = arange(0., 4., 1.).to_matrix().reshape(4, 1);
        let c = broadcast_sub(&a, &b);
        assert_eq!(
            c,
            Matrix::new(
                [0., 1., 2., 3., -1., 0., 1., 2., -2., -1., 0., 1., -3., -2., -1., 0.],
                4,
                4
            )
        );
        let d = broadcast_mul(&a, &b.t());
        assert_eq!(d, Matrix::new([0., 1., 4., 9.], 1, 4));
        let e = broadcast_add(&b, &a.t());
        assert_eq!(e, Matrix::new([0., 2., 4., 6.], 4, 1));
    }
}
