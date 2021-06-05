use std::{
    fmt::{Display, Formatter, Result},
    ops::Index,
};

use impl_ops::*;
use std::ops;

use crate::prelude::{is_square, is_symmetric};

use super::Vector;

/// Matrix struct.
#[derive(Debug, Clone)]
pub struct Matrix {
    data: Vector,
    nrows: usize,
    ncols: usize,
    is_square: bool,
    is_symmetric: bool,
    // iter_counter: usize,
}

impl Matrix {
    pub fn empty() -> Self {
        Self {
            data: Vector::empty(),
            ncols: 0,
            nrows: 0,
            is_square: false,
            is_symmetric: false,
            // iter_counter: 0,
        }
    }

    pub fn new<T>(data: T, [nrows, ncols]: [usize; 2]) -> Self
    where
        T: AsRef<[f64]>,
    {
        let v = data.as_ref();
        let is_square = match is_square(&v) {
            Ok(val) => {
                assert!(nrows == ncols && nrows == val, "matrix not square");
                true
            }
            Err(_) => false,
        };
        let is_symmetric = if is_square { is_symmetric(&v) } else { false };

        Self {
            data: Vector::from(v),
            ncols,
            nrows,
            is_square,
            is_symmetric,
            // iter_counter: 0,
        }
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.nrows, self.ncols]
    }

    pub fn size(&self) -> usize {
        self.nrows * self.ncols
    }

    pub fn get_row(&self, row: usize) -> Vector {
        assert!(row < self.nrows);
        Vector::from(&self[row])
    }

    pub fn get_col(&self, col: usize) -> Vector {
        assert!(col < self.ncols);

        let mut v = Vector::zeros(self.nrows);

        for i in 0..self.nrows {
            v[i] = self[i][col];
        }

        Vector::from(v)
    }

    pub fn sum_rows(&self) -> Vector {
        let mut sums = Vector::zeros(self.nrows);
        for row in 0..self.nrows {
            sums[row] = Vector::from(&self[row]).sum();
        }
        sums
    }

    pub fn sum_cols(&self) -> Vector {
        let mut sums = Vector::zeros(self.ncols);
        for row in 0..self.nrows {
            sums = sums + Vector::from(&self[row]);
        }
        sums
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter) -> Result {
        Ok(for (rownum, row) in self.into_iter().enumerate() {
            if rownum == 0 {
                writeln!(f, "[{:?} ", row)?;
            } else if rownum == self.nrows - 1 {
                writeln!(f, " {:?}]", row)?;
            } else {
                writeln!(f, " {:?} ", row)?;
            }
        })
    }
}

// impl Iterator for Matrix {
//     type Item = Vector;

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.iter_counter < self.nrows {
//             self.iter_counter += 1;
//             Some(Vector::from(&self[self.iter_counter - 1]))
//         } else {
//             None
//         }
//         // let iter: Vec<Vector> = self
//         //     .data
//         //     .chunks(self.ncols)
//         //     .map(|x| Vector::from(x))
//         //     .collect();
//         // iter.into_iter().next()
//     }
// }

impl<'a> IntoIterator for &'a Matrix {
    type Item = &'a [f64];
    type IntoIter = std::slice::Chunks<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.chunks(self.ncols)
    }
}

impl<'a> IntoIterator for &'a mut Matrix {
    type Item = &'a mut [f64];
    type IntoIter = std::slice::ChunksMut<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.chunks_mut(self.ncols)
    }
}

impl Default for Matrix {
    fn default() -> Self {
        Self::empty()
    }
}

impl Index<usize> for Matrix {
    type Output = [f64];
    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < self.nrows);
        &self.data[i * self.ncols..(i + 1) * self.ncols]
    }
}

impl Index<[usize; 2]> for Matrix {
    type Output = f64;

    fn index(&self, [i, j]: [usize; 2]) -> &Self::Output {
        assert!(i < self.nrows && j < self.ncols);
        &self.data[i * self.ncols + j]
    }
}

// vector-vector ops
macro_rules! impl_vv_ops_helper {
    ($op: tt) => {
        impl_op_ex!($op |u: &Matrix, v: &Matrix| -> Matrix {
            assert_eq!(u.shape(), v.shape());
            Matrix::new( &u.data $op &v.data, u.shape())
        });
    };
}

impl_vv_ops_helper!(+);
impl_vv_ops_helper!(-);
impl_vv_ops_helper!(*);
impl_vv_ops_helper!(/);

// vector-float and float-vector ops
impl_op_ex_commutative!(+ |f: f64, v: &Matrix| -> Matrix { Matrix::new(&v.data + f, v.shape()) });
impl_op_ex_commutative!(*|f: f64, v: &Matrix| -> Matrix { Matrix::new(&v.data * f, v.shape()) });
impl_op_ex!(-|f: f64, v: &Matrix| -> Matrix { Matrix::new(f - &v.data, v.shape()) });
impl_op_ex!(-|v: &Matrix, f: f64| -> Matrix { Matrix::new(&v.data - f, v.shape()) });
impl_op_ex!(/|f: f64, v: &Matrix| -> Matrix { Matrix::new(f / &v.data, v.shape()) });
impl_op_ex!(/|v: &Matrix, f: f64| -> Matrix { Matrix::new(&v.data / f, v.shape()) });

macro_rules! impl_unaryops_matrix {
    ($op: ident) => {
        impl Matrix {
            #[doc = "Apply the `f64` operation `"]
            #[doc = stringify!($op)]
            #[doc = "` element-wise to the matrix."]
            pub fn $op(&self) -> Self {
                Self::new(self.data.$op(), self.shape())
            }
        }
    };
}

impl_unaryops_matrix!(ln);
impl_unaryops_matrix!(ln_1p);
impl_unaryops_matrix!(log10);
impl_unaryops_matrix!(log2);
impl_unaryops_matrix!(exp);
impl_unaryops_matrix!(exp2);
impl_unaryops_matrix!(exp_m1);
impl_unaryops_matrix!(sin);
impl_unaryops_matrix!(cos);
impl_unaryops_matrix!(tan);
impl_unaryops_matrix!(sinh);
impl_unaryops_matrix!(cosh);
impl_unaryops_matrix!(tanh);
impl_unaryops_matrix!(asin);
impl_unaryops_matrix!(acos);
impl_unaryops_matrix!(atan);
impl_unaryops_matrix!(asinh);
impl_unaryops_matrix!(acosh);
impl_unaryops_matrix!(atanh);
impl_unaryops_matrix!(sqrt);
impl_unaryops_matrix!(cbrt);
impl_unaryops_matrix!(abs);
impl_unaryops_matrix!(floor);
impl_unaryops_matrix!(ceil);
impl_unaryops_matrix!(to_radians);
impl_unaryops_matrix!(to_degrees);
impl_unaryops_matrix!(recip);
impl_unaryops_matrix!(round);
impl_unaryops_matrix!(signum);

macro_rules! impl_unaryops_with_arg_matrix {
    ($op: ident, $argtype: ident) => {
        impl Matrix {
            #[doc = "Apply the `f64` operation `"]
            #[doc = stringify!($op)]
            #[doc = "` element-wise to the matrix."]
            pub fn $op(&self, arg: $argtype) -> Self {
                Self::new(self.data.$op(arg), self.shape())
            }
        }
    };
}

impl_unaryops_with_arg_matrix!(powi, i32);
impl_unaryops_with_arg_matrix!(powf, f64);

macro_rules! impl_reduction_fns_matrix {
    ($($fn: ident),+) => {
        impl Matrix {
            $(
                pub fn $fn(&self) -> f64 {
                    self.data.$fn()
                }
            )+
        }
    };
}

impl_reduction_fns_matrix!(norm, max, mean, min, std, sum, var, sample_std, sample_var);
