//! Implementation of matrix products for vectors and matrices.

use super::{
    super::{dot, matmul},
    Matrix, Vector,
};

/// A trait for performing matrix products. Follows the behaviour of numpy's `matmul`.
///
/// If multiplying two matrices, performs conventional matrix multiplication.
/// If multiplying a vector with a matrix, promotes the vector to a matrix by prepending a 1 to its dimensions, then performing conventional matrix multiplication, and then flattening the result back down into a vector.
/// If multiplying a matrix with a vector, promotes the vector to a matrix by appending a 1 to its dimensions, then performing conventional matrix multiplication, and then flattening the result back down into a vector.
pub trait Dot<T, S> {
    /// Performs dot product of self and other.
    fn dot(&self, other: T) -> S;
    /// Performs dot product of self and other, transposing other.
    fn dot_t(&self, other: T) -> S;
    /// Performs dot product of self and other, transposing self.
    fn t_dot(&self, other: T) -> S;
    /// Performs dot product of self and other, transposing both self and other.
    fn t_dot_t(&self, other: T) -> S;
}

macro_rules! impl_macro_for_types {
    ($macro: ident, $t1: ty, $t2: ty) => {
        $macro!($t1, $t2);
        $macro!($t1, &$t2);
        $macro!(&$t1, $t2);
        $macro!(&$t1, &$t2);
    };
}

macro_rules! impl_mat_mat_dot {
    ($selftype: ty, $othertype: ty) => {
        impl Dot<$othertype, Matrix> for $selftype {
            fn dot(&self, other: $othertype) -> Matrix {
                assert_eq!(self.ncols, other.nrows, "matrix shapes not compatible");
                let output = matmul(
                    &self.data(),
                    &other.data(),
                    self.nrows,
                    other.nrows,
                    false,
                    false,
                );
                Matrix::new(output, self.nrows as i32, other.ncols as i32)
            }

            fn t_dot(&self, other: $othertype) -> Matrix {
                assert_eq!(self.nrows, other.nrows, "matrix shapes not compatible");
                let output = matmul(
                    &self.data(),
                    &other.data(),
                    self.nrows,
                    other.nrows,
                    true,
                    false,
                );
                Matrix::new(output, self.ncols as i32, other.ncols as i32)
            }
            fn dot_t(&self, other: $othertype) -> Matrix {
                assert_eq!(self.ncols, other.ncols, "matrix shapes not compatible");
                let output = matmul(
                    &self.data(),
                    &other.data(),
                    self.nrows,
                    other.nrows,
                    false,
                    true,
                );
                Matrix::new(output, self.nrows as i32, other.nrows as i32)
            }
            fn t_dot_t(&self, other: $othertype) -> Matrix {
                assert_eq!(self.nrows, other.ncols, "matrix shapes not compatible");
                let output = matmul(
                    &self.data(),
                    &other.data(),
                    self.nrows,
                    other.nrows,
                    true,
                    true,
                );
                Matrix::new(output, self.ncols as i32, other.nrows as i32)
            }
        }
    };
}

impl_macro_for_types!(impl_mat_mat_dot, Matrix, Matrix);

macro_rules! impl_dot_append_one {
    ($othertype: ty, $innerop: ident, $($op: ident),+) => {
        $(
            fn $op(&self, other: $othertype) -> Vector {
                let mut o = other.clone().to_owned().to_matrix();
                o.t_mut();
                self.$innerop(o).to_vec()
            }
        )+
    }
}

macro_rules! impl_mat_vec_dot {
    ($selftype: ty, $othertype: ty) => {
        impl Dot<$othertype, Vector> for $selftype {
            // transpose on the vector does nothing
            impl_dot_append_one!($othertype, dot, dot, dot_t);
            impl_dot_append_one!($othertype, t_dot, t_dot, t_dot_t);
        }
    };
}

impl_macro_for_types!(impl_mat_vec_dot, Matrix, Vector);

macro_rules! impl_dot_prepend_one {
    ($othertype: ty, $innerop: ident, $($op: ident),+) => {
        $(
            fn $op(&self, other: $othertype) -> Vector {
                self.clone().to_owned().to_matrix().$innerop(other).to_vec()
            }
        )+
    }
}

macro_rules! impl_vec_mat_dot {
    ($selftype: ty, $othertype: ty) => {
        impl Dot<$othertype, Vector> for $selftype {
            // transpose on the vector does nothing
            impl_dot_prepend_one!($othertype, dot, dot, t_dot);
            impl_dot_prepend_one!($othertype, dot_t, dot_t, t_dot_t);
        }
    };
}

impl_macro_for_types!(impl_vec_mat_dot, Vector, Matrix);

macro_rules! impl_dot_vec_vec {
    ($othertype: ty, $($op: ident),+) => {
        $(
            fn $op(&self, other: $othertype) -> f64 {
                dot(&self.data(), &other.data())
            }
        )+
    }
}

macro_rules! impl_vec_vec_dot {
    ($selftype: ty, $othertype: ty) => {
        impl Dot<$othertype, f64> for $selftype {
            impl_dot_vec_vec!($othertype, dot, t_dot, dot_t, t_dot_t);
        }
    };
}

impl_macro_for_types!(impl_vec_vec_dot, Vector, Vector);
