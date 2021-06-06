use std::{
    fmt::{Display, Formatter, Result},
    mem::{replace, swap},
    ops::Index,
    panic,
};

use std::ops;

use crate::prelude::{matmul, transpose};

use super::vops::*;
use super::Vector;

/// Matrix struct.
#[derive(Debug, Clone)]
pub struct Matrix {
    data: Vector,
    pub nrows: usize,
    pub ncols: usize,
}

impl Matrix {
    pub fn empty() -> Self {
        Self {
            data: Vector::empty(),
            ncols: 0,
            nrows: 0,
        }
    }

    /// Make a matrix filled with zeros.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self::new(Vector::zeros(nrows * ncols), nrows, ncols)
    }

    /// Make a matrix filled with ones.
    pub fn ones(nrows: usize, ncols: usize) -> Self {
        Self::new(Vector::ones(nrows * ncols), nrows, ncols)
    }

    /// Make an identity matrix with size `dims`.
    pub fn eye(dims: usize) -> Self {
        let mut m = Self::zeros(dims, dims);
        for i in 0..dims {
            m.data[i * dims + i] = 1.;
        }
        m
    }

    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// Determines whether a matrix is symmetric.
    fn is_symmetric(&self) -> bool {
        if self.is_square() {
            for i in 0..self.nrows {
                for j in i..self.ncols {
                    if self.data[i * self.ncols + j] != self.data[j * self.nrows + i] {
                        return false;
                    }
                }
            }
            true
        } else {
            false
        }
    }

    fn is_positive_definite(&self) -> bool {
        if self.is_symmetric() {
            for i in 0..self.ncols {
                if self.data[i * self.ncols + i] <= 0. {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Get the diagonal elements of the matrix.
    pub fn diag(&self) -> Vector {
        let n = self.nrows.min(self.ncols);
        let mut diag = Vector::with_capacity(n);
        for i in 0..n {
            diag.push(self.data[i * n + i]);
        }
        diag
    }

    /// Make a new matrix with the given number of rows and columns.
    pub fn new<T>(data: T, nrows: usize, ncols: usize) -> Self
    where
        T: Into<Vector>,
    {
        let v = data.into();
        let is_square = nrows == ncols;

        Self {
            data: Vector::from(v),
            ncols,
            nrows,
        }
    }

    /// Get the number of rows and columns in the matrix.
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Get the total number of elements in the matrix.
    pub fn size(&self) -> usize {
        self.nrows * self.ncols
    }

    /// Reshape the matrix in-place. A size of `-1` in either the rows or the columns means that the size
    /// for that dimension will be automatically determined if possible.
    pub fn reshape_mut(&mut self, nrows: i32, ncols: i32) -> &mut Self {
        let size = self.size();
        if nrows > 0 && ncols > 0 {
            assert_eq!(nrows * ncols, size as i32, "invalid shape");
            self.nrows = nrows as usize;
            self.ncols = ncols as usize;
        } else if nrows < 0 {
            assert!(nrows == -1 && ncols > 0, "invalid number of rows");
            // automatically determine number of rows
            self.ncols = ncols as usize;
            self.nrows = size / ncols as usize;
        } else if ncols < 0 {
            assert!(ncols == -1 && nrows > 0, "invalid number of columns");
            // automatically determine number of columns
            self.nrows = nrows as usize;
            self.ncols = size / nrows as usize;
        } else {
            panic!("invalid shape");
        }
        self
    }

    /// Reshape the matrix in-place. A size of `-1` in either the rows or the columns means that the size
    /// for that dimension will be automatically determined if possible.
    pub fn reshape(&self, nrows: i32, ncols: i32) -> Self {
        let size = self.size();
        if nrows > 0 && ncols > 0 {
            assert_eq!(nrows * ncols, size as i32, "invalid shape");
            Matrix::new(self.data.clone(), nrows as usize, ncols as usize)
        } else if nrows < 0 {
            assert!(nrows == -1 && ncols > 0, "invalid shape");
            // automatically determine number of rows
            Matrix::new(self.data.clone(), size / ncols as usize, ncols as usize)
        } else if ncols < 0 {
            assert!(ncols == -1 && nrows > 0, "invalid shape");
            // automatically determine number of columns
            Matrix::new(self.data.clone(), nrows as usize, size / nrows as usize)
        } else {
            panic!("invalid shape");
        }
    }

    /// Get the row of the matrix.
    pub fn get_row(&self, row: usize) -> Vector {
        assert!(row < self.nrows);
        Vector::from(&self[row])
    }

    /// Get the column of the matrix.
    pub fn get_col(&self, col: usize) -> Vector {
        assert!(col < self.ncols);

        let mut v = Vector::zeros(self.nrows);

        for i in 0..self.nrows {
            v[i] = self[i][col];
        }

        Vector::from(v)
    }

    /// Sum the matrix across the rows.
    pub fn sum_rows(&self) -> Vector {
        let mut sums = Vector::zeros(self.nrows);
        for row in 0..self.nrows {
            sums[row] = Vector::from(&self[row]).sum();
        }
        sums
    }

    /// Sum the matrix down the columns.
    pub fn sum_cols(&self) -> Vector {
        let mut sums = Vector::zeros(self.ncols);
        for row in 0..self.nrows {
            // sums = sums + Vector::from(&self[row]);
            for col in 0..self.ncols {
                sums[col] = sums[col] + self[row][col];
            }
        }
        sums
    }

    /// Calculates the infinity norm of the matrix.
    pub fn inf_norm(&self) -> f64 {
        self.abs().sum_rows().max()
    }

    pub fn flat_idx(&self, idx: usize) -> f64 {
        assert!(idx < self.size());
        (&self.data)[idx]
    }

    pub fn flat_idx_replace(&mut self, idx: usize, val: f64) -> &mut Self {
        assert!(idx < self.size());
        self.data[idx] = val;
        self
    }

    /// Transpose the matrix.
    pub fn t(&self) -> Self {
        let t = transpose(&self.data, self.nrows);
        Matrix::new(t, self.ncols, self.nrows)
    }

    /// Transpose the matrix in-place.
    pub fn t_mut(&mut self) -> &mut Self {
        let t = transpose(&self.data, self.nrows);
        self.data = Vector::new(t);
        swap(&mut self.ncols, &mut self.nrows);
        self
    }

    /// Return a reference to the underlying Vector holding the data.
    pub fn data(&self) -> &Vector {
        &self.data
    }

    /// Return a mutable reference to the underlying Vector holding the data.
    pub fn data_mut(&mut self) -> &mut Vector {
        &mut self.data
    }

    /// Converts the matrix to a Vector.
    pub fn to_vec(&self) -> Vector {
        self.data.clone()
    }

    /// Horizontal concatenation of matrices. Adds `other` to the right of the calling matrix.
    pub fn hcat(&self, other: Self) -> Self {
        assert_eq!(self.nrows, other.nrows);
        let mut new_vec = Vector::empty();
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                new_vec.push(self.data[i * self.ncols + j]);
            }
            for j in 0..other.ncols {
                new_vec.push(other.data[i * other.ncols + j]);
            }
        }
        Matrix::new(new_vec, self.nrows, self.ncols + other.ncols)
    }

    /// Vertical concatenation of matrices. Adds `other` below the calling matrix.
    pub fn vcat(&self, other: Self) -> Self {
        assert_eq!(self.ncols, other.ncols);
        let mut new_vec = self.data.clone();
        new_vec.extend(other.data);
        Matrix::new(new_vec, self.nrows + other.nrows, self.ncols)
    }

    /// Repeat self horizontally. That is, make `n` total copies of self and horizontally
    /// concatenate them all together.
    pub fn hrepeat(&self, n: usize) -> Self {
        let total_cols = self.ncols * n;
        let mut new_vec = Vec::with_capacity(self.nrows * total_cols);
        for i in 0..self.nrows {
            for _ in 0..n {
                new_vec.extend(&self[i]);
            }
        }
        Matrix::new(new_vec, self.nrows, total_cols)
    }

    /// Repeat self vertically. That is, make `n` total copies of self and vertically
    /// concatenate them all together.
    pub fn vrepeat(&self, n: usize) -> Self {
        let total_rows = self.nrows * n;
        Matrix::new(self.data.repeat(n), total_rows, self.ncols)
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter) -> Result {
        Ok(for (rownum, row) in self.into_iter().enumerate() {
            let row_str = format!("{:?}", row);
            if rownum == 0 {
                writeln!(f, "[{} ", row_str)?;
            } else if rownum == self.nrows - 1 {
                writeln!(f, " {}]", row_str)?;
            } else {
                writeln!(f, " {} ", row_str)?;
            }
        })
    }
}

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

macro_rules! mat_mat_op {
    ($($path:ident)::+, $fn:ident, $innerfn:ident) => {
        impl $($path)::+<Matrix> for Matrix {
            type Output = Matrix;

            fn $fn(self, other: Matrix) -> Self::Output {
                assert_eq!(self.shape(), other.shape(), "matrix shapes not equal");
                Matrix::new(
                    $innerfn(&self.data, &other.data),
                    self.nrows,
                    self.ncols
                )
            }
        }

        impl $($path)::+<&Matrix> for &Matrix {
            type Output = Matrix;

            fn $fn(self, other: &Matrix) -> Self::Output {
                assert_eq!(self.shape(), other.shape(), "matrix shapes not equal");
                Matrix::new(
                    $innerfn(&self.data, &other.data),
                    self.nrows,
                    self.ncols
                )
            }
        }

        impl $($path)::+<&Matrix> for Matrix {
            type Output = Matrix;

            fn $fn(self, other: &Matrix) -> Self::Output {
                assert_eq!(self.shape(), other.shape(), "matrix shapes not equal");
                Matrix::new(
                    $innerfn(&self.data, &other.data),
                    self.nrows,
                    self.ncols
                )
            }
        }

        impl $($path)::+<Matrix> for &Matrix {
            type Output = Matrix;

            fn $fn(self, other: Matrix) -> Self::Output {
                assert_eq!(self.shape(), other.shape(), "matrix shapes not equal");
                Matrix::new(
                    $innerfn(&self.data, &other.data),
                    self.nrows,
                    self.ncols
                )
            }
        }
    };
}

macro_rules! mat_mat_opassign {
    ($($path:ident)::+, $fn:ident, $innerfn:ident) => {
        impl $($path)::+<Matrix> for Matrix {
            fn $fn(&mut self, other: Matrix) {
                assert_eq!(self.shape(), other.shape(), "matrix shapes not equal");
                $innerfn(&mut self.data, &other.data);
            }
        }

        impl $($path)::+<&Matrix> for Matrix {
            fn $fn(&mut self, other: &Matrix) {
                assert_eq!(self.shape(), other.shape(), "matrix shapes not equal");
                $innerfn(&mut self.data, &other.data);
            }
        }
    };
}

macro_rules! mat_opassign {
    ($($path:ident)::+, $fn:ident, $innerfn:ident, $ty:ty) => {
        impl $($path)::+<$ty> for Matrix {
            fn $fn(&mut self, other: $ty) {
                $innerfn(&mut self.data, other);
            }
        }
    }
}

macro_rules! mat_op {
    ($($path:ident)::+, $fn:ident, $fn_vs:ident, $fn_sv:ident, $ty:ty) => {
        // impl ops::Add::add for Matrix
        impl $($path)::+<$ty> for Matrix {
            type Output = Matrix;

            // fn add(self, other: f32) -> Self::Output
            fn $fn(self, other: $ty) -> Self::Output {
                Matrix::new(
                    $fn_vs(&self.data, other),
                    self.nrows,
                    self.ncols
                )
            }
        }

        impl $($path)::+<$ty> for &Matrix {
            type Output = Matrix;

            fn $fn(self, other: $ty) -> Self::Output {
                Matrix::new(
                    $fn_vs(&self.data, other),
                    self.nrows,
                    self.ncols
                )
            }
        }

        impl $($path)::+<Matrix> for $ty {
            type Output = Matrix;

            fn $fn(self, other: Matrix) -> Self::Output {
                Matrix::new(
                    $fn_sv(self, &other.data),
                    other.nrows,
                    other.ncols
                )
            }
        }

        impl $($path)::+<&Matrix> for $ty {
            type Output = Matrix;

            fn $fn(self, other: &Matrix) -> Self::Output {
                Matrix::new(
                    $fn_sv(self, &other.data),
                    other.nrows,
                    other.ncols
                )
            }
        }
    }
}

macro_rules! mat_op_for {
    ($ty: ty) => {
        mat_op!(ops::Add, add, vsadd, svadd, $ty);
        mat_op!(ops::Sub, sub, vssub, svsub, $ty);
        mat_op!(ops::Mul, mul, vsmul, svmul, $ty);
        mat_op!(ops::Div, div, vsdiv, svdiv, $ty);
        mat_opassign!(ops::AddAssign, add_assign, vsadd_mut, $ty);
        mat_opassign!(ops::SubAssign, sub_assign, vssub_mut, $ty);
        mat_opassign!(ops::MulAssign, mul_assign, vsmul_mut, $ty);
        mat_opassign!(ops::DivAssign, div_assign, vsdiv_mut, $ty);
    };
}

mat_mat_op!(ops::Add, add, vadd);
mat_mat_op!(ops::Sub, sub, vsub);
mat_mat_op!(ops::Mul, mul, vmul);
mat_mat_op!(ops::Div, div, vdiv);
mat_mat_opassign!(ops::AddAssign, add_assign, vadd_mut);
mat_mat_opassign!(ops::SubAssign, sub_assign, vsub_mut);
mat_mat_opassign!(ops::MulAssign, mul_assign, vmul_mut);
mat_mat_opassign!(ops::DivAssign, div_assign, vdiv_mut);
mat_op_for!(f64);

macro_rules! impl_unaryops_matrix {
    ($op: ident) => {
        impl Matrix {
            #[doc = "Apply the `f64` operation `"]
            #[doc = stringify!($op)]
            #[doc = "` element-wise to the matrix."]
            pub fn $op(&self) -> Self {
                Self::new(self.data.$op(), self.nrows, self.ncols)
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
                Self::new(self.data.$op(arg), self.nrows, self.ncols)
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
