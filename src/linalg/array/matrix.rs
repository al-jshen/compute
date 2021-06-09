use std::{
    fmt::{Display, Formatter, Result},
    mem::swap,
    ops::{Index, IndexMut, Neg},
    panic,
};

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::prelude::{transpose, Dot};

use super::super::utils::{dot, ipiv_parity};
use super::vops::*;
use super::{broadcast_add, broadcast_div, broadcast_mul, broadcast_sub, Vector};

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
            nrows: 0,
            ncols: 0,
        }
    }

    /// Make a matrix filled with zeros.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self::new(Vector::zeros(nrows * ncols), nrows as i32, ncols as i32)
    }

    /// Make a matrix filled with ones.
    pub fn ones(nrows: usize, ncols: usize) -> Self {
        Self::new(Vector::ones(nrows * ncols), nrows as i32, ncols as i32)
    }

    /// Make an identity matrix with size `dims`.
    pub fn eye(dims: usize) -> Self {
        let mut m = Self::zeros(dims, dims);
        for i in 0..dims {
            m.data[i * dims + i] = 1.;
        }
        m
    }

    /// Check whether the matrix is square.
    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// Check whether the matrix is symmetric.
    pub fn is_symmetric(&self) -> bool {
        if self.is_square() {
            for i in 0..self.nrows {
                for j in i..self.ncols {
                    if (self.data[i * self.ncols + j] - self.data[j * self.nrows + i]).abs()
                        > f64::EPSILON
                    {
                        return false;
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Check whether a matrix is close to another matrix within some tolerance.
    pub fn close_to(&self, other: &Matrix, tol: f64) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        self.data.close_to(&other.data, tol)
    }

    /// Check whether the matrix is positive definite.
    pub fn is_positive_definite(&self) -> bool {
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

    /// Solve a matrix equation of the form Lx=b, where L is a lower triangular matrix.
    /// See the [Wikipedia page](https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution).
    pub fn forward_substitution(&self, b: &[f64]) -> Vector {
        assert!(self.is_lower_triangular(), "matrix not lower triangular");
        assert_eq!(b.len(), self.nrows);
        let mut x = Vector::zeros(self.nrows);
        for i in 0..self.ncols {
            x[i] = (b[i] - dot(&self[i][0..i], &x[..i])) / self[[i, i]];
        }
        x
    }

    /// Solve a matrix equation of the form Ux=b, where U is an upper triangular matrix.
    /// See the [Wikipedia page](https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution).
    pub fn backward_substitution(&self, b: &[f64]) -> Vector {
        assert!(self.is_upper_triangular(), "matrix not upper triangular");
        assert_eq!(b.len(), self.nrows);
        let mut x = Vector::zeros(self.nrows);
        for i in (0..self.ncols).rev() {
            x[i] = (b[i] - dot(&self[i][i + 1..self.ncols], &x[i + 1..])) / self[[i, i]];
        }
        x
    }

    /// Return the Cholesky decomposition of the matrix. Resulting matrix is lower triangular.
    pub fn cholesky(&self) -> Matrix {
        assert!(self.is_positive_definite(), "matrix not positive definite");

        let mut l = Matrix::zeros(self.nrows, self.ncols);

        for i in 0..self.ncols {
            for j in 0..(i + 1) {
                let s = l.get_row_as_vector(j).dot(l.get_row_as_vector(i));

                if i == j {
                    l[[i, j]] = (self[[i, i]] - s).sqrt();
                } else {
                    l[[i, j]] = (self[[i, j]] - s) / l[[j, j]];
                }
            }
        }

        l
    }

    pub fn lu(&self) -> (Matrix, Vec<i32>) {
        assert!(self.is_square(), "matrix not square");

        let mut lu = self.clone();

        let n = self.nrows;
        let mut pivots: Vec<i32> = (0..n).map(|x| x as i32).collect();

        for j in 0..n {
            for i in 0..n {
                let mut s = 0.;
                for k in 0..i.min(j) {
                    s += lu[[i, k]] * lu[[k, j]];
                }
                lu[[i, j]] -= s;
            }

            let mut p = j;
            for i in (j + 1)..n {
                if lu[[i, j]].abs() > lu[[p, j]].abs() {
                    p = i;
                }
            }

            if p != j {
                for k in 0..n {
                    lu.data.swap(p * n + k, j * n + k)
                }
                pivots.swap(p, j);
            }

            if j < n && lu[[j, j]] != 0. {
                for i in (j + 1)..n {
                    lu[[i, j]] /= lu[[j, j]];
                }
            }
        }

        (lu, pivots)
    }

    /// Calculates the determinant of the matrix using the LU decomposition.
    pub fn det(&self) -> f64 {
        let (lu, p) = self.lu();
        lu.diag().prod() * ipiv_parity(&p) as f64
    }

    /// Calculates the determinant of an LU decomposed matrix.
    pub fn lu_det(&self, piv: &[i32]) -> f64 {
        assert!(self.is_square(), "matrix not square");
        self.diag().prod() * ipiv_parity(piv) as f64
    }

    /// Check whether the matrix is upper triangular (i.e., all the entries below the diagonal are
    /// 0).
    pub fn is_upper_triangular(&self) -> bool {
        for i in 0..self.nrows {
            for j in 0..i {
                if self[i][j] != 0. {
                    return false;
                }
            }
        }
        true
    }

    /// Check whether the matrix is lower triangular (i.e., all the entries above the diagonal are
    /// 0).
    pub fn is_lower_triangular(&self) -> bool {
        for i in 0..self.nrows {
            for j in (i + 1)..self.ncols {
                if self[i][j] != 0. {
                    return false;
                }
            }
        }
        true
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
    pub fn new<T>(data: T, nrows: i32, ncols: i32) -> Self
    where
        T: Into<Vector>,
    {
        let v = data.into();
        let len = v.len();

        let mut m = Self {
            data: v,
            nrows: 1,
            ncols: len,
        };

        m.reshape_mut(nrows, ncols);

        m
    }

    /// Get the number of rows and columns in the matrix.
    pub fn shape(&self) -> [usize; 2] {
        [self.nrows, self.ncols]
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

    /// Reshape the matrix. A size of `-1` in either the rows or the columns means that the size
    /// for that dimension will be automatically determined if possible.
    pub fn reshape(self, nrows: i32, ncols: i32) -> Self {
        let size = self.size() as i32;
        if nrows > 0 && ncols > 0 {
            assert_eq!(nrows * ncols, size as i32, "invalid shape");
            Matrix::new(self.data, nrows, ncols)
        } else if nrows < 0 {
            assert!(nrows == -1 && ncols > 0, "invalid shape");
            // automatically determine number of rows
            Matrix::new(self.data, size / ncols, ncols)
        } else if ncols < 0 {
            assert!(ncols == -1 && nrows > 0, "invalid shape");
            // automatically determine number of columns
            Matrix::new(self.data, nrows, size / nrows)
        } else {
            panic!("invalid shape");
        }
    }

    /// Apply a closure to every element in a row. The closure should take a value and return the
    /// value to replace it with.
    pub fn apply_along_row<F>(&mut self, row: usize, f: F)
    where
        F: Fn(f64) -> f64,
    {
        self[row].iter_mut().for_each(|x| *x = f(*x));
    }

    /// Apply a closure to every element in a row.
    pub fn apply_along_col<F>(&mut self, col: usize, f: F)
    where
        F: Fn(f64) -> f64,
    {
        for row in self {
            row[col] = f(row[col]);
        }
    }

    /// Return a copy of the row of the matrix as a Vector.
    pub fn get_row_as_vector(&self, row: usize) -> Vector {
        assert!(row < self.nrows);
        Vector::from(&self[row])
    }

    /// Return a copy of the column of the matrix as a Vector.
    pub fn get_col_as_vector(&self, col: usize) -> Vector {
        assert!(col < self.ncols);

        let mut v = Vector::zeros(self.nrows);

        for i in 0..self.nrows {
            v[i] = self[i][col];
        }

        v
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
                sums[col] += self[row][col];
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
        Matrix::new(t, self.ncols as i32, self.nrows as i32)
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
    pub fn to_vec(self) -> Vector {
        self.data
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
        Matrix::new(
            new_vec,
            self.nrows as i32,
            (self.ncols + other.ncols) as i32,
        )
    }

    /// Vertical concatenation of matrices. Adds `other` below the calling matrix.
    pub fn vcat(&self, other: Self) -> Self {
        assert_eq!(self.ncols, other.ncols);
        let mut new_vec = self.data.clone();
        new_vec.extend(other.data);
        Matrix::new(
            new_vec,
            (self.nrows + other.nrows) as i32,
            self.ncols as i32,
        )
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
        Matrix::new(new_vec, self.nrows as i32, total_cols as i32)
    }

    /// Repeat self vertically. That is, make `n` total copies of self and vertically
    /// concatenate them all together.
    pub fn vrepeat(&self, n: usize) -> Self {
        let total_rows = self.nrows * n;
        Matrix::new(self.data.repeat(n), total_rows as i32, self.ncols as i32)
    }
}

pub trait Solve<T> {
    fn cholesky_solve(&self, system: &T) -> T;
    fn lu_solve(&self, pivots: &[i32], system: &T) -> T;
    fn solve(&self, system: &T) -> T;
}

impl Solve<Vector> for Matrix {
    fn cholesky_solve(&self, system: &Vector) -> Vector {
        assert!(
            self.is_lower_triangular(),
            "matrix not a cholesky decomposed matrix"
        );
        assert_eq!(self.nrows, system.len());
        let y = self.forward_substitution(system);

        // back substitution
        self.t().backward_substitution(&y)
    }

    /// Solve the linear system Ax = b given a LU decomposed matrix A. The first argument should be a
    /// tuple, where the first element is the LU decomposed matrix and the second element is the pivots
    /// P.
    fn lu_solve(&self, pivots: &[i32], system: &Vector) -> Vector {
        assert!(self.is_square(), "matrix not square");
        assert_eq!(self.nrows, system.len());

        let mut x = Vector::zeros(self.nrows);

        for i in 0..pivots.len() {
            x[i] = system[pivots[i] as usize];
        }

        for k in 0..self.ncols {
            for i in (k + 1)..self.ncols {
                x[i] -= x[k] * self[[i, k]];
            }
        }

        for k in (0..self.ncols).rev() {
            x[k] /= self[[k, k]];
            for i in 0..k {
                x[i] -= x[k] * self[[i, k]];
            }
        }

        x
    }

    /// Solve the linear system Ax = b using LU decomposition.
    fn solve(&self, system: &Vector) -> Vector {
        let (lu, piv) = self.lu();
        lu.lu_solve(&piv, system)
    }
}

impl Solve<Matrix> for Matrix {
    fn cholesky_solve(&self, system: &Matrix) -> Matrix {
        let mut solutions = Vector::with_capacity(system.nrows * system.ncols);
        for i in 0..system.ncols {
            let x = system.get_col_as_vector(i);
            solutions.extend(self.cholesky_solve(&x));
        }

        Matrix::new(solutions, system.ncols as i32, system.nrows as i32).t()
    }

    fn lu_solve(&self, pivots: &[i32], system: &Matrix) -> Matrix {
        let mut solutions = Vector::with_capacity(system.nrows * system.ncols);
        for i in 0..system.ncols {
            let x = system.get_col_as_vector(i);
            solutions.extend(self.lu_solve(&pivots, &x));
        }

        Matrix::new(solutions, system.ncols as i32, system.nrows as i32).t()
    }

    fn solve(&self, system: &Matrix) -> Matrix {
        let (lu, piv) = self.lu();
        lu.lu_solve(&piv, system)
    }
}

impl Neg for Matrix {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Matrix::new(-self.data, self.nrows as i32, self.ncols as i32)
    }
}

impl PartialEq<Matrix> for Matrix {
    fn eq(&self, other: &Matrix) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        self.data.eq(&other.data)
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter) -> Result {
        for (rownum, row) in self.into_iter().enumerate() {
            let row_str = format!("{:?}", row);
            if rownum == 0 {
                writeln!(f, "[{},", row_str)?
            } else if rownum == self.nrows - 1 {
                writeln!(f, " {}]", row_str)?
            } else {
                writeln!(f, " {},", row_str)?
            }
        }
        Ok(())
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

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        assert!(i < self.nrows);
        &mut self.data[i * self.ncols..(i + 1) * self.ncols]
    }
}

impl IndexMut<[usize; 2]> for Matrix {
    fn index_mut(&mut self, [i, j]: [usize; 2]) -> &mut Self::Output {
        assert!(i < self.nrows && j < self.ncols);
        &mut self.data[i * self.ncols + j]
    }
}

// Functions to do Matrix + Matrix, Matrix - Matrix, etc.
macro_rules! makefn_matops {
    ($fn: ident, $innerfn: ident) => {
        pub fn $fn(m1: &Matrix, m2: &Matrix) -> Matrix {
            assert_eq!(m1.shape(), m2.shape(), "matrix shapes not equal");
            Matrix::new(
                $innerfn(&m1.data, &m2.data),
                m1.nrows as i32,
                m1.ncols as i32,
            )
        }
    };
}

// Make the Matrix-Matrix functions.
makefn_matops!(matmatadd, vadd);
makefn_matops!(matmatsub, vsub);
makefn_matops!(matmatmul, vmul);
makefn_matops!(matmatdiv, vdiv);

// Helper macro to implement Matrix + Matrix, Matrix + &Matrix, etc...
macro_rules! impl_mat_mat_op_helper {
    ($op: ident, $fn: ident, $innerfn: ident, $selftype: ty, $othertype: ty) => {
        impl $op<$othertype> for $selftype {
            type Output = Matrix;

            fn $fn(self, other: $othertype) -> Self::Output {
                $innerfn(&self, &other)
            }
        }
    };
}

macro_rules! impl_mat_vec_op_helper {
    ($op: ident, $fn: ident, $innerfn: ident, $selftype: ty, $othertype: ty$(, $to_owned: ident)?) => {
        impl $op<$othertype> for $selftype {
            type Output = Matrix;

            fn $fn(self, other: $othertype) -> Self::Output {
                $innerfn(&self, &other$(.$to_owned())?.to_matrix())
            }
        }
    };
}

macro_rules! impl_vec_mat_op_helper {
    ($op: ident, $fn: ident, $innerfn: ident, $selftype: ty, $othertype: ty$(, $to_owned: ident)?) => {
        impl $op<$othertype> for $selftype {
            type Output = Matrix;

            fn $fn(self, other: $othertype) -> Self::Output {
                $innerfn(&self$(.$to_owned())?.to_matrix(), &other)
            }
        }
    };
}

// Implement Matrix + Matrix, Matrix + &Matrix, etc...
macro_rules! mat_mat_op {
    ($op:ident, $fn:ident, $innerfn:ident) => {
        impl_mat_mat_op_helper!($op, $fn, $innerfn, Matrix, Matrix);
        impl_mat_mat_op_helper!($op, $fn, $innerfn, Matrix, &Matrix);
        impl_mat_mat_op_helper!($op, $fn, $innerfn, &Matrix, Matrix);
        impl_mat_mat_op_helper!($op, $fn, $innerfn, &Matrix, &Matrix);
    };
}

macro_rules! mat_vec_op {
    ($op:ident, $fn:ident, $innerfn:ident) => {
        impl_mat_vec_op_helper!($op, $fn, $innerfn, Matrix, Vector);
        impl_mat_vec_op_helper!($op, $fn, $innerfn, Matrix, &Vector, to_owned);
        impl_mat_vec_op_helper!($op, $fn, $innerfn, &Matrix, Vector);
        impl_mat_vec_op_helper!($op, $fn, $innerfn, &Matrix, &Vector, to_owned);
    };
}

macro_rules! vec_mat_op {
    ($op:ident, $fn:ident, $innerfn:ident) => {
        impl_vec_mat_op_helper!($op, $fn, $innerfn, Vector, Matrix);
        impl_vec_mat_op_helper!($op, $fn, $innerfn, Vector, &Matrix);
        impl_vec_mat_op_helper!($op, $fn, $innerfn, &Vector, Matrix, to_owned);
        impl_vec_mat_op_helper!($op, $fn, $innerfn, &Vector, &Matrix, to_owned);
    };
}

// Macro to implement Matrix += Matrix, Matrix += &Matrix, etc...
macro_rules! mat_mat_opassign {
    ($op: ident, $fn:ident, $innerfn:ident) => {
        impl $op<Matrix> for Matrix {
            fn $fn(&mut self, other: Matrix) {
                assert_eq!(self.shape(), other.shape(), "matrix shapes not equal");
                $innerfn(&mut self.data, &other.data);
            }
        }

        impl $op<&Matrix> for Matrix {
            fn $fn(&mut self, other: &Matrix) {
                assert_eq!(self.shape(), other.shape(), "matrix shapes not equal");
                $innerfn(&mut self.data, &other.data);
            }
        }
    };
}

// Macro to implement Matrix += Scalar
macro_rules! mat_scalar_opassign {
    ($op: ident, $fn:ident, $innerfn:ident, $ty:ty) => {
        impl $op<$ty> for Matrix {
            fn $fn(&mut self, other: $ty) {
                $innerfn(&mut self.data, other);
            }
        }
    };
}

// Helper macro to implement Matrix + Scalar, &Matrix + Scalar, Scalar + Matrix, Scalar + &Matrix
macro_rules! impl_mat_scalar_op_for_type {
    ($op:ident, $fn:ident, $fn_vs:ident, $fn_sv:ident, $selftype:ty, $othertype: ty) => {
        impl $op<$othertype> for $selftype {
            type Output = Matrix;

            fn $fn(self, other: $othertype) -> Self::Output {
                Matrix::new(
                    $fn_vs(&self.data, other),
                    self.nrows as i32,
                    self.ncols as i32,
                )
            }
        }

        impl $op<$selftype> for $othertype {
            type Output = Matrix;
            fn $fn(self, other: $selftype) -> Self::Output {
                Matrix::new(
                    $fn_sv(self, &other.data),
                    other.nrows as i32,
                    other.ncols as i32,
                )
            }
        }
    };
}

// Implement Matrix + Scalar, &Matrix + Scalar, Scalar + Matrix, Scalar + &Matrix
macro_rules! mat_scalar_op {
    ($op:ident, $fn:ident, $fn_vs:ident, $fn_sv:ident, $ty:ty) => {
        impl_mat_scalar_op_for_type!($op, $fn, $fn_vs, $fn_sv, Matrix, $ty);
        impl_mat_scalar_op_for_type!($op, $fn, $fn_vs, $fn_sv, &Matrix, $ty);
    };
}

// Implement Matrix-Scalar and Scalar-Matrix Op and OpAssign methods.
macro_rules! mat_scalar_op_for {
    ($ty: ty) => {
        mat_scalar_op!(Add, add, vsadd, svadd, $ty);
        mat_scalar_op!(Sub, sub, vssub, svsub, $ty);
        mat_scalar_op!(Mul, mul, vsmul, svmul, $ty);
        mat_scalar_op!(Div, div, vsdiv, svdiv, $ty);
        mat_scalar_opassign!(AddAssign, add_assign, vsadd_mut, $ty);
        mat_scalar_opassign!(SubAssign, sub_assign, vssub_mut, $ty);
        mat_scalar_opassign!(MulAssign, mul_assign, vsmul_mut, $ty);
        mat_scalar_opassign!(DivAssign, div_assign, vsdiv_mut, $ty);
    };
}

macro_rules! impl_mat_ops {
    ($($macro_name:ident),+) => {
        $(
            $macro_name!(Add, add, broadcast_add);
            $macro_name!(Sub, sub, broadcast_sub);
            $macro_name!(Mul, mul, broadcast_mul);
            $macro_name!(Div, div, broadcast_div);
        )+
    };
}

impl_mat_ops!(mat_mat_op, mat_vec_op, vec_mat_op);

// Implement Matrix-Matrix assignment operations.
mat_mat_opassign!(AddAssign, add_assign, vadd_mut);
mat_mat_opassign!(SubAssign, sub_assign, vsub_mut);
mat_mat_opassign!(MulAssign, mul_assign, vmul_mut);
mat_mat_opassign!(DivAssign, div_assign, vdiv_mut);
// Implement Matrix-Scalar (assignment) operations.
mat_scalar_op_for!(f64);

macro_rules! impl_unary_ops_matrix {
    ($($op: ident),+) => {
        $(
            impl Matrix {
                #[doc = "Apply the `f64` operation `"]
                #[doc = stringify!($op)]
                #[doc = "` element-wise to the matrix."]
                pub fn $op(&self) -> Self {
                    Self::new(self.data.$op(), self.nrows as i32, self.ncols as i32)
                }
            }
        )+
    };
}

impl_unary_ops_matrix!(
    ln, ln_1p, log10, log2, exp, exp2, exp_m1, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan,
    asinh, acosh, atanh, sqrt, cbrt, abs, floor, ceil, to_radians, to_degrees, recip, round,
    signum
);

macro_rules! impl_unaryops_with_arg_matrix {
    ($op: ident, $argtype: ident) => {
        impl Matrix {
            #[doc = "Apply the `f64` operation `"]
            #[doc = stringify!($op)]
            #[doc = "` element-wise to the matrix."]
            pub fn $op(&self, arg: $argtype) -> Self {
                Self::new(self.data.$op(arg), self.nrows as i32, self.ncols as i32)
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

impl_reduction_fns_matrix!(norm, max, mean, min, std, sum, prod, var, sample_std, sample_var);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_vec_ops() {
        let a = Matrix::new(
            [
                1.69968, -1.48008, 0.66542, -0.60947, 1.6508, -0.15097, -0.11973, 0.21233, 0.80922,
            ],
            3,
            3,
        );
        let b = Vector::new([1.93579, 0.43448, -0.73636]);
        let c = &a + &b;
        assert_eq!(
            c,
            Matrix::new(
                [
                    3.6354699999999998,
                    -1.0456,
                    -0.07094,
                    1.32632,
                    2.08528,
                    -0.88733,
                    1.8160599999999998,
                    0.64681,
                    0.07286000000000004
                ],
                3,
                3
            )
        );
        let d = &a / &b;
        assert_eq!(
            d,
            Matrix::new(
                [
                    0.878029125060053,
                    -3.406554962253729,
                    -0.9036612526481612,
                    -0.31484303565985977,
                    3.7994844411710553,
                    0.2050220001086425,
                    -0.061850717278217164,
                    0.48869913459768,
                    -1.0989461676353958
                ],
                3,
                3
            )
        );
        let e = &a.t() * &b;
        assert_eq!(
            e,
            Matrix::new(
                [
                    3.2902235472,
                    -0.26480252559999995,
                    0.08816438280000001,
                    -2.8651240632,
                    0.717239584,
                    -0.1563513188,
                    1.2881133818,
                    -0.0655934456,
                    -0.5958772392
                ],
                3,
                3
            )
        );
    }

    #[test]
    fn test_cholesky() {
        let a = Matrix::new(
            [
                8.97062104740134,
                0.26943982630456786,
                -0.9534319972273332,
                0.26943982630456786,
                3.307274425269507,
                -1.5063311267171873,
                -0.9534319972273332,
                -1.5063311267171873,
                1.5071780730242237,
            ],
            3,
            3,
        );
        let ac = a.cholesky();
        assert!(ac.close_to(
            &Matrix::new(
                [
                    2.9950995054257112,
                    0.,
                    0.,
                    0.0899602253001844,
                    1.8163649366615309,
                    0.,
                    -0.3183306582970492,
                    -0.813544678798316,
                    0.8625478077250766
                ],
                3,
                3
            ),
            1e-10
        ));

        let b = Vector::new([0.7220683901726338, -0.06367193965727952, 1.0077206300677382]);
        let x = ac.cholesky_solve(&b);
        assert!(x.close_to(
            &Vector::new([0.21181290359830895, 0.6039812818017399, 1.4062476574275615]),
            1e-10
        ));
    }
}
