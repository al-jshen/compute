use std::ops::Index;

use crate::prelude::{is_square, is_symmetric};

use super::Vector;

#[derive(Debug, Clone)]
pub struct Matrix {
    data: Vector,
    nrows: usize,
    ncols: usize,
    is_square: bool,
    is_symmetric: bool,
}

impl Matrix {
    pub fn empty() -> Self {
        Self {
            data: Vector::empty(),
            ncols: 0,
            nrows: 0,
            is_square: false,
            is_symmetric: false,
        }
    }

    pub fn new(data: Vector, [nrows, ncols]: [usize; 2]) -> Self {
        let is_square = match is_square(&data) {
            Ok(val) => {
                assert!(nrows == ncols && nrows == val, "matrix not square");
                true
            }
            Err(_) => false,
        };
        let is_symmetric = if is_square {
            is_symmetric(&data)
        } else {
            false
        };

        Self {
            data,
            ncols,
            nrows,
            is_square,
            is_symmetric,
        }
    }
}

impl Default for Matrix {
    fn default() -> Self {
        Self::empty()
    }
}

impl Index<[usize; 2]> for Matrix {
    type Output = f64;

    fn index(&self, [i, j]: [usize; 2]) -> &Self::Output {
        assert!(i < self.nrows && j < self.ncols);
        &self.data[i * self.ncols + j]
    }
}
