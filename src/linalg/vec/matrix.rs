use std::{
    array::IntoIter,
    fmt::{Display, Formatter, Result},
    iter::Map,
    ops::Index,
    ops::Range,
    slice::Chunks,
};

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

    pub fn row(&self, row: usize) -> Vector {
        Vector::from(&self[row])
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
