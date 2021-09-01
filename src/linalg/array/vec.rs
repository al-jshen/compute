use super::{vops::*, Matrix};
use crate::linalg::{logmeanexp, logsumexp, norm, prod, sum};
use crate::statistics::{argmax, argmin, max, mean, min, sample_std, sample_var, std, var};
use approx_eq::rel_diff;
use std::convert::From;
use std::fmt::{Display, Formatter, Result};
use std::iter::{FromIterator, IntoIterator};
use std::ops::{self, Deref, DerefMut, Neg};

/// A row-major ordering vector struct with various useful methods.
#[derive(Debug, Clone)]
pub struct Vector {
    v: Vec<f64>,
}

impl Vector {
    pub fn empty() -> Self {
        Self { v: Vec::new() }
    }

    pub fn new<T>(v: T) -> Self
    where
        T: Into<Vec<f64>>,
    {
        Self { v: v.into() }
    }

    pub fn data(&self) -> &[f64] {
        &self.v
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            v: Vec::with_capacity(n),
        }
    }

    pub fn empty_n(n: usize) -> Self {
        let mut v = Vec::with_capacity(n);
        unsafe {
            v.set_len(n);
        }
        Self { v }
    }

    pub fn sort(&mut self) {
        self.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    pub fn sorted(&self) -> Self {
        let mut x = self.clone();
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        x
    }

    pub fn zeros(n: usize) -> Self {
        Self { v: vec![0.; n] }
    }

    pub fn ones(n: usize) -> Self {
        Self { v: vec![1.; n] }
    }

    pub fn to_matrix(self) -> Matrix {
        let n = self.len();
        Matrix::new(self, 1, n as i32)
    }

    pub fn reshape(&self, nrows: i32, ncols: i32) -> Matrix {
        Matrix::new(self.clone(), nrows, ncols)
    }

    pub fn close_to(&self, other: &Vector, tol: f64) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for i in 0..self.len() {
            if rel_diff(self[i], other[i]) > tol {
                return false;
            }
        }
        true
    }
}

impl Default for Vector {
    fn default() -> Self {
        Self::empty()
    }
}

impl PartialEq<Vector> for Vector {
    fn eq(&self, other: &Vector) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for i in 0..self.len() {
            if (self[i] - other[i]).abs() > f64::EPSILON {
                return false;
            }
        }
        true
    }
}

impl<T> From<T> for Vector
where
    T: Into<Vec<f64>>,
{
    fn from(v: T) -> Self {
        Self { v: v.into() }
    }
}

impl AsRef<[f64]> for Vector {
    fn as_ref(&self) -> &[f64] {
        &self.v
    }
}

impl Display for Vector {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "{:?}", self.v)
    }
}

impl IntoIterator for Vector {
    type Item = f64;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.v.into_iter()
    }
}

impl<'a> IntoIterator for &'a Vector {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.v.as_slice().iter()
    }
}

impl<'a> IntoIterator for &'a mut Vector {
    type Item = &'a mut f64;
    type IntoIter = std::slice::IterMut<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.v.iter_mut()
    }
}

impl FromIterator<f64> for Vector {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = f64>,
    {
        Self {
            v: Vec::from_iter(iter),
        }
    }
}

impl Extend<f64> for Vector {
    fn extend<T: IntoIterator<Item = f64>>(&mut self, iter: T) {
        self.v.extend(iter);
    }
}

impl Deref for Vector {
    type Target = Vec<f64>;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl DerefMut for Vector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}

impl Neg for Vector {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.v.into_iter().map(|x| -x).collect()
    }
}

macro_rules! vec_vec_op {
    ($($path:ident)::+, $fn:ident, $innerfn:ident) => {
        impl $($path)::+<Vector> for Vector {
            type Output = Vector;

            fn $fn(self, other: Vector) -> Self::Output {
                Vector {
                    v: $innerfn(&self, &other)
                }
            }
        }

        impl $($path)::+<&Vector> for &Vector {
            type Output = Vector;

            fn $fn(self, other: &Vector) -> Self::Output {
                Vector {
                    v: $innerfn(&self, &other)
                }
            }
        }

        impl $($path)::+<&Vector> for Vector {
            type Output = Vector;

            fn $fn(self, other: &Vector) -> Self::Output {
                Vector {
                    v: $innerfn(&self, &other)
                }
            }
        }

        impl $($path)::+<Vector> for &Vector {
            type Output = Vector;

            fn $fn(self, other: Vector) -> Self::Output {
                Vector {
                    v: $innerfn(&self, &other)
                }
            }
        }
    };
}

macro_rules! vec_vec_opassign {
    ($($path:ident)::+, $fn:ident, $innerfn:ident) => {
        impl $($path)::+<Vector> for Vector {
            fn $fn(&mut self, other: Vector) {
                $innerfn(self, &other);
            }
        }

        impl $($path)::+<&Vector> for Vector {
            fn $fn(&mut self, other: &Vector) {
                $innerfn(self, &other);
            }
        }
    };
}

macro_rules! vec_opassign {
    ($($path:ident)::+, $fn:ident, $innerfn:ident, $ty:ty) => {
        impl $($path)::+<$ty> for Vector {
            fn $fn(&mut self, other: $ty) {
                $innerfn(self, other);
            }
        }
    }
}

macro_rules! vec_op {
    ($($path:ident)::+, $fn:ident, $fn_vs:ident, $fn_sv:ident, $ty:ty) => {
        // impl ops::Add::add for Vector
        impl $($path)::+<$ty> for Vector {
            type Output = Vector;

            // fn add(self, other: f32) -> Self::Output
            fn $fn(self, other: $ty) -> Self::Output {
                Vector {
                    v: $fn_vs(&self, other)
                }
            }
        }

        impl $($path)::+<$ty> for &Vector {
            type Output = Vector;

            fn $fn(self, other: $ty) -> Self::Output {
                Vector {
                    v: $fn_vs(&self, other)
                }
            }
        }

        impl $($path)::+<Vector> for $ty {
            type Output = Vector;

            fn $fn(self, other: Vector) -> Self::Output {
                Vector {
                    v: $fn_sv(self, &other)
                }
            }
        }

        impl $($path)::+<&Vector> for $ty {
            type Output = Vector;

            fn $fn(self, other: &Vector) -> Self::Output {
                Vector {
                    v: $fn_sv(self, &other)
                }
            }
        }
    }
}

macro_rules! vec_op_for {
    ($ty: ty) => {
        vec_op!(ops::Add, add, vsadd, svadd, $ty);
        vec_op!(ops::Sub, sub, vssub, svsub, $ty);
        vec_op!(ops::Mul, mul, vsmul, svmul, $ty);
        vec_op!(ops::Div, div, vsdiv, svdiv, $ty);
        vec_opassign!(ops::AddAssign, add_assign, vsadd_mut, $ty);
        vec_opassign!(ops::SubAssign, sub_assign, vssub_mut, $ty);
        vec_opassign!(ops::MulAssign, mul_assign, vsmul_mut, $ty);
        vec_opassign!(ops::DivAssign, div_assign, vsdiv_mut, $ty);
    };
}

vec_vec_op!(ops::Add, add, vadd);
vec_vec_op!(ops::Sub, sub, vsub);
vec_vec_op!(ops::Mul, mul, vmul);
vec_vec_op!(ops::Div, div, vdiv);
vec_vec_opassign!(ops::AddAssign, add_assign, vadd_mut);
vec_vec_opassign!(ops::SubAssign, sub_assign, vsub_mut);
vec_vec_opassign!(ops::MulAssign, mul_assign, vmul_mut);
vec_vec_opassign!(ops::DivAssign, div_assign, vdiv_mut);
vec_op_for!(f64);

macro_rules! impl_unaryops_vector {
    ($fn: ident, $op: ident) => {
        impl Vector {
            #[doc = "Apply the `f64` operation `"]
            #[doc = stringify!($op)]
            #[doc = "` element-wise to the vector."]
            pub fn $op(&self) -> Self {
                $fn(&self.v).into()
            }
        }
    };
}

impl_unaryops_vector!(vln, ln);
impl_unaryops_vector!(vln1p, ln_1p);
impl_unaryops_vector!(vlog10, log10);
impl_unaryops_vector!(vlog2, log2);
impl_unaryops_vector!(vexp, exp);
impl_unaryops_vector!(vexp2, exp2);
impl_unaryops_vector!(vexpm1, exp_m1);
impl_unaryops_vector!(vsin, sin);
impl_unaryops_vector!(vcos, cos);
impl_unaryops_vector!(vtan, tan);
impl_unaryops_vector!(vsinh, sinh);
impl_unaryops_vector!(vcosh, cosh);
impl_unaryops_vector!(vtanh, tanh);
impl_unaryops_vector!(vasin, asin);
impl_unaryops_vector!(vacos, acos);
impl_unaryops_vector!(vatan, atan);
impl_unaryops_vector!(vasinh, asinh);
impl_unaryops_vector!(vacosh, acosh);
impl_unaryops_vector!(vatanh, atanh);
impl_unaryops_vector!(vsqrt, sqrt);
impl_unaryops_vector!(vcbrt, cbrt);
impl_unaryops_vector!(vabs, abs);
impl_unaryops_vector!(vfloor, floor);
impl_unaryops_vector!(vceil, ceil);
impl_unaryops_vector!(vtoradians, to_radians);
impl_unaryops_vector!(vtodegrees, to_degrees);
impl_unaryops_vector!(vrecip, recip);
impl_unaryops_vector!(vround, round);
impl_unaryops_vector!(vsignum, signum);

macro_rules! impl_unaryops_with_arg_vector {
    ($fn: ident, $op: ident, $argtype: ident) => {
        impl Vector {
            #[doc = "Apply the `f64` operation `"]
            #[doc = stringify!($op)]
            #[doc = "` element-wise to the vector."]
            pub fn $op(&self, arg: $argtype) -> Self {
                $fn(&self.v, arg).into()
            }
        }
    };
}

impl_unaryops_with_arg_vector!(vpowi, powi, i32);
impl_unaryops_with_arg_vector!(vpowf, powf, f64);

macro_rules! impl_inner_fn {
    ($output_type: ident for $($fn: ident),+) => {
        impl Vector {
            $(
                pub fn $fn(&self) -> $output_type {
                    $fn(&self.v)
                }
            )+
        }
    };
}

impl_inner_fn!(f64 for norm, max, mean, min, std, sum, prod, var, sample_std, sample_var, logsumexp, logmeanexp);

impl_inner_fn!(usize for argmin, argmax);
