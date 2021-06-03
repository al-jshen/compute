use crate::linalg::Vector;

/// Vector-vector operations.
macro_rules! makefn_vops_binary {
    ($opname: ident, $op:tt) => {
        #[doc = "Implements a loop-unrolled version of the `"]
        #[doc = stringify!($op)]
        #[doc = "` function to be applied element-wise to two vectors."]
        pub fn $opname(v1: &[f64], v2: &[f64]) -> Vec<f64> {
            assert_eq!(v1.len(), v2.len());
            let n = v1.len();

            let mut v = vec![0.; n];
            let chunks = (n - (n % 8)) / 8;

            // unroll
            for i in 0..chunks {
                let idx = i * 8;
                assert!(n > idx + 7);
                v[idx] = v1[idx] $op v2[idx];
                v[idx + 1] = v1[idx + 1] $op v2[idx + 1];
                v[idx + 2] = v1[idx + 2] $op v2[idx + 2];
                v[idx + 3] = v1[idx + 3] $op v2[idx + 3];
                v[idx + 4] = v1[idx + 4] $op v2[idx + 4];
                v[idx + 5] = v1[idx + 5] $op v2[idx + 5];
                v[idx + 6] = v1[idx + 6] $op v2[idx + 6];
                v[idx + 7] = v1[idx + 7] $op v2[idx + 7];
            }

            // do the rest
            for j in (chunks * 8)..n {
                v[j] = v1[j] $op v2[j];
            }

            v
        }
    }
}

makefn_vops_binary!(vadd, +);
makefn_vops_binary!(vsub, -);
makefn_vops_binary!(vmul, *);
makefn_vops_binary!(vdiv, /);

/// Single vector operations.
macro_rules! makefn_vops_unary {
    ($opname: ident, $op:ident) => {
        #[doc = "Implements a loop-unrolled version of the `"]
        #[doc = stringify!($op)]
        #[doc = "` function to be applied element-wise to a vector."]
        pub fn $opname(v1: &[f64]) -> Vec<f64> {
            let n = v1.len();

            let mut v = vec![0.; n];
            let chunks = (n - (n % 8)) / 8;

            // unroll
            for i in 0..chunks {
                let idx = i * 8;
                assert!(n > idx + 7);
                v[idx] = v1[idx].$op();
                v[idx + 1] = v1[idx + 1].$op();
                v[idx + 2] = v1[idx + 2].$op();
                v[idx + 3] = v1[idx + 3].$op();
                v[idx + 4] = v1[idx + 4].$op();
                v[idx + 5] = v1[idx + 5].$op();
                v[idx + 6] = v1[idx + 6].$op();
                v[idx + 7] = v1[idx + 7].$op();
            }

            // do the rest
            for j in (chunks * 8)..n {
                v[j] = v1[j].$op();
            }

            v
        }
    };
}

makefn_vops_unary!(vln, ln);
makefn_vops_unary!(vln1p, ln_1p);
makefn_vops_unary!(vlog10, log10);
makefn_vops_unary!(vlog2, log2);
makefn_vops_unary!(vexp, exp);
makefn_vops_unary!(vexp2, exp2);
makefn_vops_unary!(vexpm1, exp_m1);
makefn_vops_unary!(vsin, sin);
makefn_vops_unary!(vcos, cos);
makefn_vops_unary!(vtan, tan);
makefn_vops_unary!(vsinh, sinh);
makefn_vops_unary!(vcosh, cosh);
makefn_vops_unary!(vtanh, tanh);
makefn_vops_unary!(vasin, asin);
makefn_vops_unary!(vacos, acos);
makefn_vops_unary!(vatan, atan);
makefn_vops_unary!(vasinh, asinh);
makefn_vops_unary!(vacosh, acosh);
makefn_vops_unary!(vatanh, atanh);
makefn_vops_unary!(vsqrt, sqrt);
makefn_vops_unary!(vcbrt, cbrt);
makefn_vops_unary!(vabs, abs);
makefn_vops_unary!(vfloor, floor);
makefn_vops_unary!(vceil, ceil);
makefn_vops_unary!(vtoradians, to_radians);
makefn_vops_unary!(vtodegrees, to_degrees);
makefn_vops_unary!(vrecip, recip);
makefn_vops_unary!(vround, round);
makefn_vops_unary!(vsignum, signum);

/// Single vector operations with a single argument.
macro_rules! makefn_vops_unary_with_arg {
    ($opname: ident, $op:ident, $argtype: ident) => {
        #[doc = "Implements a loop-unrolled version of the `"]
        #[doc = stringify!($op)]
        #[doc = "` function to be applied element-wise to a vector."]
        pub fn $opname(v1: &[f64], arg: $argtype) -> Vec<f64> {
            let n = v1.len();

            let mut v = vec![0.; n];
            let chunks = (n - (n % 8)) / 8;

            // unroll
            for i in 0..chunks {
                let idx = i * 8;
                assert!(n > idx + 7);
                v[idx] = v1[idx].$op(arg);
                v[idx + 1] = v1[idx + 1].$op(arg);
                v[idx + 2] = v1[idx + 2].$op(arg);
                v[idx + 3] = v1[idx + 3].$op(arg);
                v[idx + 4] = v1[idx + 4].$op(arg);
                v[idx + 5] = v1[idx + 5].$op(arg);
                v[idx + 6] = v1[idx + 6].$op(arg);
                v[idx + 7] = v1[idx + 7].$op(arg);
            }

            // do the rest
            for j in (chunks * 8)..n {
                v[j] = v1[j].$op(arg);
            }

            v
        }
    };
}

makefn_vops_unary_with_arg!(vpowi, powi, i32);
makefn_vops_unary_with_arg!(vpowf, powf, f64);

/// Vector-scalar operations.
macro_rules! makefn_vsops {
    ($opname: ident, $op:tt) => {
        #[doc = "Implements a loop-unrolled version of the `"]
        #[doc = stringify!($op)]
        #[doc = "` function to be applied element-wise between"]
        #[doc = "a vector and a scalar (in that order)."]
        pub fn $opname(v1: &[f64], scalar: f64) -> Vec<f64> {
            let n = v1.len();

            let mut v = vec![0.; n];
            let chunks = (n - (n % 8)) / 8;

            // unroll
            for i in 0..chunks {
                let idx = i * 8;
                assert!(n > idx + 7);
                v[idx] = v1[idx] $op scalar;
                v[idx + 1] = v1[idx + 1] $op scalar;
                v[idx + 2] = v1[idx + 2] $op scalar;
                v[idx + 3] = v1[idx + 3] $op scalar;
                v[idx + 4] = v1[idx + 4] $op scalar;
                v[idx + 5] = v1[idx + 5] $op scalar;
                v[idx + 6] = v1[idx + 6] $op scalar;
                v[idx + 7] = v1[idx + 7] $op scalar;
            }

            // do the rest
            for j in (chunks * 8)..n {
                v[j] = v1[j] $op scalar;
            }

            v
        }
    }
}

makefn_vsops!(vsadd, +);
makefn_vsops!(vssub, -);
makefn_vsops!(vsmul, *);
makefn_vsops!(vsdiv, /);

/// Scalar-vector operations.
macro_rules! makefn_svops {
    ($opname: ident, $op:tt) => {
        #[doc = "Implements a loop-unrolled version of the `"]
        #[doc = stringify!($op)]
        #[doc = "` function to be applied element-wise between"]
        #[doc = "a scalar and a vector (in that order)."]
        pub fn $opname(scalar: f64, v1: &[f64]) -> Vec<f64> {
            let n = v1.len();

            let mut v = vec![0.; n];
            let chunks = (n - (n % 8)) / 8;

            // unroll
            for i in 0..chunks {
                let idx = i * 8;
                assert!(n > idx + 7);
                v[idx] = scalar $op v1[idx];
                v[idx + 1] = scalar $op v1[idx + 1];
                v[idx + 2] = scalar $op v1[idx + 2];
                v[idx + 3] = scalar $op v1[idx + 3];
                v[idx + 4] = scalar $op v1[idx + 4];
                v[idx + 5] = scalar $op v1[idx + 5];
                v[idx + 6] = scalar $op v1[idx + 6];
                v[idx + 7] = scalar $op v1[idx + 7];
            }

            // do the rest
            for j in (chunks * 8)..n {
                v[j] = scalar $op v1[j];
            }

            v
        }
    }
}

makefn_svops!(svadd, +);
makefn_svops!(svsub, -);
makefn_svops!(svmul, *);
makefn_svops!(svdiv, /);
