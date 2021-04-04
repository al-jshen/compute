/// Vector-vector operations.
#[macro_export]
macro_rules! impl_vops_binary {
    ($opname: ident, $op:tt) => {
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

impl_vops_binary!(vadd, +);
impl_vops_binary!(vsub, -);
impl_vops_binary!(vmul, *);
impl_vops_binary!(vdiv, /);

/// Single vector operations.
#[macro_export]
macro_rules! impl_vops_unary {
    ($opname: ident, $op:ident) => {
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

impl_vops_unary!(vln, ln);
impl_vops_unary!(vln1p, ln_1p);
impl_vops_unary!(vlog10, log10);
impl_vops_unary!(vlog2, log2);
impl_vops_unary!(vexp, exp);
impl_vops_unary!(vexp2, exp2);
impl_vops_unary!(vsin, sin);
impl_vops_unary!(vcos, cos);
impl_vops_unary!(vtan, tan);
impl_vops_unary!(vsinh, sinh);
impl_vops_unary!(vcosh, cosh);
impl_vops_unary!(vtanh, tanh);
impl_vops_unary!(vasin, asin);
impl_vops_unary!(vacos, acos);
impl_vops_unary!(vatan, atan);
impl_vops_unary!(vasinh, asinh);
impl_vops_unary!(vacosh, acosh);
impl_vops_unary!(vatanh, atanh);
impl_vops_unary!(vsqrt, sqrt);
impl_vops_unary!(vcbrt, cbrt);
impl_vops_unary!(vabs, abs);
impl_vops_unary!(vfloor, floor);
impl_vops_unary!(vceil, ceil);
impl_vops_unary!(vtoradians, to_radians);
impl_vops_unary!(vtodegrees, to_degrees);
impl_vops_unary!(vrecip, recip);

/// Vector-scalar operations.
#[macro_export]
macro_rules! impl_vsops {
        ($opname: ident, $op:tt) => {
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

impl_vsops!(vsadd, +);
impl_vsops!(vssub, -);
impl_vsops!(vsmul, *);
impl_vsops!(vsdiv, /);

/// Scalar-vector operations.
#[macro_export]
macro_rules! impl_svops {
    ($opname: ident, $op:tt) => {
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

impl_svops!(svadd, +);
impl_svops!(svsub, -);
impl_svops!(svmul, *);
impl_svops!(svdiv, /);
