use crate::distributions::*;
use crate::functions::binom_coeff;

/// Implements the [Binomial](https://en.wikipedia.org/wiki/https://en.wikipedia.org/wiki/Binomial_distribution)
/// distribution with trials `n` and probability of success `p`.
#[derive(Debug, Clone, Copy)]
pub struct Binomial {
    n: u64,
    p: f64,
}

impl Binomial {
    /// Create a new Binomial distribution with parameters `n` and `p`.
    ///
    /// # Remarks
    /// `n` must be a non-negative integer, and `p` must be in [0, 1].
    pub fn new(n: u64, p: f64) -> Self {
        if !(0. ..=1.).contains(&p) {
            panic!("`p` must be in [0, 1]");
        }
        Binomial { n, p }
    }
    pub fn set_n(&mut self, n: u64) -> &mut Self {
        self.n = n;
        self
    }
    pub fn set_p(&mut self, p: f64) -> &mut Self {
        if !(0. ..=1.).contains(&p) {
            panic!("`p` must be in [0, 1]");
        }
        self.p = p;
        self
    }
}

impl Default for Binomial {
    fn default() -> Self {
        Self::new(1, 0.5)
    }
}

impl Distribution for Binomial {
    type Output = f64;
    /// Samples from the given Binomial distribution. For `np <= 30`, this is done with an inversion algorithm.
    /// Otherwise, this is done with the BTPE algorithm from Kachitvichyanukul and Schmeiser 1988.
    fn sample(&self) -> f64 {
        if self.n == 0 || self.p == 0. {
            return 0.;
        } else if (self.p - 1.).abs() <= f64::EPSILON {
            return self.n as f64;
        }

        let p = if self.p <= 0.5 { self.p } else { 1. - self.p };

        let res = if p * self.n as f64 <= 30. {
            binomial_inversion(self.n, p)
        } else {
            binomial_btpe(self.n, p)
        };

        res as f64
    }
}

impl Distribution1D for Binomial {
    fn update(&mut self, params: &[f64]) {
        self.set_n(params[0] as u64);
        self.set_p(params[1]);
    }
}

pub fn binomial_inversion(n: u64, p: f64) -> u64 {
    let s = p / (1. - p);
    let a = ((n + 1) as f64) * s;
    let mut r = (1. - p).powi(n as i32);
    let mut u = fastrand::f64();
    let mut x: u64 = 0;
    while u > r as f64 {
        u -= r;
        x += 1;
        r *= a / (x as f64) - s;
    }
    x
}

pub fn binomial_btpe(n: u64, p: f64) -> u64 {
    // step 0
    let nf = n as f64;
    let r = if p <= 0.5 { p } else { 1. - p };
    let q = 1. - r;
    let nrq = nf * r * q;
    let fm = nf * r + r;
    let m = fm.floor();
    let p1 = (2.195 * nrq.sqrt() - 4.6 * q).floor() + 0.5;
    let xm = m + 0.5;
    let xl = xm - p1;
    let xr = xm + p1;
    let lambda = |x: f64| x * (1. + x / 2.);
    let c = 0.134 + 20.5 / (15.3 + m);
    let ll = lambda((fm - xl) / (fm - xl * r));
    let lr = lambda((xr - fm) / (xr * q));
    let p2 = p1 * (1. + 2. * c);
    let p3 = p2 + c / ll;
    let p4 = p3 + c / lr;
    let mut y: f64;

    let ugen = Uniform::new(0., p4);
    let vgen = Uniform::new(0., 1.);

    loop {
        // step 1
        let u = ugen.sample();
        let mut v = vgen.sample();

        // clippy says dont do this
        // if !(u > p1) {

        // clippy suggests this, then says dont do this...
        // let u_p1_cmp = match u.partial_cmp(&p1) {
        //     None | Some(std::cmp::Ordering::Equal) | Some(std::cmp::Ordering::Less) => true,
        //     _ => false,
        // };
        //
        // if u_p1_cmp {

        if matches!(
            u.partial_cmp(&p1),
            None | Some(std::cmp::Ordering::Equal) | Some(std::cmp::Ordering::Less)
        ) {
            y = (xm - p1 * v + u).floor();
            // go to step 6
            break;
        }

        if matches!(
            u.partial_cmp(&p2),
            None | Some(std::cmp::Ordering::Equal) | Some(std::cmp::Ordering::Less)
        ) {
            // step 2
            let x = xl + (u - p1) / c;
            v = v * c + 1. - (m - x + 0.5).abs() / p1;
            if v > 1. {
                // go to step 1
                continue;
            } else {
                y = x.floor();
                // go to step 5
            }
        } else if matches!(
            u.partial_cmp(&p3),
            None | Some(std::cmp::Ordering::Equal) | Some(std::cmp::Ordering::Less)
        ) {
            // step 3
            y = (xl + v.ln() / ll).floor();
            if y < 0. {
                // go to step 1
                continue;
            } else {
                v *= (u - p2) * ll;
                // go to step 5
            }
        } else {
            // step 4
            y = (xr - v.ln() / lr).floor();
            if y > nf {
                // go to step 1
                continue;
            } else {
                v *= (u - p3) * lr;
                // go to step 5
            }
        }

        // step 5.0
        let k = (y - m).abs();
        if !(k > 20. && k < 0.5 * (nrq) - 1.) {
            // step 5.1
            let s = p / q;
            let a = s * (n as f64 + 1.);
            let mut f = 1.;

            if m < y {
                let mut i = m;
                loop {
                    i += 1.;
                    f *= (a / i) - s;
                    if (i - y).abs() < f64::EPSILON {
                        break;
                    }
                }
            } else if m > y {
                let mut i = y;
                loop {
                    i += 1.;
                    f /= (a / i) - s;
                    if (i - m).abs() < f64::EPSILON {
                        break;
                    }
                }
            }
            if v > f {
                // go to step 1
                continue;
            } else {
                // go to step 6
                break;
            }
        }

        // step 5.2
        let rho = (k / nrq) * ((k * (k / 3. + 0.625) + 1. / 6.) / nrq + 0.5);
        let t = -k * k / (2. * nrq);
        let biga = v.ln();
        if biga < t - rho {
            // go to step 6
            break;
        }
        if biga > t + rho {
            // go to step 1
            continue;
        }

        // step 5.3
        let x1 = y + 1.;
        let f1 = m + 1.;
        let z = nf + 1. - m;
        let w = nf - y + 1.;

        let st = |x: f64| {
            (13860. - (462. - (132. - (99. - 140. / (x * x)) / (x * x)) / (x * x)) / (x * x))
                / x
                / 166320.
        };

        if biga
            > xm * (f1 / x1).ln()
                + (nf - m + 0.5) * (z / w).ln()
                + (y - m) * (w * r / (x1 * q)).ln()
                + st(f1)
                + st(z)
                + st(x1)
                + st(w)
        {
            // go to step 1
            continue;
        }
        // go to step 6
        break;
    }

    // step 6
    if p > 0.5 {
        y = nf - y;
    }

    y as u64
}

impl Discrete for Binomial {
    /// Calculates the [probability mass
    /// function](https://en.wikipedia.org/wiki/Probability_mass_function) for the given Binomial
    /// distribution at `k`.
    ///
    fn pmf(&self, k: i64) -> f64 {
        binom_coeff(self.n, k as u64) as f64
            * self.p.powi(k as i32)
            * (1. - self.p).powi((self.n - k as u64) as i32)
    }
}

impl Mean for Binomial {
    /// Calculates the mean, which is given by `np`.
    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }
}

impl Variance for Binomial {
    /// Calculates the variance, which is given by `npq`, where `q = 1-p`
    fn var(&self) -> f64 {
        self.n as f64 * self.p * (1. - self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::{mean, var};
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_moments() {
        let distr1 = Binomial::new(15, 0.3);
        let data1 = distr1.sample_vec(1e6 as usize);
        let mean1 = mean(&data1);
        let var1 = var(&data1);
        assert_approx_eq!(mean1, 4.5, 1e-2);
        assert_approx_eq!(var1, 3.15, 1e-2);

        let distr2 = Binomial::new(70, 0.5);
        let data2 = distr2.sample_vec(1e6 as usize);
        let mean2 = mean(&data2);
        let var2 = var(&data2);
        assert_approx_eq!(mean2, 35., 1e-2);
        assert_approx_eq!(var2, 17.5, 1e-2);
    }
}
