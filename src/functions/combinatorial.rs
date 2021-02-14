//! Combinatorial functions.

use crate::functions::gamma;
/// Calculates the [binomial coefficient](https://en.wikipedia.org/wiki/Binomial_coefficient)
/// nCk for two integers `n` and `k`, with `n >= k`.
///
pub fn binom_coeff(n: u64, k: u64) -> u64 {
    let mut nk = k;
    if k > n - k {
        nk = n - k;
    }

    let mut c = 1;
    for i in 1..=nk {
        if c / i > std::u64::MAX / nk {
            return 0;
        }
        c = c / i * (n - i + 1) + c % i * (n - i + 1) / i;
    }
    c
}

/// An alternative method for computing binomial coefficients. There is no significant difference
/// between the compute time using the `binom_coeff` method and this method. This method becomes
/// slightly inaccurate (by 1 or 2) starting at `n ~ 50`.
pub fn binom_coeff_alt(n: u64, k: u64) -> u64 {
    (gamma(n as f64 + 1.).ln() - gamma(k as f64 + 1.).ln() - gamma((n - k) as f64 + 1.).ln())
        .exp()
        .round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{DiscreteUniform, Distribution};

    #[test]
    fn test_binom_methods() {
        let n: Vec<u64> = DiscreteUniform::new(5, 45)
            .sample_vec(1000)
            .iter()
            .map(|x| *x as u64)
            .collect();
        let k: Vec<u64> = n
            .iter()
            .map(|x| DiscreteUniform::new(0, *x as i64).sample() as u64)
            .collect();
        for i in 0..1000 {
            assert_eq!(binom_coeff(n[i], k[i]), binom_coeff_alt(n[i], k[i]));
        }
    }

    #[test]
    fn test_binom_pascal() {
        let n: Vec<u64> = DiscreteUniform::new(5, 50)
            .sample_vec(1000)
            .iter()
            .map(|x| *x as u64)
            .collect();
        let k: Vec<u64> = n
            .iter()
            .map(|x| DiscreteUniform::new(0, (*x - 1) as i64).sample() as u64)
            .collect();
        for i in 0..1000 {
            assert_eq!(
                binom_coeff(n[i], k[i]) + binom_coeff(n[i], k[i] + 1),
                binom_coeff(n[i] + 1, k[i] + 1)
            );
        }
    }
}
