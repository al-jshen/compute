use crate::distributions::*;
use fastrand::Rng;

pub struct Uniform {
    lower: f64,
    upper: f64,
    rng: Rng,
}

impl Uniform {
    pub fn new(lower: f64, upper: f64) -> Self {
        Uniform {
            lower,
            upper,
            rng: Rng::new(),
        }
    }
}

impl Distribution for Uniform {
    fn sample(&self) -> f64 {
        (self.upper - self.lower) * self.rng.f64() + self.lower
    }
}

impl Continuous for Uniform {
    fn pdf(&self, _: f64) -> f64 {
        1. / (self.upper - self.lower)
    }
}

#[test]
fn inrange() {
    let u = self::Uniform::new(-2., 6.);
    let samples = u.sample_iter(100);
    samples.into_iter().for_each(|x| {
        assert!(-2. <= x);
        assert!(x <= 6.);
    })
}
