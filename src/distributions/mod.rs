mod gamma;
mod normal;
mod uniform;

pub trait Distribution {
    fn sample(&self) -> f64;
    fn sample_iter(&self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample()).collect()
    }
}

pub trait Continuous: Distribution {
    fn pdf(&self, x: f64) -> f64;
}

pub use self::gamma::Gamma;
pub use self::normal::Normal;
pub use self::uniform::Uniform;
