pub mod normal;

pub trait Distribution {
    fn sample(&self) -> f64;
    fn sample_iter(&self, n: usize) -> Vec<f64>;
}

pub trait Continuous: Distribution {
    fn pdf(&self, x: f64) -> f64;
}
