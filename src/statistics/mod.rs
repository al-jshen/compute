//! A module for computing statistics of data.

mod covariance;
mod hist;
mod moments;
mod order;
// mod tests;

pub use self::covariance::*;
pub use self::hist::*;
pub use self::moments::*;
pub use self::order::*;
// pub use self::tests::*;
