//! A module for computing statistics of data.

mod covariance;
mod moments;
mod order;
mod tests;

pub use self::covariance::*;
pub use self::moments::*;
pub use self::order::*;
pub use self::tests::*;
