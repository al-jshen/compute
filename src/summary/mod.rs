//! A module for computing summary statistics of data.

mod covariance;
mod moments;
mod order;

pub use self::covariance::*;
pub use self::moments::*;
pub use self::order::*;
