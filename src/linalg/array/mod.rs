//! Vector and matrix structs and supporting functionality.

mod broadcast;
mod dot;
mod matrix;
mod vec;
mod vops;

pub use broadcast::*;
pub use dot::*;
pub use matrix::*;
pub use vec::*;
pub(crate) use vops::*;
