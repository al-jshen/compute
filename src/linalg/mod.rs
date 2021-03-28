//! Provides general linear algebra methods and matrix decompositions with a focus on low-dimensional data.

mod decomposition;
mod rotations;
mod utils;

pub use decomposition::*;
pub use rotations::*;
pub use utils::*;
