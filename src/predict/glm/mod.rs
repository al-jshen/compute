//! Implementation of the [generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model).
//! Includes [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression),
//! [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression), and other models.
//!
//! # Resources
//! This code is a translation of [py-glm](https://github.com/madrury/py-glm).

mod families;
// mod formula;
mod glm;

pub use families::*;
// pub use formula::*;
pub use glm::*;
