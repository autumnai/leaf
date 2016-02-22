//! Provides various helpful layers, which might be not directly related to
//! neural networks in general.
//!
//! These layers do not have to necesarrely manipulate the data flowing through
//! them and might have
//! no effect on the Networks' capabilities to learn (e.g. loging) but obey all
//! the rules of a [Layer][1].
//! The type of these layers can vary a lot. From data normalization to
//! specific data access layers for e.g. a database like LevelDB.
//!
//! [1]: ../../layer/index.html
pub use self::flatten::Flatten;
pub use self::reshape::{Reshape, ReshapeConfig};

pub mod flatten;
pub mod reshape;
