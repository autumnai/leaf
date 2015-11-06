//! **Leaf is a open, fast and a well-designed, modular Framework for distributed Deep Learning on
//! {C, G}PUs.**
//!
//! ## Overview
//!
//! To build a Deep Neural Network you create a [Network][network] which is a container for all
//! different types of [Layers][layers]. These layers are grouped in different types such as
//! [Activation Layers][activation], [Loss Layers][loss] which states the characteristics of the
//! layer.
//!
//! Now to train your network you will use one of the [Solvers][solvers]. The Solver defines the
//! [Optimization Method][optimization] and keeps track on the learning progress.
//!
//! The operations can run on different Backends {CPU, GPU} and must not be defined at compile
//! time, which allows for easy backend swapping.
//!
//! ## Examples
//!
//! ```
//! # extern crate leaf;
//! # use leaf::network::{NetworkConfig};
//! # fn main() {
//! # }
//! ```
//!
//! ## Development
//!
//! The implementation of various Layers is pretty scarce at the moment.  
//! There are around a dozen layers, which are really important and would increase the value and
//! functionality of Leaf tremendously.  
//! Progress get tracked at
//! - [Issue #18 for Loss Layers][issue-loss]
//! - [Issue #19 for Activation Layers][issue-activation]
//! - [Issue #20 for Common Layers][issue-common]
//!
//! To better structure these different layer types and state their functionality and behavior more
//! clearly, we will give them their own mod. [Issue #21][issue-mod]
//!
//! [network]: ./network/index.html
//! [layers]: ./layers/index.html
//! [activation]: #
//! [loss]: #
//! [solvers]: ./solvers/index.html
//! [optimization]: https://en.wikipedia.org/wiki/Stochastic_optimization
//!
//! [issue-loss]: https://github.com/autumnai/leaf/issues/18
//! [issue-activation]: https://github.com/autumnai/leaf/issues/19
//! [issue-common]: https://github.com/autumnai/leaf/issues/20
//! [issue-mod]: https://github.com/autumnai/leaf/issues/21
#![feature(plugin)]
#![feature(augmented_assignments)]
#![plugin(clippy)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unsafe_code,
        unused_import_braces, unused_qualifications)]

#[macro_use]
extern crate log;
extern crate rblas;
extern crate phloem;
pub mod shared_memory;
mod math;
/// The Layer and Layer Interface
pub mod layer;
pub mod layers;
/// The Solver
pub mod solver;
/// The specific Solvers
pub mod solvers;
pub mod network;
