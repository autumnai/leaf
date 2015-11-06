//! Leaf is a open, fast and a well-designed, modular Framework for distributed
//! Deep Learning on {C, G}PUs.
//!
//! ## Overview
//!
//! To build a Deep Neural Network you create a [Network][network] which is a
//! container for all different types of [Layers][layers].
//! These layers are grouped in different types such as
//! [Activation Layers][activation] and [Loss Layers][loss] which states the
//! characteristics of the layer.
//!
//! Now to train your network you will use one of the [Solvers][solvers].
//! The Solver defines the [Optimization Method][optimization]
//! and keeps track on the learning progress.
//!
//! The operations can run on different Backends {CPU, GPU} and must not be
//! defined at compile time, which allows for easy backend swapping.
//!
//! ## Philosophy
//!
//! We are strong believers in the technology of Machine Learning.
//! We put our experience in
//! software engineering into Leaf, to solve our own need for a modern,
//! performant and easy-to-use Deep Learning Framework.
//! These principles direct our decisions on Leaf and related projects.
//!
//! * __Performance__:</br>
//! For research and industry speed and efficency are curcial for
//! state-of-the-art machine learning over massive data and networks.
//! * __Architecture__:</br>
//! Designing an open architecture that follows best practices and concepts in
//! Engineering such as modularity, flexibility and expressiveness is critical
//! to stimulate future innovation.
//! * __Documentation__:</br>
//! A well-written documentation that addresses both concepts and
//! implementations, empowers developers and researchers to contribute their
//! unique experience to the project for the benefit of everyone.
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
//! The implementation of various Layers is pretty scarce at the moment.<br/>
//! There are around a dozen layers, which are really important and would
//! increase the value and functionality of Leaf tremendously.<br/>
//! Progress is tracked at<br/>
//!
//! - [Issue #18 for Loss Layers][issue-loss]
//! - [Issue #19 for Activation Layers][issue-activation]
//! - [Issue #20 for Common Layers][issue-common]
//!
//! [network]: ./network/index.html
//! [layers]: ./layers/index.html
//! [activation]: ./layers/activation/index.html
//! [loss]: ./layers/loss/index.html
//! [solvers]: ./solvers/index.html
//! [optimization]: https://en.wikipedia.org/wiki/Stochastic_optimization
//!
//! [issue-loss]: https://github.com/autumnai/leaf/issues/18
//! [issue-activation]: https://github.com/autumnai/leaf/issues/19
//! [issue-common]: https://github.com/autumnai/leaf/issues/20
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
