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

//! Leaf aims to become a open, fast and well-designed Framework for
//! distributed Deep Learning on {C, G}PUs.

#[macro_use]
extern crate log;
extern crate rblas;
extern crate phloem;
mod shared_memory;
mod math;
/// The Layer and Layer Interface
pub mod layer;
/// The specific Layers
pub mod layers;
/// The Solver
pub mod solver;
/// The specific Solvers
pub mod solvers;
/// The Network
pub mod network;
