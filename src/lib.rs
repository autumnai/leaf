#![feature(plugin)]
#![plugin(clippy)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![warn(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unsafe_code,
        unstable_features,
        unused_import_braces, unused_qualifications)]

#![feature(vec_resize)]

#[macro_use]
extern crate log;
extern crate rblas;
extern crate phloem;
mod shared_memory;
mod math;
mod layer;
mod layers;
mod solver;
mod network;

#[test]
fn it_works() {
}
