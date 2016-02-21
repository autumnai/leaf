//! Leaf is a open, modular and clear-designed Machine Intelligence Framework providing
//! state-of-the-art performance for distributed (Deep|Machine) Learning - sharing concepts from
//! Tensorflow and Caffe.
//!
//! An important module in Leaf is the backend-agnostic, high-performance computation Framework
//! [Collenchyma][collenchyma], which combines performance and usability for Leaf Networks.
//! This allows you to run and deploy Leaf Networks to servers, desktops or even mobiles
//! using the full available computation power of GPUs or other CUDA/OpenCL supported
//! devices for the learning of your Networks. And if your machine does not have a GPU or you do
//! not want to install CUDA/OpenCL on your local machine, Leaf will gracefully fall back to
//! your native host CPU.
//!
//! ## Architecture
//!
//! Leaf's [Network][network] is a compositional model, representing a collection of connected
//! [layers][layers], making operations over numerical data.
//!
//! The Network defines the entire model, by defining the hirarchical structure of layers from
//! bottom to top. At execution time, the Network passes the data, flowing through the Network,
//! from one layer to the next. The output of one layer is the input for the layer on top. On a
//! backward pass, the Network passes the derivatives inverted through the Network.
//!
//! Layers, the building block of a Leaf Network, are small units, describing computation over
//! numerical input data. Generally speaking Layers take input and produce an output, but
//! essentially a Layer can describe any functionality e.g. logging as long as it obeys to the
//! general behaviour specifications of a Layer. Any Layer can be grouped in one of four
//! Layer types which are closer defined at the [Layers page][layers]. Every
//! layer serves a special purpose and can occur zero, one or many times inside a Network.
//!
//! Leaf uses a Blob, provided by the [Phloem][phloem] module, an N-dimensional array
//! for a unified memory interface over the actual data for automatic synchronization between
//! different devices (CUDA, OpenCL, host CPU). A Blob stores the actual data as well as the
//! derivatives and is used for the data flowing through the system and for the state
//! representation of Layers, which is important for portability and performance.
//! A Blob can be swapped from backend to backend and can be used for computations on CUDA, OpenCL
//! and native host CPU. It provides performance optimizations and automatically takes care of
//! memory management and synchronization.
//!
//! The learning and optimization of the Network happens at the [Solver][solver] and is decoupled
//! from the Network making the setup clean and flexible. One of the four layer types is a Loss
//! Layer, which is used for the interaction of Network and Solver. The Network produces the loss
//! and gradients, which the Solver uses to optimize the Network through parameter updates. Beside
//! that, the Solver provides housekeeping and other evaluations of the Network. All operation
//! on the Solver happen through Collenchyma, therefore can be executed on Cuda, OpenCL or native
//! host CPU as well.
//!
//! Leaf provides a robust and modular design, which allows to express almost any numerical
//! computation including SVMs, RNNs and other popular learning algorithms. We hope that Leaf can
//! help future research and production development alike as it combines expressiveness,
//! performance and usability.
//!
//! [network]: ./network/index.html
//! [layers]: ./layers/index.html
//! [phloem]: https://github.com/autumnai/phloem
//! [solver]: ./solvers/index.html
//!
//! ## Philosophy
//!
//! We are strong believers in the technology of Machine Learning.
//!
//! We put our experience in software engineering into Leaf, to solve our own
//! need for a modern, performant and easy-to-use Deep Learning Framework.
//! These principles direct our decisions on Leaf and related projects.
//!
//! * __Cutting Edge Performance__:</br>
//! For research and industry speed and efficency are curcial for
//! state-of-the-art machine learning over massive data and networks.
//! * __Open and Expressive Architecture__:</br>
//! Designing an open architecture that follows best practices and concepts in
//! Engineering such as modularity, flexibility and expressiveness is critical
//! to stimulate future innovation.
//! * __Clear and Transparent Documentation__:</br>
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
//! [collenchyma]: https://github.com/autumnai/collenchyma
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
#![cfg_attr(lint, feature(plugin))]
#![cfg_attr(lint, plugin(clippy))]
#![feature(augmented_assignments)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unsafe_code,
        unused_import_braces, unused_qualifications)]

// used when run with  cargo test --no-run --features clippy
// or cargo build --features clippy
#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]
#![cfg_attr(feature="clippy", deny(clippy, clippy_pedantic))]

#[macro_use]
extern crate log;
extern crate phloem;
extern crate collenchyma as co;
pub mod shared_memory;
pub mod layer;
pub mod layers;
pub mod solver;
pub mod solvers;
pub mod network;
