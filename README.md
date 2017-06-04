# Leaf • [![Join the chat at https://gitter.im/autumnai/leaf](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/leaf?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/leaf.svg?branch=master)](https://travis-ci.org/autumnai/leaf) [![Crates.io](http://meritbadge.herokuapp.com/leaf)](https://crates.io/crates/leaf) [![License](https://img.shields.io/crates/l/leaf.svg)](LICENSE)

## This project is no longer being maintained
https://medium.com/@mjhirn/tensorflow-wins-89b78b29aafb


## Introduction

Leaf is a open Machine Learning Framework for hackers to build classical, deep
or hybrid machine learning applications. It was inspired by the brilliant people
behind TensorFlow, Torch, Caffe, Rust and numerous research papers and brings
modularity, performance and portability to deep learning.

Leaf has one of the simplest APIs, is lean and tries to introduce minimal
technical debt to your stack.

See the [Leaf - Machine Learning for Hackers][leaf-book] book for more.

Leaf is a few months old, but thanks to its architecture and Rust, it is already
one of the fastest Machine Intelligence Frameworks available.

<div align="center">
  <img src="http://autumnai.com/images/autumn_leaf_benchmarks_alexnet.png"><br><br>
</div>

> See more Deep Neural Networks benchmarks on [Deep Learning Benchmarks][deep-learning-benchmarks-website].

Leaf is portable. Run it on CPUs, GPUs, and FPGAs, on machines with an OS, or on
machines without one. Run it with OpenCL or CUDA. Credit goes to
[Collenchyma][collenchyma] and Rust.

Leaf is part of the [Autumn][autumn] Machine Intelligence Platform, which is
working on making AI algorithms 100x more computational efficient.

We see Leaf as the core of constructing high-performance machine intelligence
applications. Leaf's design makes it easy to publish independent modules to make
e.g. deep reinforcement learning, visualization and monitoring, network
distribution, [automated preprocessing][cuticula] or scaleable production
deployment easily accessible for everyone.

[caffe]: https://github.com/BVLC/caffe
[rust]: https://www.rust-lang.org/
[autumn]: http://autumnai.com
[leaf-book]: http://autumnai.com/leaf/book
[tensorflow]: https://github.com/tensorflow/tensorflow
[benchmarks]: #benchmarks
[leaf-examples]: #examples
[deep-learning-benchmarks-website]: http://autumnai.com/deep-learning-benchmarks
[documentation]: http://autumnai.github.io/leaf

> Disclaimer: Leaf is currently in an early stage of development.
> If you are experiencing any bugs with features that have been
> implemented, feel free to create a issue.

## Getting Started

### Documentation

To learn how to build classical, deep or hybrid machine learning applications with Leaf, check out the [Leaf - Machine Learning for Hackers][leaf-book] book.

For additional information see the [Rust API Documentation][documentation] or the [Autumn Website][autumn].

Or start by running the **Leaf examples**.

We are providing a [Leaf examples repository][leaf-examples], where we and
others publish executable machine learning models build with Leaf. It features
a CLI for easy usage and has a detailed guide in the [project
README.md][leaf-examples].

Leaf comes with an examples directory as well, which features popular neural
networks (e.g. Alexnet, Overfeat, VGG). To run them on your machine, just follow
the install guide, clone this repoistory and then run

```bash
# The examples currently require CUDA support.
cargo run --release --no-default-features --features cuda --example benchmarks alexnet
```

[leaf-examples]: https://github.com/autumnai/leaf-examples

### Installation

> Leaf is build in [Rust][rust]. If you are new to Rust you can install Rust as detailed [here][rust_download].
We also recommend taking a look at the [official Rust - Getting Started Guide][rust_getting_started].

To start building a machine learning application (Rust only for now. Wrappers are welcome) and you are using Cargo, just add Leaf to your `Cargo.toml`:

```toml
[dependencies]
leaf = "0.2.1"
```

[rust_download]: https://www.rust-lang.org/downloads.html
[rust_getting_started]: https://doc.rust-lang.org/book/getting-started.html
[cargo-edit]: https://github.com/killercup/cargo-edit

If you are on a machine that doesn't have support for CUDA or OpenCL you
can selectively enable them like this in your `Cargo.toml`:

```toml
[dependencies]
leaf = { version = "0.2.1", default-features = false }

[features]
default = ["native"] # include only the ones you want to use, in this case "native"
native  = ["leaf/native"]
cuda    = ["leaf/cuda"]
opencl  = ["leaf/opencl"]
```

> More information on the use of feature flags in Leaf can be found in [FEATURE-FLAGS.md](./FEATURE-FLAGS.md)

### Contributing

If you want to start hacking on Leaf (e.g.
  [adding a new `Layer`](http://autumnai.com/leaf/book/create-new-layer.html))
you should start with forking and cloning the repository.

We have more instructions to help you get started in the [CONTRIBUTING.md][contributing].

We also has a near real-time collaboration culture, which happens
here on Github and on the [Leaf Gitter Channel][gitter-leaf].

> Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as below, without any additional terms or conditions.

[contributing]: CONTRIBUTING.md
[gitter-leaf]: https://gitter.im/autumnai/leaf
[mj]: https://twitter.com/mjhirn
[hobofan]: https://twitter.com/hobofan
[irc]: https://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust-machine-learning

## Ecosystem / Extensions

We designed Leaf and the other crates of the [Autumn Platform][autumn] to be as modular
and extensible as possible. More helpful crates you can use with Leaf:

- [**Cuticula**][cuticula]: Preprocessing Framework for Machine Learning
- [**Collenchyma**][collenchyma]: Portable, HPC-Framework on any hardware with CUDA, OpenCL, Rust

[cuticula]: https://github.com/autumnai/cuticula
[collenchyma]: https://github.com/autumnai/collenchyma

## Support / Contact

- With a bit of luck, you can find us online on the #rust-machine-learning IRC at irc.mozilla.org,
- but we are always approachable on [Gitter/Leaf][gitter-leaf]
- For bugs and feature request, you can create a [Github issue][leaf-issue]
- For more private matters, send us email straight to our inbox: developers@autumnai.com
- Refer to [Autumn][autumn] for more information

[leaf-issue]: https://github.com/autumnai/leaf/issues

## Changelog

You can find the release history at the [CHANGELOG.md][changelog]. We are using [Clog][clog], the Rust tool for auto-generating CHANGELOG files.

[changelog]: CHANGELOG.md
[Clog]: https://github.com/clog-tool/clog-cli

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
