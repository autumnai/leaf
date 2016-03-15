# Leaf • [![Join the chat at https://gitter.im/autumnai/leaf](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/leaf?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/leaf.svg?branch=master)](https://travis-ci.org/autumnai/leaf) [![Crates.io](http://meritbadge.herokuapp.com/leaf)](https://crates.io/crates/leaf) [![License](https://img.shields.io/crates/l/leaf.svg)](LICENSE)

## Introduction

Leaf is a Machine Intelligence Framework engineered by software developers, not
scientists. It was inspired by the brilliant people behind TensorFlow, Torch,
Caffe, Rust and numerous research papers and brings modularity, performance and
portability to deep learning. Leaf is lean and tries to introduce minimal
technical debt to your stack.

Leaf is a few months old, but thanks to its architecture and Rust, it is already one of
the fastest Machine Intelligence Frameworks in the world.

<div align="center">
  <img src="http://autumnai.com/images/autumn_leaf_benchmarks_alexnet.png"><br><br>
</div>

> See more Deep Neural Networks benchmarks on [Deep Learning Benchmarks][deep-learning-benchmarks-website].

Leaf is portable. Run it on CPUs, GPUs, FPGAs on machines with an OS or on
machines without one. Run it with OpenCL or CUDA. Credit goes to
[Collenchyma][collenchyma] and Rust.

Leaf is part of the [Autumn][autumn] Machine Intelligence Platform, which is
working on making AI algorithms 100x more computational efficient. It seeks to bring
real-time, offline AI to smartphones and embedded devices.

We see Leaf as the core of constructing high-performance machine intelligence
applications. Leaf's design makes it easy to publish independent modules to make
e.g. deep reinforcement learning, visualization and monitoring, network
distribution, [automated preprocessing][cuticula] or scaleable production
deployment easily accessible for everyone.

For more info, refer to
* the [Leaf examples][leaf-examples],
* the [Leaf Documentation][documentation],
* the [Autumn Website][autumn] or
* the [Q&A](#qa)

[caffe]: https://github.com/BVLC/caffe
[rust]: https://www.rust-lang.org/
[autumn]: http://autumnai.com
[tensorflow]: https://github.com/tensorflow/tensorflow
[benchmarks]: #benchmarks
[leaf-examples]: #examples
[deep-learning-benchmarks-website]: http://autumnai.com/deep-learning-benchmarks
[documentation]: http://autumnai.github.io/leaf

> Disclaimer: Leaf is currently in an early stage of development.
> If you are experiencing any bugs with features that have been
> implemented, feel free to create a issue.

## Getting Started

If you are new to Rust you can install it as detailed [here][rust_download].
We also recommend taking a look at the [official Getting Started Guide][rust_getting_started].

If you're using Cargo, just add Leaf to your `Cargo.toml`:

```toml
[dependencies]
leaf = "0.2.0"
```

If you're using [Cargo Edit][cargo-edit], you can
call:

```bash
cargo add leaf
```
[rust_download]: https://www.rust-lang.org/downloads.html
[rust_getting_started]: https://doc.rust-lang.org/book/getting-started.html
[cargo-edit]: https://github.com/killercup/cargo-edit

If you are on a machine that doesn't have support for CUDA or OpenCL you
can selectively enable them like this in your `Cargo.toml`:

```toml
[dependencies]
leaf = { version = "0.2.0", default-features = false }

[features]
default = ["native"] # include only the ones you want to use, in this case "native"
native  = ["leaf/native"]
cuda    = ["leaf/cuda"]
opencl  = ["leaf/opencl"]
```

> More information on the use of feature flags in Leaf can be found in [FEATURE-FLAGS.md](./FEATURE-FLAGS.md)


## Examples

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

## Ecosystem / Extensions

We designed Leaf and the other crates of the [Autumn Platform][autumn] to be as modular
and extensible as possible. More helpful crates you can use with Leaf:

- [**Cuticula**][cuticula]: Preprocessing Framework for Machine Learning
- [**Collenchyma**][collenchyma]: Portable, HPC-Framework on any hardware with CUDA, OpenCL, Rust

[cuticula]: https://github.com/autumnai/cuticula
[collenchyma]: https://github.com/autumnai/collenchyma

## Support / Contact

- With a bit of luck, you can find us online on the #rust-machine-learing IRC at irc.mozilla.org,
- but we are always approachable on [Gitter/Leaf][gitter-leaf]
- For bugs and feature request, you can create a [Github issue][leaf-issue]
- For more private matters, send us email straight to our inbox: developers@autumnai.com
- Refer to [Autumn][autumn] for more information

[leaf-issue]: https://github.com/autumnai/leaf/issues

## Contributing

Want to contribute? Awesome! We have [instructions to help you get started][contributing].

Leaf has a near real-time collaboration culture, and it happens here on Github and
on the [Leaf Gitter Channel][gitter-leaf].

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as below, without any additional terms or
conditions.

[contributing]: CONTRIBUTING.md
[gitter-leaf]: https://gitter.im/autumnai/leaf
[mj]: https://twitter.com/mjhirn
[hobofan]: https://twitter.com/hobofan
[irc]: https://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust-machine-learning

## Changelog

You can find the release history at the [CHANGELOG.md][changelog]. We are using [Clog][clog], the Rust tool for auto-generating CHANGELOG files.

[changelog]: CHANGELOG.md
[Clog]: https://github.com/clog-tool/clog-cli

## Q&A

#### _Why Rust?_

Hardware has just recently become strong enough to support real-world
usage of machine intelligence e.g. super-human image recognition, self-driving
cars, etc. To take advantage of the computational power of the underlying
hardware, from GPUs to clusters, you need a low-level language that allows for
control of memory. But to make machine intelligence widely accessible you want
to have a high-level, comfortable abstraction over the underlying hardware.

Rust allows us to cross this chasm.
Rust promises performance like C/C++ but with safe memory-control. For now we
can use C Rust wrappers for performant libraries. But in the future Rust
rewritten libraries will have the advantage of zero-cost safe memory control,
that will make large, parallel learning networks over CPUs and GPUs more
feasible and more reliable to develop. The development of these future libraries
is already under way e.g. [Glium][glium].

On the usability side, Rust offers a trait-system that makes it easy for
researchers and hobbyists alike to extend and work with Leaf as if it were
written in a higher-level language such as Ruby, Python, or Java.

#### _Who can use Leaf?_

We develop Leaf under the MIT open source license, which, paired with the easy
access and performance, makes Leaf a first-choice option for researchers and
developers alike.

#### _Why did you open source Leaf?_

We believe strongly in machine intelligence and think that it will have a major
impact on future innovations, products and our society. At Autumn, we experienced
a lack of common and well engineered tools for machine learning and therefore
started to create a modular toolbox for machine learning in Rust. We hope that,
by making our work open source, we will speed up research and development of
production-ready applications and make that work easier as well.

#### _Who is Autumn?_

Autumn is a startup working on automated decision making. Autumn was started by
two developers, MJ and Max. The startup is located in Berlin and recently
received a pre-seed investment from Axel Springer and Plug&Play.

[glium]: https://github.com/tomaka/glium

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
