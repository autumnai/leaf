# Leaf • [![Join the chat at https://gitter.im/autumnai/leaf](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/leaf?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/leaf.svg?branch=master)](https://travis-ci.org/autumnai/leaf) [![Coverage Status](https://coveralls.io/repos/autumnai/leaf/badge.svg?branch=master&service=github)](https://coveralls.io/github/autumnai/leaf?branch=master) [![Crates.io](http://meritbadge.herokuapp.com/leaf)](https://crates.io/crates/leaf) [![License](https://img.shields.io/crates/l/leaf.svg)](LICENSE)

## Introduction

Leaf is the Hacker's Machine Intelligence Framework. It is built by software developers
and includes the very latest Machine Learning research, but it is not built by scientists.
It was designed to bring Machine Intelligence into production and onto any machine, from servers to embedded devices.

It is open source for anyone, licensed under the MIT and Apache-2 license. Leaf shares concepts from
TensorFlow, Torch and Caffe and was inspired by Rust and numerous research papers.

Leaf was started at [Autumn][autumn] to make the algorithms of researchers 100x more efficient
when implementing them in software. Leaf is written in [Rust][rust], a
language which is well suited for state-of-the-art machine learning. It allows
for the performance, memory-security, and extensibility, that other frameworks
(TensorFlow, Caffe, Theano) only gain by combining high-level languages (Python)
with low-level languages (C, C++).

__Top 3 Performance__<br/>
Leaf is a few months old, and thanks to its architecture and Rust Platform already one of the
fastest Machine Intelligence Framework in the world. See the performance Benchmarks for popular Deep Neural Networks.

__OS and Device Portable__<br/>
None of todays Machine Intelligence Frameworks runs anywhere. Not on any Operating System,
not on any GPU, FPGA as OpenCL support is lacking, not on all the available GPUs in the
machine as not all can run distributed and not on all machines possible, as many models are
to resource hungry for smartphones and IoT. Leaf brings portability to a whole new level, with [smart abstractions][collenchyma] and delivers
what TensorFlow set out to do.

We see Leaf as the core of constructing high-performance learning networks that
can be distributed and extended with other libraries e.g. for reinforcement
learning (Q-learning), visualizing and monitoring the learning of the network,
[automated preprocessing of non-numerical data][cuticula] or scale, deploy and
distribute your network to the cloud.

For more information see,

* [Leaf's Documentation][documentation],
* [Autumn's Website][autumn] or
* the [Q&A](#qa)

[caffe]: https://github.com/BVLC/caffe
[rust]: https://www.rust-lang.org/
[autumn]: http://autumnai.com
[tensorflow]: https://github.com/tensorflow/tensorflow
[benchmarks]: #benchmarks
[documentation]: http://autumnai.github.io/leaf

> Disclaimer: Leaf is currently in an early stage of development.
> If you are experiencing any bugs that are not due to not yet implemented
> features, feel free to create a issue.

## Getting Started

If you're using Cargo, just add Leaf to your Cargo.toml:

    [dependencies]
    leaf = "0.1.2"

If you're using [Cargo Edit][cargo-edit], you can
call:

    $ cargo add leaf


You can find examples at [Leaf Examples][leaf-examples].
Leaf Examples provides a CLI, so you can run popular Deep Learning examples with
Leaf right from the command line.

[cargo-edit]: https://github.com/killercup/cargo-edit

## Examples

We are providing a [Leaf Examples repository][leaf-examples], where we and
others publish executable machine learning examples build with Leaf. It features
a CLI for easy usage and has a detailed guide in the [project
README.md][leaf-examples].

Leaf itself comes with an examples directory, which features popular neural
networks (e.g. Alexnet, Overfeat, VGG). To run them on your machine, just follow
the install guide and then run

```bash
cargo run --release --example benchmark
```

[leaf-examples]: https://github.com/autumnai/leaf-examples

## Ecosystem / Extensions

We design Leaf and the other crates of the [Autumn Platform][autumn] as modular
and extensible as possible. More helpful crates you can use with Leaf:

- [**Cuticula**][cuticula]: Preprocessing Framework for Machine Learning
- [**Collenchyma**][collenchyma]: Portable, HPC-Framework on any hardware with CUDA, OpenCL, Rust

[cuticula]: https://github.com/autumnai/cuticula
[collenchyma]: https://github.com/autumnai/collenchyma

## Support / Contact

[...]

## Contributing

Want to contribute? Awesome! We have [instructions to help you get started][contributing].

Leaf has a near real-time collaboration culture and happens here on Github and
on the [Leaf Gitter Channel][gitter-leaf]. You can also reach out to the
Maintainers [@MJ][mj] and [@hobofan][hobofan] or engage at the
[#rust-machine-learning][irc] IRC on irc.mozilla.org.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as below, without any additional terms or
conditions.

[contributing]: CONTRIBUTING.md
[gitter-leaf]: https://gitter.im/autumnai/leaf
[mj]: https://twitter.com/mjhirn
[hobofan]: https://twitter.com/hobofan
[irc]: https://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust-machine-learning

## Changelog

You can find the release history in the root file [CHANGELOG.md][changelog].

We are using [Clog][clog], the Rust tool for auto-generating CHANGELOG files.

[changelog]: CHANGELOG.md
[Clog]: https://github.com/clog-tool/clog-cli

## Q&A

#### _Why Rust?_

The current hardware just recently became strong enough to support real-world
usage of machine intelligence e.g. super-human image recognition, self-driving
cars, etc.. For taking advantage of the computational power of the underlying
hardware from GPUs to clusters you need a low-level language that allows for
control of memory. But to make machine intelligence widely accessible you want
to have a high-level comfortable abstraction over the underlying hardware.

Rust allows us to cross this chasm.
Rust promises performance like C/C++ but with safe memory-control. For now we
can use C Rust wrappers for performant libraries. But in the future Rust
rewritten libraries will have the advantage of zero-cost safe memory control,
that will make large, parallel learning networks over CPUs and GPUs more
feasible and more reliable to develop. The development of these future libraries
is already under way e.g. [Glium][glium].

On the usability side, Rust offers a trait-system, that makes it easy for
researchers and hobbyists alike to extend and work with Leaf as if Leaf would
have been written in a higher-level language such as Ruby, Python, Java, etc.

#### _Who can use Leaf?_

We develop Leaf under the MIT open source license, which, paired with the easy
access and performance, makes Leaf a first-choice option for researchers and
developers alike.

#### _Why did you open source Leaf?_

We believe strongly in machine intelligence and think that it will have a major
impact on future innovations, products and our society. At Autumn, we experienced
a lack of common and well engineered tools for machine learning and therefore
started to create a modular toolbox for machine learning in Rust. We hope, that
with making our work open source, we will speed-up research and development of
production-ready applications and make their work easier as well.

#### _Who is Autumn?_

Autumn is a startup working on automated decision making. Autumn was started by
two developers MJ and Max. The startup is located in Berlin and recently
received a pre-seed investment from Axel Springer and Plug&Play.

[glium]: https://github.com/tomaka/glium

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
