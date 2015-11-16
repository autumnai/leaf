# Leaf • [![Join the chat at https://gitter.im/autumnai/leaf](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/leaf?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/leaf.svg?branch=master)](https://travis-ci.org/autumnai/leaf) [![Coverage Status](https://coveralls.io/repos/autumnai/leaf/badge.svg?branch=master&service=github)](https://coveralls.io/github/autumnai/leaf?branch=master) [![Crates.io](http://meritbadge.herokuapp.com/leaf)](https://crates.io/crates/leaf) [![License](https://img.shields.io/crates/l/leaf.svg)](LICENSE)

## Introduction

Leaf is a open source framework for machine intelligence, sharing concepts from
[TensorFlow][tensorflow] and [Caffe][caffe].

Leaf was started at [Autumn][autumn] to bridge the gap between research and
production of deep learning networks. Leaf is written in [Rust][rust], a
language which is well suited for state-of-the-art machine learning. It allows
for the performance, memory-security, and extensibility, that other frameworks
(TensorFlow, Caffe, Theano) only gain by combining high-level languages (Python)
with low-level languages (C, C++).

The architecture of Leaf's network is composed of layers, which represent
operations over n-dimensional numerical inputs into the network, known as Blobs.
This expressive and highly modular architecture allows you to deploy and
distribute your network over multiple device types, such as servers, desktops, mobile,
and use a variable number of CPUs or GPUs for computation.  
Layers usually implement mathematical operations, but can be used for many more
such as feeding in data, logging, or returning results. You can use the layers
that ship with Leaf (e.g. Convolutional, ReLU, RNN, SVM,
etc.) or thanks to Rust, easily extend Leaf with your own layers.

Leaf strives for leading-edge performance
([benchmarks are next][benchmarks-issue]), while providing a clear and
expressive architecture that creates - as we hope - an innovative and active
community around machine intelligence and fuels future research.

We see Leaf as the core of constructing high-performance learning networks that
can be distributed and extended with other libraries e.g. for reinforcement
learning (Q-learning), visualizing and monitoring the learning of the network,
[automated preprocessing of non-numerical data][cuticula] or scale, deploy and
distribute your network to the cloud.

For more information,

* see Leafs' [Documentation][documentation] or
* the [Q&A](#qa)

[caffe]: https://github.com/BVLC/caffe
[rust]: https://www.rust-lang.org/
[autumn]: http://autumnai.com
[tensorflow]: https://github.com/tensorflow/tensorflow
[benchmarks-issue]: https://github.com/autumnai/leaf/issues/26
[documentation]: http://autumnai.github.io/leaf

> Disclaimer: Leaf is currently in a very early and heavy stage of development.
> If you are experiencing any bugs that are not due to not yet implemented
> features, feel free to create a issue.

## Getting Started

If you're using Cargo, just add Leaf to your Cargo.toml:

    [dependencies]
    leaf = "0.1.0"

If you're using [Cargo Edit][cargo-edit], you can
call:

    $ cargo add leaf


You can find examples at [Leaf Examples][leaf-examples].
Leaf Examples provides a CLI, so you can run popular Deep Learning examples with
Leaf right from the command line.

[cargo-edit]: https://github.com/killercup/cargo-edit
[leaf-examples]: https://github.com/autumnai/leaf-examples

## Leaf Ecosystem and Extensions

We design Leaf and all other crates for machine learning completely modular and
as extensible as possible. More helpful crates you can use with Leaf:

- [**Cuticula**][cuticula]: Preprocessing Framework for Machine Learning
- [**Phloem**][phloem]: Universal CPU/GPU Data Blob for Machine Learning

[cuticula]: https://github.com/autumnai/cuticula
[phloem]: https://github.com/autumnai/phloem


## Contributing

Want to contribute? Awesome! We have
[instructions to help you get started contributing code or documentation][contributing].

Leaf has a near real-time collaboration culture and happens here on Github and
on the [Leaf Gitter Channels][gitter-leaf].
You can also reach out to the Maintainers
{[@MJ][mj], [@hobofan][hobofan]}.

[contributing]: CONTRIBUTING.md
[gitter-leaf]: https://gitter.im/autumnai/leaf
[mj]: https://twitter.com/mjhirn
[hobofan]: https://twitter.com/hobofan

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

Leaf is released under the [MIT License][license].

[license]: LICENSE
