# Leaf • [![Join the chat at https://gitter.im/autumnai/leaf](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/leaf?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/leaf.svg?branch=master)](https://travis-ci.org/autumnai/leaf) [![Coverage Status](https://coveralls.io/repos/autumnai/leaf/badge.svg?branch=master&service=github)](https://coveralls.io/github/autumnai/leaf?branch=master) [![Crates.io](http://meritbadge.herokuapp.com/leaf)](https://crates.io/crates/leaf) [![License](https://img.shields.io/crates/l/leaf.svg)](LICENSE)

*A modular, fast and open Deep Learning Framework for distributed
state of the art Deep Learning on both {C, G}PUs*

For more information see the [Documentation][documentation].

[documentation]: http://autumnai.github.io/leaf

## Getting Started

> Disclaimer: Leaf is currently in a very early and heavy stage of development.
> If you are experiencing any bugs that are not due to not yet implemented features,
> feel free to create a issue.

If you're using Cargo, just add Leaf to your Cargo.toml:

    [dependencies]
    leaf = "0.0.1"

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

Leaf has a mostly real-time collaboration culture and happens here on Github and
on the [Leaf Gitter Channels][gitter-leaf].
You can also reach out to the Maintainers
{[@MJ][mj], [@hobofan][hobofan]}.

[contributing]: CONTRIBUTING.md
[gitter-leaf]: https://gitter.im/autumnai/leaf
[mj]: https://twitter.com/mjhirn
[hobofan]: https://twitter.com/hobofan

## License

Leaf is released under the [MIT License][license].

[license]: LICENSE
