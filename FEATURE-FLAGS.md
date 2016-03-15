# Feature flags in Leaf

## The problem(s)

Supporting different backends is an important concept in Leaf.

Optimally we would like to always have to choice of running Leaf on all backends.
However in reality there are some tradeoffs that have to be made.

One problem is that certain backends require the presence of special hardware to
run (CUDA needs NVIDIA GPUs), or the libraries to address them are not present on
the developers machine which is necessary for compilation.

Another challenge is that not all backends have support for the same operations,
which constrains neural networks with special requirements to the backends that
provide those operations. Due to some limitations in the current version of Rust
(1.7) allowing differently featured backends can not be that easily supported.
See [Issue #81](https://github.com/autumnai/leaf/issues/81).

## The solution

Feature flags are a well known concept to add opt-in functionality that is
not necessary for every use-case of a library and are a good solution to the first
problem.
Luckily, Cargo, Rust's package manager has built-in support for feature flags.

A simple dependency with additional features enabled in a `Cargo.toml` looks like this:
```toml
[dependencies]
leaf = { version = "0.2.0", features = ["cuda"] }
```

Feature flags are usually used in an additive way, but **some configurations
of features for Leaf might actually take away some functionality**.
We do this because we want the models to be portable across different backends,
which is not possible if e.g. the CUDA backend supports Convolution layers while
the Native backend doesn't. To make it possible we deactivate those features that
are only available on a single backend, effectively "dumbing down" the backends.

Example:
- feature flags are `cuda` -> `Convolution` Layer **is available** since the CUDA backend provides the required traits and there is no native backend it has to be compatible with.
- feature flags are `native` -> `Convolution` Layer **is not available** since the native backend does not provide the required traits and there are no other frameworks present.
- feature flags are `native cuda` -> `Convolution` Layer **is not available** since the native backend does not provide the required traits, and the CUDA backend has been dumbed down.

## Using the feature flags

One thing we have ignored until now are default feature flags. Cargo allows to
define a set of features that should be included in a package by default .
One of the default feature flags of Leaf is the `native` flag. When looking at
the above example you might notice that the only way we can unleash the full
power of the CUDA backend is by deactivating the default `native` flag.
Cargo allows us to do that either via the `--no-default-features` on the CLI or
by specifying `default-feature = false` for a dependency in `Cargo.toml`.

#### In your project

The simple `Cargo.toml` example above works in simple cases but if you want
to provide the same flexibility of backends in your project, you can reexport
the feature flags.

A typical example (including collenchyma) would look like this:
```toml
[dependencies]
leaf = { version = "0.2.0", default-features = false }
# the native collenchyma feature is neccesary to read/write tensors
collenchyma = { version = "0.0.8", default-features = false, features = ["native"] }

[features]
default = ["native"]
native  = ["leaf/native"]
opencl  = ["leaf/opencl", "collenchyma/opencl"]
cuda    = ["leaf/cuda", "collenchyma/cuda"]

```

Building your project would then look like this:
```sh
# having both native and CUDA backends
# `native` is provided by default, and `cuda` explicitly specified by `--features cuda`
cargo build --features cuda
# unleashing CUDA
# `native` default not included because of `--no-default-features`, and `cuda` explicitly specified by `--features cuda`
cargo build --no-default-features --features cuda
```
