# Contributing to Autumn

We love, that you are interested in contributing to Autumn. There are many ways
to contribute and we appreciate all of them. This document gives a rough
overview of how you can contribute to Autumn.

* [Pull Requests](#pull-requests)
* [Bug Reports](#bug-reports)
* [Feature Requests](#feature-requests)

If you have questions hop on the [Autumn Chat](https://gitter.im/autumnai/autumn)
, or reach out to {@[MJ](https://twitter.com/mjhirn), @[Max](https://twitter.com/hobofan)}.

## Pull Requests

#### Preparation

Before you get started, please find the page of the project you're looking to
improve. We encourage you to poke around in the code a little bit, familiarize
yourself with their development styles, check the commit log to see who is
contributing.

Before you start working, you might check out the **Network** tab on the project
to see all the other forks other people have made. Somebody might be already
working on the problem you would love to solve.

#### Making a PR

Pull requests are the primary mechanism we use to change Autumn repos. GitHub
itself has some [great documentation](https://help.github.com/articles/using-pull-requests/)
on using the Pull Request feature. We use the 'fork and pull' model described
there.

Please make pull requests against the `master` branch.

All pull requests are reviewed by another person.

> **Highfive not yet integrated**:
> *We have a bot, @rust-highfive, that will automatically assign a random*
> *person to review your request.*
>
> *If you want to request that a specific person reviews your pull request,*
> *you can add an `r?` to the message. For example, MJ usually reviews*
> *documentation changes. So if you were to make a documentation change, add*
>
>    r? @MichaelHirn
>
> *to the end of the message, and @rust-highfive will assign @MichaelHirn*
> *instead of a random person. This is entirely optional.*

After someone has reviewed your pull request, they will leave an annotation
on the pull request with an `r+`. It will look something like this:

   @homu: r+ 38fe8d2

This tells @homu, our lovable integration bot, that your pull request has
been approved. The PR then enters the
[merge queue][http://buildbot.rust-lang.org/homu/queue/rust], where
@homu will run all the tests on every platform we support. If it all works
out, @homu will merge your code into `master` and close the pull request.

## Bug Reports

While bugs are unfortunate, they're a reality in software. We can't fix what we
don't know about, so please report liberally. If you're not sure if something
is a bug or not, feel free to file a bug anyway.

If you have the chance, before reporting a bug, please search existing issues,
as it's possible that someone else has already reported your error. This doesn't
always work, and sometimes it's hard to know what to search for, so consider this
extra credit. We won't mind if you accidentally file a duplicate report.

[Opening an issue is easy](https://guides.github.com/features/issues/)
Here's a template that you can use to file a bug, though it's not necessary to
use it exactly:

    <short summary of the bug>

    I tried this code:

    <code sample that causes the bug>

    I expected to see this happen: <explanation>

    Instead, this happened: <explanation>

    ## Meta

    {Library, Rust, OS} versions

    Backtrace:

All three components are important: what you did, what you expected, what
happened instead. Please include information about what platform you're on, what
version of Rust and library you're using, etc.

Sometimes, a backtrace is helpful, and so including that is nice. To get
a backtrace, set the `RUST_BACKTRACE` environment variable. The easiest way
to do this is to invoke `rustc` like this:

```bash
$ RUST_BACKTRACE=1 rustc ...
```

## Feature Requests

To request a change to the way that one of the Autumn libraries work, please
open an issue in the repository.
