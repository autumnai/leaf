# Contributing to Leaf

We love, that you are interested in contributing to Leaf. There are many ways
to contribute and we appreciate all of them. This document gives a rough
overview of how you can contribute to Leaf.

* [Pull Requests](#pull-requests)
* [Bug Reports](#bug-reports)
* [Feature Requests](#feature-requests)
* [Appendix](#appendix)
  * [Git Commit Guidelines](#git-commit-guidelines)
  * [Documentation Guidelines](#documentation-guidelines)


If you have questions hop on the [Leaf Chat](https://gitter.im/autumnai/leaf)
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

Pull requests are the primary mechanism we use to change Leaf repos. GitHub
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
[merge queue](http://buildbot.rust-lang.org/homu/queue/rust), where
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

To request a change to the way that one of the Leaf libraries work, please
open an issue in the repository.

## Appendix

### Git Commit Guidelines

We have very precise rules over how git commit messages should be formatted.
This leads to more readable messages that are easy to follow when looking
through the project history. But also, we may use the git commit messages to
auto-generate the Leaf change log.

#### Commit Message Format

Each commit message consists of a header, a body and a footer. The header has a
special format that includes a type, a scope and a subject:

    <type>/<scope>: <subject>
    \n
    <body>
    \n
    <footer>

Any line of the commit message cannot be longer 100 characters! This allows the
message to be easier to read on GitHub as well as in various git tools.

<**type**>:

Must be one of the following:

- *`feat`*: A new feature
- *`fix`*: A bug fix
- *`docs`*: Documentation only changes
- *`style`*: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- *`refactor`*: A code change that neither fixes a bug nor adds a feature
- *`perf`*: A code change that improves performance
- *`test`*: Adding missing tests
- *`chore`*: Changes to the build process or auxiliary tools and libraries such as documentation generation

<**scope**>:

The scope could be anything that specifies the place of the commit change.
For example: `feature1`, `tests`, `lib`, etc...

<**subject**>:

The subject contains succinct description of the change:
- use the imperative, present tense: "change" not "changed" nor "changes"
- don't capitalize first letter
- no dot (.) at the end

<**body**>:

The body should include the motivation for the change, contrast this with
previous behaviour and overall information about, why that commit matters.

- Just as in the `subject`, use the imperative, present tense

<**footer**>:

The footer should contain any information about Breaking Changes and is also the
place to reference GitHub issues that this commit closes. For Example:

    BREAKING CHANGE: [specify what is breaking]

    { REFERENCE, CLOSE, FIX } #Issue


#### Revert

If the commit reverts a previous commit, it should begin with `revert:`,
followed by the header of the reverted commit. In the body it should say:
`This reverts commit <hash>.`, where the hash is the SHA of the commit being
reverted.

### Documentation Guidelines

We created an extensive [Documentation Guide][1] for you, which outlines an easy
and efficient communication Framework for providing developers and users with
helpful Documentation about the Deep Learning Framework.

[1] https://medium.com/@autumn_eng/increasing-open-source-engagement-with-structural-communication-guidelines-for-code-documentation-e72533de8e45
