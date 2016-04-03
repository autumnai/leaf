# Leaf - Machine Learning for Hackers

> Our life is frittered away by detail. Simplify, simplify. -
> _Henry David Thoreau_

This short book teaches you how you can build Machine Learning applications (with
[Leaf][leaf]).

Leaf is a Machine Intelligence Framework engineered by hackers, not scientists.
It has a very simple API, [Layers][layers] and [Solvers][solvers], with which
you can build classical Machine as well as Deep Learning and other fancy Machine
Intelligence applications. Although Leaf is just a few months old, it is
already, thanks to Rust and Collenchyma, one of the fastest Machine Intelligence
Frameworks available.

Leaf was inspired by the brilliant people behind TensorFlow, Torch, Caffe,
Rust and numerous research papers and brings modularity, performance and
portability to Deep Learning.

<br/>

<div align="center">
  <iframe src="https://ghbtns.com/github-btn.html?user=autumnai&repo=leaf&type=star&count=true" frameborder="0" scrolling="0" width="120px" height="20px"></iframe>
  <a href="https://twitter.com/autumn_eng" class="twitter-follow-button" data-show-count="false">Follow @autumn_eng</a>
  <script>!function(d,s,id){var js,fjs=d.getElementsByTagName(s)[0],p=/^http:/.test(d.location)?'http':'https';if(!d.getElementById(id)){js=d.createElement(s);js.id=id;js.src=p+'://platform.twitter.com/widgets.js';fjs.parentNode.insertBefore(js,fjs);}}(document, 'script', 'twitter-wjs');</script>
</div>

<br/>

> To make the most of the book, a basic understanding of the fundamental concepts
> of Machine and Deep Learning is recommended. Good resources to get you from
> zero to almost-ready-to-build-Machine-Learning-applications:
>
> * [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) or
> * [Stanford Course on (Convolutional) Neural Networks](http://cs231n.github.io/)
>
> And in case some knowledge already exist, [A 'brief' history of Deep Learning](http://www.andreykurenkov.com/writing/a-brief-history-of-neural-nets-and-deep-learning/) or [The Glossary](./deep-learning-glossary.html)
> might prove informative.

But Machine- and Deep Learning are really easy with Leaf.

Chain [layers](./deep-learning-glossary.html#Layer) to construct a
[network](./deep-learning-glossary.html#Network), optimize the network by
feeding it examples.
That is why Leaf's entire API consists of only two concepts: [Layers][layers]
and [Solvers][solvers]. Layers to construct almost any kind of model - deep,
classical, stochastic or hybrids. And Solvers for executing and optimizing the
model.

This is already the entire API for Machine Learning with Leaf. To learn how
this is possible and how to build Machine Learning applications, refer to 
[2. Layers](./layers.html) and [3. Solvers](./solvers.html). Enjoy!

[leaf]: https://github.com/autumnai/leaf
[layers]: ./layers.html
[solvers]: ./solvers.html

## Benefits+

Leaf was build with three concepts in mind: accessibility/simplicity,
performance and portability. We want that developers and companies are able to
run their Machine Learning application everywhere - servers, desktops,
smartphones, embedded devices. Therefore any platform and
computation language (OpenCL, CUDA, etc.) is a first class citizen in Leaf.

We coupled portability with simplicity, meaning you can deploy your Machine
Learning applications to almost any machine and device - no code changes
required. Learn more at chapter [4. Backend](./backend.html) or at the
[Collenchyma Github repository](https://github.com/autumnai/collenchyma).

## Contributing

Want to contribute? Awesome!
[We have instructions to help you get started](https://github.com/autumnai/leaf/blob/master/CONTRIBUTING.md).

Leaf has a near real-time collaboration culture, which happens at the [Github
repository](https://github.com/autumnai/leaf) and on the
[Leaf Gitter Channel](https://gitter.im/autumnai/leaf).

## API Documentation

Alongside this book you can also read the Rust API documentation if
you would like to use Leaf as a crate, write a library on top of it or
just want a more low-level overview.

[> Rust API documentation][api-docs]

[api-docs]: http://autumnai.github.io/leaf/

## License

Leaf is free for anyone for whatever purpose.
Leaf is licensed under either
[Apache License v2.0](https://github.com/autumnai/leaf/blob/master/LICENSE-APACHE) or,
[MIT license](https://github.com/autumnai/leaf/blob/master/LICENSE-MIT) -
whatever strikes your fancy.
