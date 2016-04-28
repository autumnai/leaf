<a name="0.2.1"></a>
## 0.2.1 (2016-04-21)


#### Bug Fixes

* **sgd:**  initialize weight gradient history with zeroes ([68689955](https://github.com/autumnai/leaf/commit/68689955c29c5e061389067b4dd4729b71404ad0))
* **solvers:**  remove CUDA build flag ([1f5f6b87](https://github.com/autumnai/leaf/commit/1f5f6b87260f7b7c2d202d59cedc686c9c3f6b1e))

#### Breaking Changes

* **container:**  put sequential layer into container dir ([bb23b76b](https://github.com/autumnai/leaf/commit/bb23b76b41935a572135564c41c9df2b627d73d5), breaks [#](https://github.com/autumnai/leaf/issues/))

#### Features

* **container:**  put sequential layer into container dir ([bb23b76b](https://github.com/autumnai/leaf/commit/bb23b76b41935a572135564c41c9df2b627d73d5), breaks [#](https://github.com/autumnai/leaf/issues/))
* **features:**  change meaning of framework features ([58d72f50](https://github.com/autumnai/leaf/commit/58d72f50964f8ddc3fc43d5f8b91b31af5881a7d))
* **layers:**  add tanh layer ([b1d5ec91](https://github.com/autumnai/leaf/commit/b1d5ec913be86c91a58ac281151d2cdf1ca976aa))
* **serialization:**
  *  add deserialization ([df7c9d88](https://github.com/autumnai/leaf/commit/df7c9d88713cffbc50261c488e940f0926edfea4))
  *  add serialization ([cb1a1b4b](https://github.com/autumnai/leaf/commit/cb1a1b4b72eddcbba0414c32829f1e1e23b10ca1))

#### Performance

* **sgd:**  use GPU for computation of weight updates ([08fd965b](https://github.com/autumnai/leaf/commit/08fd965b19a64879bc364f488ebb393b07e3f413))
* **solver:**  don't zero fill weight gradients ([6c4482c5](https://github.com/autumnai/leaf/commit/6c4482c5c17de761516f3cce7e860e5651041ea1))


<a name="0.2.0"></a>
## 0.2.0 (2016-03-04)


#### Bug Fixes

* **convolution:**  add missing weight initialization ([79f71095](https://github.com/autumnai/leaf/commit/79f710955374daf1a878edfdf5dd5977edd75550))
* **dependency:**  make collenchyma version constraint stricter ([6b3f6af3](https://github.com/autumnai/leaf/commit/6b3f6af30005ebed40bf84b3dc9d36770db509a2))
* **nll:**  add NLLConfig to specify number of classes ([34568774](https://github.com/autumnai/leaf/commit/34568774ad0e491f7a31d624f68513b3dedaa14c))
* **reshape:**  fix reshaping of network input blobs ([20d97e9d](https://github.com/autumnai/leaf/commit/20d97e9d42161db7ac5aa95c48c8db04f8f950e9))
* **sequential:**  synchronize after forward/backward ([d1c1030f](https://github.com/autumnai/leaf/commit/d1c1030ff64166263012aad0572cdc7e2be865bf))
* **test:**  fix tests after adding collenchyma ([a7f8a695](https://github.com/autumnai/leaf/commit/a7f8a69521130289fc6b5a3eebc7f28133a7fac4))

#### Features

* **activations:**  add in-place activations ([920b6419](https://github.com/autumnai/leaf/commit/920b64191d642e8536deffdf55d7c26bf287a7b9))
* **convolution:**
  *  remove convolution axis ([a8345ee1](https://github.com/autumnai/leaf/commit/a8345ee1555c0c256f5a30cdabf7ebcc46d52455))
  *  add shared workspace for convolution layer ([f5f25c31](https://github.com/autumnai/leaf/commit/f5f25c31a4c8bd058cd576c789ba732241ab4496))
* **everything:**  introduce most of the changes for 0.2.0 ([1e0db777](https://github.com/autumnai/leaf/commit/1e0db7774b5bdb38615d444c881265ec7cec390e))
* **layer:**  add Sequential layer ([aaacc1ed](https://github.com/autumnai/leaf/commit/aaacc1edf351d2fb07f1f7a375cffac03d9932ed))
* **layers:**  implement Into<LayerType> for all layers ([b9a4e8f6](https://github.com/autumnai/leaf/commit/b9a4e8f6dd86fe1bd80a669ae1aa2e17a017c4c1))
* **license:**  change license to dual MIT/Apache-2.0 ([a06b7c52](https://github.com/autumnai/leaf/commit/a06b7c522c7e9f1c837b96ce27a9eca4b34d2bad))
* **reshape:**  added in-place functionality to reshape layer ([f03bfc20](https://github.com/autumnai/leaf/commit/f03bfc20711451493a8324cad553f7f2f00ffcbe))
* **solvers:**  reintroduce solvers for Layers ([0254a432](https://github.com/autumnai/leaf/commit/0254a432b0d990990564aed8c25b237bda15a685))

#### Performance

* **sequential:**  enable in-place inside Sequential containers ([5f0a40cb](https://github.com/autumnai/leaf/commit/5f0a40cba2becb86eb948363192895855fa49c75))


<a name="0.1.2"></a>
## 0.1.2 (2015-12-19)


#### Bug Fixes

* **dependency:**  make collenchyma version constraint stricter ([594f207c](https://github.com/autumnai/leaf/commit/594f207c129da424637285185ca804429d48c8b0))


<a name="0.1.1"></a>
## 0.1.1 (2015-11-30)


#### Bug Fixes

* **dependency:**  make collenchyma version constraint stricter ([355620ad](https://github.com/autumnai/leaf/commit/355620ad2383973267f3480715f0f160e60d9089))
* **test:**  fix tests after adding collenchyma ([cc0d340e](https://github.com/autumnai/leaf/commit/cc0d340eb9684970ec94d547edbacaa1805fc16f))



<a name="0.1.0"></a>
## 0.1.0 (2015-11-10)


#### Features

* **backend:**  switch to collenchyma and update blob ([7556f55a](https://github.com/autumnai/leaf/commit/7556f55a0bee3b8c73017cdb2023c37831fb5a33))
* **backpropagation:**  implemented backpropagation ([1e97f9d8](https://github.com/autumnai/leaf/commit/1e97f9d8c4ebe32f8fb521de0e1f7183ce78879e))
* **ci:**  Added travis for CI and doc building ([324ea1b0](https://github.com/autumnai/leaf/commit/324ea1b0c92439f447f589219be303ca9e952e87))
* **layer:**  progress on forwarding network; introducted ReadBlob and WriteBlob for Layers ([ab56a021](https://github.com/autumnai/leaf/commit/ab56a02156585747ade254ebaaa074f6c6102bc8))
* **network:**  network forwarding and helpers ([0415f637](https://github.com/autumnai/leaf/commit/0415f637bbcff9301afa5f6bd02a4188cc4022d9))
* **release:**  prepare for 0.0.1 ([52c5a95f](https://github.com/autumnai/leaf/commit/52c5a95f676b18298e14482648b82536bea00a18))
* **solver:**
  *  implement solver and sgd ([83db20d4](https://github.com/autumnai/leaf/commit/83db20d4540240aaa8c0031bd8b67ae4d6e4c264))
  *  calculation of learning rate from config ([84a74449](https://github.com/autumnai/leaf/commit/84a74449d14fd7b8782917dbfe92099d620828ed))
  *  started fleshing out sgd solver ([5985d581](https://github.com/autumnai/leaf/commit/5985d581743f7b0dfeb5f2675b26dadb026f118f))

#### Bug Fixes

* **build:**  added lib blas to travis dependency ([3fd3a285](https://github.com/autumnai/leaf/commit/3fd3a2858811a31c3c8b35a14b8faa2db74f9ea7))
* **cargo:**  fixed homepage spelling ([c71c3196](https://github.com/autumnai/leaf/commit/c71c319639bc77a89eed7a2414b2e620d3890aa6))
* **dependencies:**  locked dependencies more thightly as required by crates.io ([6c2a45ec](https://github.com/autumnai/leaf/commit/6c2a45ec887f9d3aaa42d0a11c13da995ebfb5ac))
* **dim_check:**  fixed layer dimension checking based on the new interface of phloem ([f685ce7e](https://github.com/autumnai/leaf/commit/f685ce7e8da9bb1607636b07d8b8c8b64a989694))
* **docs:**  own gh-pages token for doc upload ([0967dead](https://github.com/autumnai/leaf/commit/0967dead21818aac60204a8f79e4382448215bf7))
* **keywords:**  remove whitespace from cargo keywords ([6b54de82](https://github.com/autumnai/leaf/commit/6b54de823282f6688347bcea4b4def674b90b1ae))
* **phloem:**  updated shape interface change ([6169645c](https://github.com/autumnai/leaf/commit/6169645c4078b1f078cdac1969cf5915297cbcf4))
* **typo:**  broken link in contribution guide ([71aec33b](https://github.com/autumnai/leaf/commit/71aec33b8ae9b1228f90cccb3c375f9b823f2b73))
* **wording:**  Fixed wording of contribution guide; also some smaller typo fixes ([8bdea30a](https://github.com/autumnai/leaf/commit/8bdea30a382fda8cddd46a50784db8b97673f1bb))



<a name="0.0.1"></a>
## 0.0.1 (2015-11-02)


#### Features

* **ci:**  Added travis for CI and doc building ([324ea1b0](https://github.com/autumnai/leaf/commit/324ea1b0c92439f447f589219be303ca9e952e87))
* **layer:**  progress on forwarding network; introducted ReadBlob and WriteBlob for Layers ([ab56a021](https://github.com/autumnai/leaf/commit/ab56a02156585747ade254ebaaa074f6c6102bc8))
* **network:**  network forwarding and helpers ([0415f637](https://github.com/autumnai/leaf/commit/0415f637bbcff9301afa5f6bd02a4188cc4022d9))

#### Bug Fixes

* **build:**  added lib blas to travis dependency ([3fd3a285](https://github.com/autumnai/leaf/commit/3fd3a2858811a31c3c8b35a14b8faa2db74f9ea7))
* **cargo:**  fixed homepage spelling ([c71c3196](https://github.com/autumnai/leaf/commit/c71c319639bc77a89eed7a2414b2e620d3890aa6))
* **dim_check:**  fixed layer dimension checking based on the new interface of phloem ([f685ce7e](https://github.com/autumnai/leaf/commit/f685ce7e8da9bb1607636b07d8b8c8b64a989694))
* **docs:**  own gh-pages token for doc upload ([0967dead](https://github.com/autumnai/leaf/commit/0967dead21818aac60204a8f79e4382448215bf7))
* **phloem:**  updated shape interface change ([6169645c](https://github.com/autumnai/leaf/commit/6169645c4078b1f078cdac1969cf5915297cbcf4))
* **wording:**  Fixed wording of contribution guide; also some smaller typo fixes ([8bdea30a](https://github.com/autumnai/leaf/commit/8bdea30a382fda8cddd46a50784db8b97673f1bb))



