@0x8316e0f30c445924;

# The structs here try to mirror all the *Config structs as close as possible.
# Before changing anything take a look at https://capnproto.org/language.html#evolving-your-protocol

struct Weight {
  name @0 :Text;
  tensor @1 :Tensor;
}

struct Tensor {
  shape @0 :List(UInt64);
  data @1 :List(Float32);
}

struct Layer {
  name @0 :Text;
  config @1 :LayerConfig;
  weightsData @2 :List(Weight);
}

struct LayerConfig {
  name @0 :Text;
  layerType :union {
    # Common layers
    convolution @1 :ConvolutionConfig;
    linear @2 :LinearConfig;
    logSoftmax @3 :Void;
    pooling @4 :PoolingConfig;
    sequential @5 :SequentialConfig;
    softmax @6 :Void;
    # Activation layers
    relu @7 :Void;
    sigmoid @8 :Void;
    # Loss layers
    negativeLogLikelihood @9 :NegativeLogLikelihoodConfig;
    # Utility layers
    reshape @10 :ReshapeConfig;
  }

  outputs @11 :List(Text);
  inputs @12 :List(Text);
  params @13 :List(WeightConfig);
  propagateDown @14 :List(Bool);
}

# TODO: incomplete since WeightConfig isn't really used internally in Leaf.
struct WeightConfig {
  name @0 :Text;
}

struct ConvolutionConfig {
  numOutput @0 :UInt64;
  filterShape @1 :List(UInt64);
  stride @2 :List(UInt64);
  padding @3 :List(UInt64);
}

struct LinearConfig {
  outputSize @0 :UInt64;
}

struct PoolingConfig {
  mode @0 :PoolingMode;
  filterShape @1 :List(UInt64);
  stride @2 :List(UInt64);
  padding @3 :List(UInt64);
}

enum PoolingMode {
  max @0;
  average @1; # not implemented yet, but we can't create a single variant enum so this is better than a meaningless "Dummy" value.
}

struct SequentialConfig {
  layers @0 :List(LayerConfig);
  inputs @1 :List(ShapedInput);
  forceBackward @2 :Bool;
}

struct ShapedInput {
  name @0 :Text;
  shape @1 :List(UInt64);
}

struct NegativeLogLikelihoodConfig {
  numClasses @0 :UInt64;
}

struct ReshapeConfig {
  shape @0 :List(UInt64);
}
