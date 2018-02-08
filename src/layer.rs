//! Provides the generics and interfaces for the specific Layers.
//!
//! See [Layers][layers]
//! [layers]: ../layers/index.html
use co::prelude::*;
use layers::*;
use weight::WeightConfig;
use util::{ArcLock, native_backend, LayerOps};
use std::fmt;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader};
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use leaf_capnp::layer as capnp_layer;
use leaf_capnp::layer_config as capnp_layer_config;
use leaf_capnp::layer_config::layer_type as capnp_layer_type;
use capnp_util::*;

#[derive(Debug)]
/// The generic Layer
pub struct Layer<B: IBackend> {
    /// Identifies the Network
    ///
    /// The name is mainly used for logging purposes.
    pub name: String,
    /// The configuration of the Layer
    pub config: Box<LayerConfig>,
    /// The [implementation][1] of the Layer.
    /// [1]: ../layers/index.html
    ///
    /// This is the part that does most of the work ([forward][2]/[backward][3]).
    /// [2]: ./trait.ILayer.html#method.forward
    /// [3]: ./trait.ILayer.html#method.backward
    pub worker: Box<ILayer<B>>,

    backend: Rc<B>,

    /// Determines if layer will skip comutations for [backward][1] step.
    /// [1]: ./trait.ILayer.html#method.backward
    needs_backward: bool,

    /// The vector that stores shared references to the weights in the form of blobs.
    pub weights_data: Vec<ArcLock<SharedTensor<f32>>>,
    /// The vector that stores shared references to the weights in the form of blobs.
    pub weights_gradient: Vec<ArcLock<SharedTensor<f32>>>,
    // contains all the learnable weights (does not include bias(?) and shared weights)
    learnable_weights: Vec<ArcLock<SharedTensor<f32>>>,
    // learning rate for each weight
    weights_lr: Vec<Option<f32>>,
    // weight decay for each weight
    weights_weight_decay: Vec<Option<f32>>,
    // display name for each weight
    weights_display_names: Vec<String>,

    /// Vector indicating whether to compute the diff of each weight blob.
    ///
    /// You can safely ignore false values and always compute gradients
    /// for all weights, but possibly with wasteful computation.
    ///
    /// Can be used by some [Layer implementations][1] to optimize performance.
    /// [1]: ../layers/index.html
    weight_propagate_down: Vec<bool>,

    /// References to all the input blobs of the layer.
    pub input_blobs_data: Vec<ArcLock<SharedTensor<f32>>>,
    /// References to all the input blobs of the layer.
    pub input_blobs_gradient: Vec<ArcLock<SharedTensor<f32>>>,
    /// Names for all the input blobs of the layer.
    pub input_blob_names: Vec<String>,
    input_need_backwards: Vec<bool>,

    /// References to all the output blobs of the layer.
    pub output_blobs_data: Vec<ArcLock<SharedTensor<f32>>>,
    /// References to all the output blobs of the layer.
    pub output_blobs_gradient: Vec<ArcLock<SharedTensor<f32>>>,
    output_blob_names: Vec<String>,
    /// The vector that indicates whether each output blob contributes to
    /// the [loss][1] of the network and with which weight.
    /// [1]: http://caffe.berkeleyvision.org/tutorial/loss.html
    loss: Vec<f32>,

    /// All the blobs of the layer that can be addressed by name.
    ///
    /// Does not contain anonymous blobs.
    pub blob_names: HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>)>,
}

impl<B: IBackend> Layer<B> {
    /// Connect the layer to another layers and set up tensors for intermediate results and weights.
    ///
    /// Connects to the outputs provided by other layers via the `registry`.
    /// Adds output blobs to the layer and then adds them to the `registry`, so the next
    /// layers can connect them as their inputs.
    /// In the end it initializes the underlying [layer implementation][2].
    ///
    /// [2]: ./trait.ILayer.html
    ///
    /// Called during initialization of containter layers.
    pub fn connect(
        &mut self,
        registry: &mut HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>)>,
        weight_registry: &mut HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>, Option<f32>, Option<f32>)>) {
        // connect to all required inputs
        for input_name in &self.config.inputs.clone() {
            self.connect_input(input_name, registry)
        }
        // setup outputs
        for (output_id, _) in self.config.outputs.clone().iter().rev().enumerate() {
            self.append_output(output_id, registry);
        }
        let config = self.config.clone();
        for (output_id, _) in self.config.outputs.clone().iter().rev().enumerate() {
            self.append_weight(&config, weight_registry, 0, output_id);
        }

        // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
        // specified fewer than the required number (as specified by
        // exact_num_top_blobs() or min_output_blobs()), allocate them here.
        let auto_output_blobs = self.worker.auto_output_blobs();
        debug!("Layer {} - auto_output_blobs: {}", &self.name, &auto_output_blobs);
        let min_output_blobs = self.worker.min_output_blobs();
        let exact_num_output_blobs = self.worker.exact_num_output_blobs().unwrap_or(0);
        if auto_output_blobs {
            let needed_num_outputs = cmp::max(min_output_blobs, exact_num_output_blobs);
            for _ in 0..(needed_num_outputs - self.output_blobs_data.len()) {
                // Add "anonymous" output blobs -- do not add to registry
                // as we don't want these blobs to be usable as input
                // to other layers.
                info!("Adding anonymous output blob for layer {}", &self.name);
                self.create_anonymous_output();
            }
        }

        self.worker.init(self.backend.clone());
        self.reshape();
        self.worker.resize_shared_workspace(self.backend.clone(), None);
        for t in &self.output_blobs_data {
            debug!("Layer {} - output shape: {:?}", self.name, t.read().unwrap().desc());
        }
    }

    /// Append blob as [input blob][1] to the Layer.
    /// [1]: ../layer/index.html
    ///
    /// During network initalization the blobs will be appended to the Layers as per their
    /// [LayerConfig][3]. It is also determined if a output blob skips backpropagation
    /// from [LayerConfig.propagate_down][3] (see also [init_backprop][5]).
    ///
    /// [3]: ../layer/struct.LayerConfig.html
    /// [5]: #method.init_backprop
    fn connect_input(&mut self, blob_name: &str, available_blobs: &mut HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>)>) {
        let input_id = self.config.inputs.iter().position(|input_name| input_name == blob_name).unwrap();

        if !available_blobs.contains_key(&*blob_name) {
            error!("Unknown input blob {} (layer '{}', input_id: {})",
                   blob_name,
                   self.name,
                   input_id);
        }
        info!("Input {:<15} -> Layer {:>15}", blob_name, self.name);

        self.input_blob_names.push(blob_name.to_owned());
        self.input_blobs_data.push(available_blobs.get(&*blob_name).expect(&format!("Unknown blob name {}", blob_name)).0.clone());
        self.input_blobs_gradient.push(available_blobs.get(&*blob_name).expect(&format!("Unknown blob name {}", blob_name)).1.clone());
        // available_blobs.remove(&*blob_name);

        let mut propagate_down = true;
        // Check if the backpropagation on input_id should be skipped
        if !self.config.propagate_down.is_empty() {
            propagate_down = self.config.propagate_down[input_id];
        }
        let need_backward = propagate_down;
        self.input_need_backwards.push(need_backward);
    }

    /// Append blob as [output blob][1] to the Layer.
    /// [1]: ../layer/index.html
    ///
    /// During network initalization the blobs will be appended to the Layers as per their
    /// [LayerConfig][2]. It is also determined if computations can be done in-place, in which
    /// no additional Blob will be allocated.</br>
    /// Finally, the new blob will be added to the registry, so that the other layers can
    /// connect it as their input.
    /// [2]: ../layer/struct.LayerConfig.html
    fn append_output(&mut self,
                  output_id: usize,
                  registry: &mut HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>)>) {
        let layer_config = &self.config;

        let blob_name = layer_config.output(output_id).unwrap().clone();
        let blob_data: ArcLock<SharedTensor<f32>>;
        let blob_gradient: ArcLock<SharedTensor<f32>>;

        if layer_config.input(output_id).is_some() && *layer_config.input(output_id).unwrap() == blob_name {
            info!("Layer {:<15} -> Output {:>15} (in-place)", layer_config.name, blob_name);
            blob_data = registry[&blob_name].0.clone();
            blob_gradient = registry[&blob_name].1.clone();
        } else if registry.contains_key(&blob_name) {
            // If we are not doing in-place computation but have duplicated blobs, raise an
            // error.
            error!("Top blob {} produced by multiple sources.", blob_name);
            return
        } else {
            {
                info!("Layer {:<15} -> Output {:>15}", self.name, blob_name);
                info!("Output {} = {}", output_id, blob_name);
            }

            let backend: Rc<IBackend<F=B::F>> = self.backend.clone();
            blob_data = Arc::new(RwLock::new(SharedTensor::new(&[1,1,1]))); // [1,1,1] for CUDA
            blob_gradient = Arc::new(RwLock::new(SharedTensor::new(&[1,1,1]))); // [1,1,1] for CUDA
        }
        self.output_blob_names.push(blob_name.clone());
        self.output_blobs_data.push(blob_data.clone());
        self.output_blobs_gradient.push(blob_gradient.clone());
        self.blob_names.insert(blob_name.clone(), (blob_data.clone(), blob_gradient.clone()));
        registry.insert(blob_name.clone(), (blob_data.clone(), blob_gradient.clone()));
    }

    /// Append anonymous blob as [output blob][1] to the Layer.
    /// [1]: ../layer/index.html
    ///
    /// [Layer implementations][2] may request creation of anonymous output blobs
    /// via [auto_output_blobs][3]. Since the blobs are not named, other layers can
    /// not use them as their input blobs.
    /// [2]: ./trait.ILayer.html
    /// [3]: ./trait.ILayer.html#method.auto_output_blobs
    fn create_anonymous_output(&mut self) {
        let blob_name = "(automatic)".to_owned();

        info!("{} -> {}", self.name, blob_name);

        let backend: Rc<IBackend<F=B::F>> = self.backend.clone();
        let output_data = Arc::new(RwLock::new(SharedTensor::new(&[1,1,1]))); // [1,1,1] for CUDA
        let output_gradient = Arc::new(RwLock::new(SharedTensor::new(&[1,1,1]))); // [1,1,1] for CUDA
        self.output_blobs_data.push(output_data);
        self.output_blobs_gradient.push(output_gradient);
    }

    fn append_weight(&mut self, layer_config: &LayerConfig, registry: &mut HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>, Option<f32>, Option<f32>)>, layer_id: usize, weight_id: usize) {
        if self.worker.auto_weight_blobs() {
            info!("Layer {} - appending weight", &layer_config.name);
            let weights_len = self.weights_data.len();
            let weight_name = if weights_len > weight_id {
                layer_config.param(weight_id).unwrap().name.clone()
            } else {
                "".to_owned()
            };

            // use weight_name (or weight_id as a fallback) as display_name
            let display_name = if !weight_name.is_empty() {
                weight_name.clone()
            } else {
                format!("{}-{}", self.name, weight_id)
            };
            self.weights_display_names.push(display_name.clone());
            // create name for registry
            let registry_name = format!("SHARED_WEIGHT_{}", display_name);

            // add to tracking vectors
            let net_weight_id = weights_len;
            let output_data = self.output_blobs_data[weight_id].read().unwrap();
            debug!("Layer {} - creating weight and gradient of size {:?}", &layer_config.name, output_data.desc());
            let weight_data = Arc::new(RwLock::new(SharedTensor::new(output_data.desc())));
            let weight_gradient = Arc::new(RwLock::new(SharedTensor::new(output_data.desc())));
            self.weights_data.push(weight_data.clone());
            self.weights_gradient.push(weight_gradient.clone());

            let mut weight_config = &WeightConfig::default();
            if layer_config.params_len() > weight_id {
                weight_config = layer_config.param(weight_id).unwrap();
            }
            // This layer "owns" this weight blob -- it is either anonymous
            // (i.e., not given a weight_name) or explicitly given a name that we
            // haven't already seen.
            if weight_name.is_empty() || !registry.contains_key(&registry_name) {
                // self.weight_owners.push(None);
                if !weight_name.is_empty() {
                    registry.insert(weight_name.clone(),
                        (weight_data.clone(), weight_gradient.clone(), weight_config.lr_mult, weight_config.decay_mult));
                }
                let learnable_weight_id = self.learnable_weights.len();
                self.learnable_weights.push(weight_data.clone());
                // self.learnable_weight_ids.push(learnable_weight_id);
                self.weights_lr.push(weight_config.lr_mult);
                self.weights_weight_decay.push(weight_config.decay_mult);
            } else {
                // Named weight blob with name we've seen before: share weights

                let (shared_weight_data, shared_weight_gradient, shared_lr, shared_decay_mult) = registry.get(&registry_name).unwrap().clone();
                info!("Sharing weight blob '{}'", weight_name.clone());

                // can only share parameters if both have same lr_mult
                if let Some(lr_mult) = weight_config.lr_mult {
                    if let Some(owner_lr_mult) = shared_lr {
                        if !lr_mult.eq(&owner_lr_mult) {
                            error!("Shared param '{}' has mismatched lr_mult.",
                                   weight_name.clone());
                        }
                    } else {
                        // this is the first shared instance that has a lr_mult value so we take that
                        registry.remove(&registry_name).unwrap();
                        registry.insert(registry_name.clone(), (shared_weight_data.clone(), shared_weight_gradient.clone(), weight_config.lr_mult, shared_decay_mult));
                    }
                }
                // can only share weights if both have same decay_mult
                if let Some(decay_mult) = weight_config.decay_mult {
                    if let Some(owner_decay_mult) = shared_decay_mult {
                        if !decay_mult.eq(&owner_decay_mult) {
                            error!("Shared param '{}' has mismatched decay_mult.",
                                   weight_name.clone());
                        }
                    } else {
                        // this is the first shared instance that has a decay_mult value so we take that
                        registry.remove(&registry_name).unwrap();
                        registry.insert(registry_name, (shared_weight_data.clone(), shared_weight_gradient.clone(), shared_lr, weight_config.decay_mult));
                    }
                }
            }
        }
    }

    fn reshape(&mut self) {
        match self.is_using_in_place() {
            false => {
                self.worker.reshape(self.backend.clone(),
                                    &mut self.input_blobs_data,
                                    &mut self.input_blobs_gradient,
                                    &mut self.weights_data,
                                    &mut self.weights_gradient,
                                    &mut self.output_blobs_data,
                                    &mut self.output_blobs_gradient);
            },
            true => {
                self.worker.reshape(self.backend.clone(),
                                    &mut vec![],
                                    &mut vec![],
                                    &mut self.weights_data,
                                    &mut self.weights_gradient,
                                    &mut self.output_blobs_data,
                                    &mut self.output_blobs_gradient);
            },
        }
    }

    /// Initializes layer for [backpropagation][1]
    /// [1]: https://en.wikipedia.org/wiki/Backpropagation
    ///
    /// Go through all the blobs of a layer to determine which blobs contribute to the
    /// loss of the next layer. We can skip backward computation for blobs that don't contribute
    /// to the loss.
    /// If all of the blobs skip backpropagation we set a flag to skip backpropagation
    /// of the whole layer.
    pub fn init_backprop(&mut self,
                     blobs_under_loss: &mut HashSet<String>,
                     blobs_skip_backp: &mut HashSet<String>) {
        let mut layer_contributes_loss = false;
        let mut layer_skip_propagate_down = true;
        for (output_id, _) in self.output_blobs_data.iter().enumerate() {
            let blob_name = self.output_blob_names.get(output_id);

            // layer is a loss layer or under a loss layer
            if self.loss(output_id).is_some() || blob_name.is_some() && blobs_under_loss.contains(blob_name.unwrap()) {
                layer_contributes_loss = true;
            }
            // layer is not marked to skip backpropagation
            if blob_name.is_none() || blob_name.is_some() && !blobs_skip_backp.contains(blob_name.unwrap()) {
                layer_skip_propagate_down = false;
            }
            // layer contributes loss to some
            if layer_contributes_loss && !layer_skip_propagate_down {
                break;
            }
        }

        // If this layer can skip backward computation, also all his input blobs
        // don't need backpropagation
        if self.needs_backward && layer_skip_propagate_down {
            self.needs_backward = false;
            for (input_id, _) in self.input_blobs_data.iter().enumerate() {
                self.input_need_backwards[input_id] = false;
            }
        }
        // layer doesn't contribute loss so it does not need to be backpropagated
        if !layer_contributes_loss {
            self.needs_backward = false;
        }
        {
            info!("{} needs backward computation: {}",
                  self.name,
                  self.needs_backward);
        }

        for (input_id, input_name) in self.input_blob_names.iter().enumerate() {
            if layer_contributes_loss {
                blobs_under_loss.insert(input_name.clone());
            } else {
                self.input_need_backwards[input_id] = false;
            }
            if !self.input_need_backwards[input_id] {
                blobs_skip_backp.insert(input_name.clone());
            }
        }
    }

    /// Set [backpropagation][1] flags to force this layer to backpropagate.
    /// [1]: https://en.wikipedia.org/wiki/Backpropagation
    ///
    /// Is executed during Network initalization if [NetworkConfig][2].force_backward is true.
    /// Forcing backpropagation is useful for debugging.
    pub fn init_force_backward(&mut self) {
        self.needs_backward = true;
        for (input_id, _) in self.input_need_backwards.clone().iter().enumerate() {
            self.input_need_backwards[input_id] =
                *self.input_need_backwards
                     .get(input_id)
                     .unwrap_or(&self.worker.allow_force_backward(input_id));
        }
        for (weight_id, _) in self.weights_data.clone().iter().enumerate() {
            self.set_weight_propagate_down(weight_id, true);
        }
    }

    /// Expose the internal inputs of a container layer.
    fn expose_inputs(&mut self) {
        if let Some(inputs) = self.worker.inputs_data() {
            self.input_blobs_data = inputs;
        }
        if let Some(gradients) = self.worker.inputs_gradients() {
            self.input_blobs_gradient = gradients;
        }
    }

    /// Expose the internal outputs of a container layer.
    fn expose_outputs(&mut self) {
        if let Some(outputs) = self.worker.outputs_data() {
            self.output_blobs_data = outputs;
        }
        if let Some(gradients) = self.worker.outputs_gradients() {
            self.output_blobs_gradient = gradients;
        }
    }

    /// Uses the underlying layer implementation to compute a forward step.
    ///
    /// See [ILayer.forward](./trait.ILayer.html#method.forward)
    pub fn forward(&mut self, inputs: &[ArcLock<SharedTensor<f32>>]) -> Vec<ArcLock<SharedTensor<f32>>> {
        debug!("LAYER: {:?}", &self.name);
        for (input_i, input) in inputs.iter().enumerate() {
            let reshaped_shape = self.input_blobs_data[input_i].read().unwrap().desc().clone();
            self.input_blobs_data[input_i] = input.clone();
            // reshape input tensor to the reshaped shape
            let old_shape = self.input_blobs_data[input_i].read().unwrap().desc().clone();
            if old_shape.size() != reshaped_shape.size() {
                panic!("The provided input does not have the expected shape of {:?}", reshaped_shape);
            }
            self.input_blobs_data[input_i].write().unwrap().reshape(&reshaped_shape).unwrap();
        }

        let forward_time = timeit_loops!(1, {
            if self.is_using_in_place() {
                self.worker.forward(&self.backend, &vec![], &self.weights_data, &mut self.output_blobs_data);
            } else {
                self.worker.forward(&self.backend, &self.input_blobs_data, &self.weights_data, &mut self.output_blobs_data);
            }
        });
        debug!("{:<15} - Forward time: {:.5} ms", &self.name, forward_time / 0.001);
        self.output_blobs_data.clone()
    }

    /// Uses the underlying layer implementation to compute a backward step.
    ///
    /// See [ILayer.backward](./trait.ILayer.html#method.backward)
    pub fn backward(&mut self, output_gradients: &[ArcLock<SharedTensor<f32>>]) -> Vec<ArcLock<SharedTensor<f32>>> {
        if self.needs_backward {
            let input_gradients = self.backward_input(output_gradients);
            self.backward_parameters();
            input_gradients
        } else {
            vec![]
        }
    }

    /// Calculate the gradient w.r.t. input.
    ///
    /// This method is mostly used when doing backpropagation.
    pub fn backward_input(&mut self, output_gradients: &[ArcLock<SharedTensor<f32>>]) -> Vec<ArcLock<SharedTensor<f32>>> {
        for (output_i, output) in output_gradients.iter().enumerate() {
            self.output_blobs_gradient[output_i] = output.clone();
        }

        if self.is_using_in_place() {
            self.worker.backward_input(&self.backend,
                                 &self.weights_data,
                                 &vec![],
                                 &vec![],
                                 &self.input_blobs_data,
                                 &mut self.input_blobs_gradient)
        } else {
            self.worker.backward_input(&self.backend,
                                 &self.weights_data,
                                 &self.output_blobs_data,
                                 &self.output_blobs_gradient,
                                 &self.input_blobs_data,
                                 &mut self.input_blobs_gradient)
        }

        self.input_blobs_gradient.clone()
    }

    /// Calculate the gradient w.r.t. parameters.
    ///
    /// "Parameters" here refers to weights and also possibly bias, depending on the layer.
    ///
    /// This method is mostly used when doing backpropagation.
    pub fn backward_parameters(&mut self) {
        self.worker.backward_parameters(&self.backend,
                             &self.output_blobs_data,
                             &self.output_blobs_gradient,
                             &self.input_blobs_data,
                             &mut self.weights_gradient)
    }

    /// Synchronize the layers backend.
    pub fn synchronize(&self) {
        self.backend.synchronize().unwrap();
    }

    /// Updates the [weights][1] with the weight update computed by the [Solver][2].
    /// [1]: https://en.wikipedia.org/wiki/Synaptic_weight
    /// [2]: ../solver/struct.Solver.html
    ///
    /// Updating the weights is the last step of computing a [Solver][2] minibatch.
    /// The update value is computed in previous steps according to the [learning rate policy][3]
    ///
    /// [3]: ../solver/enum.LRPolicy.html
    pub fn update_weights<SolverB: IBackend + ::util::SolverOps<f32>>(&mut self, backend: &SolverB) {
        // PERF: allocate this scalar once
        let shared_a = ::util::native_scalar(-1f32);
        for (weight_gradient, weight_data) in self.learnable_weights_gradients().iter().zip(&mut self.learnable_weights_data()) {
            backend.axpy(&shared_a, &weight_gradient.read().unwrap(),
                         &mut weight_data.write().unwrap()).unwrap();
        }
    }

    /// Clears the [weights][1] gradients and zero-inits them.
    /// [1]: https://en.wikipedia.org/wiki/Synaptic_weight
    ///
    /// The gradients for the weights accumulate over the backpropagation steps of
    /// a [Solver][2] minibatch and are cleared between each minibatch
    /// to start over with a clean slate.
    ///
    /// [2]: ../solver/struct.Solver.html
    pub fn clear_weights_gradients(&mut self) {
        for weight_gradient in &mut self.learnable_weights_gradients().iter() {
            let filler = ::weight::FillerType::Constant {
                value: 0f32
            };
            filler.fill(&mut weight_gradient.write().unwrap());
        }
    }

    /// Serialize the Layer and it's weights to a Cap'n Proto file at the specified path.
    ///
    /// You can find the capnp schema [here](../../../../capnp/leaf.capnp).
    ///
    /// ```
    /// # #[cfg(feature = "native")]
    /// # mod native {
    /// # use std::rc::Rc;
    /// # use leaf::layer::*;
    /// # use leaf::layers::*;
    /// # use leaf::util;
    /// # pub fn test() {
    /// #
    /// let mut net_cfg = SequentialConfig::default();
    /// // ... set up network ...
    /// let cfg = LayerConfig::new("network", net_cfg);
    ///
    /// let native_backend = Rc::new(util::native_backend());
    /// let mut layer = Layer::from_config(native_backend, &cfg);
    /// // ... do stuff with the layer ...
    /// // ... and save it
    /// layer.save("mynetwork").unwrap();
    /// #
    /// # }}
    /// #
    /// # #[cfg(not(feature = "native"))]
    /// # mod native {
    /// # pub fn test() {}
    /// # }
    /// #
    /// # fn main() {
    /// #     if cfg!(feature = "native") {
    /// #         ::native::test();
    /// #    }
    /// # }
    /// ```
    pub fn save<P: AsRef<Path>>(&mut self, path: P) -> io::Result<()> {
        let path = path.as_ref();
        let ref mut out = try!(File::create(path));

        let mut message = ::capnp::message::Builder::new_default();
        {
            let mut layer = message.init_root::<capnp_layer::Builder>();
            self.write_capnp(&mut layer);
        }
        ::capnp::serialize_packed::write_message(out, &message).unwrap();

        Ok(())
    }

    /// Read a Cap'n Proto file at the specified path and deserialize the Layer inside it.
    ///
    /// You can find the capnp schema [here](../../../../capnp/leaf.capnp).
    ///
    /// ```
    /// # extern crate leaf;
    /// # extern crate collenchyma;
    /// # #[cfg(feature = "native")]
    /// # mod native {
    /// # use std::rc::Rc;
    /// # use leaf::layer::*;
    /// # use leaf::layers::*;
    /// # use leaf::util;
    /// use collenchyma::prelude::*;
    /// # pub fn test() {
    ///
    /// let native_backend = Rc::new(util::native_backend());
    /// # let mut net_cfg = SequentialConfig::default();
    /// # let cfg = LayerConfig::new("network", net_cfg);
    /// # let mut layer = Layer::from_config(native_backend.clone(), &cfg);
    /// # layer.save("mynetwork").unwrap();
    /// // Load layer from file "mynetwork"
    /// let layer = Layer::<Backend<Native>>::load(native_backend, "mynetwork").unwrap();
    /// #
    /// # }}
    /// #
    /// # #[cfg(not(feature = "native"))]
    /// # mod native {
    /// # pub fn test() {}
    /// # }
    /// #
    /// # fn main() {
    /// #     if cfg!(feature = "native") {
    /// #         ::native::test();
    /// #    }
    /// # }
    /// ```
    pub fn load<LB: IBackend + LayerOps<f32> + 'static, P: AsRef<Path>>(backend: Rc<LB>, path: P) -> io::Result<Layer<LB>> {
        let path = path.as_ref();
        let ref mut file = try!(File::open(path));
        let mut reader = BufReader::new(file);

        let message_reader = ::capnp::serialize_packed::read_message(&mut reader,
                                                                     ::capnp::message::ReaderOptions::new()).unwrap();
        let read_layer = message_reader.get_root::<capnp_layer::Reader>().unwrap();

        let name = read_layer.get_name().unwrap().to_owned();
        let layer_config = LayerConfig::read_capnp(read_layer.get_config().unwrap());
        let mut layer = Layer::from_config(backend, &layer_config);
        layer.name = name;

        let read_weights = read_layer.get_weights_data().unwrap();

        let names = layer.learnable_weights_names();
        let weights_data = layer.learnable_weights_data();

        let native_backend = Backend::<Native>::default().unwrap();
        for (i, (name, weight)) in names.iter().zip(weights_data).enumerate() {
            for j in 0..read_weights.len() {
                let capnp_weight = read_weights.get(i as u32);
                if capnp_weight.get_name().unwrap() != name {
                    continue
                }

                let mut weight_lock = weight.write().unwrap();

                let capnp_tensor = capnp_weight.get_tensor().unwrap();
                let mut shape = Vec::new();
                let capnp_shape = capnp_tensor.get_shape().unwrap();
                for k in 0..capnp_shape.len() {
                    shape.push(capnp_shape.get(k) as usize)
                }
                weight_lock.reshape(&shape).unwrap();

                let mut native_slice = weight_lock.write_only(native_backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice::<f32>();
                let data = capnp_tensor.get_data().unwrap();
                for k in 0..data.len() {
                    native_slice[k as usize] = data.get(k);
                }
            }
        }

        Ok(layer)
    }

    /// Sets whether the layer should compute gradients w.r.t. a
    /// weight at a particular index given by `weight_id`.
    ///
    /// See [`weight_propagate_down`][1]
    /// ./struct.Layer.html
    pub fn set_weight_propagate_down(&mut self, weight_id: usize, value: bool) {
        if self.weight_propagate_down.len() <= weight_id {
            self.weight_propagate_down.resize(weight_id + 1, true);
        }
        self.weight_propagate_down[weight_id] = value;

    }

    /// Returns `true` when the layer is using in-place computation.
    ///
    /// For a layer to use in-place computation it needs to support it via `compute_in_place`
    /// and the names of the first input and output tensor have to match.
    pub fn is_using_in_place(&self) -> bool {
        self.worker.compute_in_place() &&
        self.input_blob_names.get(0).is_some() &&
        self.output_blob_names.get(0).is_some() &&
        self.input_blob_names[0] == self.output_blob_names[0]
    }

    /// Returns the names of all the input blobs.
    pub fn input_blob_names(&self) -> &[String] {
        &self.input_blob_names
    }

    /// Returns the [loss weight][1] associated with the weight blob
    /// with id `weight_id`.
    /// [1]: http://caffe.berkeleyvision.org/tutorial/loss.html
    pub fn loss(&self, weight_id: usize) -> Option<&f32> {
        self.loss.get(weight_id)
    }

    /// Returns all the learnable weights in the layer.
    ///
    /// If the layer is a container layer it will return all the weights of the
    /// layers inside it.
    pub fn learnable_weights_data(&self) -> Vec<ArcLock<SharedTensor<f32>>> {
        if let Some(weights) = self.worker.learnable_weights() { weights }
        else { self.weights_data.clone() }
    }

    /// Returns the gradients for all the learnable weights in the layer.
    ///
    /// If the layer is a container layer it will return all the gradients of the
    /// layers inside it.
    pub fn learnable_weights_gradients(&self) -> Vec<ArcLock<SharedTensor<f32>>> {
        if let Some(gradients) = self.worker.learnable_weights_gradients() { gradients }
        else { self.weights_gradient.clone() }
    }

    /// Returns the names of all the learnable weights in the layer.
    ///
    /// If the layer is a container layer it will return all the names of the
    /// layers inside it.
    pub fn learnable_weights_names(&self) -> Vec<String> {
        if let Some(names) = self.worker.learnable_weights_names() { names }
        else { self.weights_display_names.clone() }
    }

    /// Returns the learning rate for all the learnable weights in the layer.
    ///
    /// If the layer is a container layer it will return all learning rates of the
    /// layers inside it.
    pub fn learnable_weights_lr(&self) -> Vec<Option<f32>> {
        if let Some(lr) = self.worker.learnable_weights_lr() { lr }
        // else { self.weights_lr.clone() }
        else {
            self.learnable_weights_data().iter().map(|_| Some(1f32)).collect::<Vec<_>>() }
    }
}

#[allow(unsafe_code)]
unsafe impl<B: IBackend> Send for Layer<B> {}

impl<'a, B: IBackend> CapnpWrite<'a> for Layer<B> {
    type Builder = capnp_layer::Builder<'a>;

    /// Write the Layer into a capnp message.
    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_name(&self.name);
        {
            let mut layer_config = builder.borrow().init_config();
            self.config.write_capnp(&mut layer_config);
        }
        {
            let native_backend = Backend::<Native>::default().unwrap();
            let mut weights = builder.borrow().init_weights_data(self.learnable_weights_names().len() as u32);
            let names = self.learnable_weights_names();
            let weights_data = self.learnable_weights_data();

            for (i, (name, weight)) in names.iter().zip(weights_data).enumerate() {
                let mut capnp_weight = weights.borrow().get(i as u32);
                capnp_weight.set_name(name);

                let weight_lock = weight.write().unwrap();

                let mut tensor = capnp_weight.init_tensor();
                {
                    let mut tensor_shape = tensor.borrow().init_shape(weight_lock.desc().len() as u32);
                    for (i, dim) in weight_lock.desc().iter().enumerate() {
                        tensor_shape.set(i as u32, *dim as u64);
                    }
                }
                {
                    let native_slice = weight_lock.read(native_backend.device())
                        .unwrap().as_native().unwrap().as_slice::<f32>();
                    let mut tensor_data = tensor.borrow().init_data(native_slice.len() as u32);
                    for (i, datum) in native_slice.iter().enumerate() {
                        tensor_data.set(i as u32, *datum);
                    }
                }
            }
        }
    }
}

impl<B: IBackend + LayerOps<f32> + 'static> Layer<B> {
    /// Creates a new Layer from a [LayerConfig][1].
    /// [1]: ./struct.LayerConfig.html
    pub fn from_config(backend: Rc<B>, config: &LayerConfig) -> Layer<B> {
        let cl = config.clone();
        let cfg = Box::<LayerConfig>::new(cl);
        let mut layer = Layer {
            name: cfg.name.clone(),

            needs_backward: true,

            weights_data: Vec::new(),
            weights_gradient: Vec::new(),
            learnable_weights: Vec::new(),
            weight_propagate_down: Vec::new(),
            weights_lr: Vec::new(),
            weights_weight_decay: Vec::new(),
            weights_display_names: Vec::new(),

            input_blobs_data: Vec::new(),
            input_blobs_gradient: Vec::new(),
            input_blob_names: Vec::new(),
            input_need_backwards: Vec::new(),

            output_blobs_data: Vec::new(),
            output_blobs_gradient: Vec::new(),
            output_blob_names: Vec::new(),
            loss: vec![1f32, 1f32, 1f32],

            blob_names: HashMap::new(),

            backend: backend.clone(),

            worker: Layer::<B>::worker_from_config(backend, &cfg),
            config: cfg,
        };
        layer.expose_inputs();
        layer.expose_outputs();

        layer
    }

    /// Helper for [from_config] to match a [LayerType][2] to its [implementation][3].
    /// [1]: #method.from_config
    /// [2]: ./enum.LayerType.html
    /// [3]: ../layers/index.html
    fn worker_from_config(backend: Rc<B>, config: &LayerConfig) -> Box<ILayer<B>> {
        match config.layer_type.clone() {
            #[cfg(all(feature="cuda", not(feature="native")))]
            LayerType::Convolution(layer_config) => Box::new(Convolution::from_config(&layer_config)),
            LayerType::Linear(layer_config) => Box::new(Linear::from_config(&layer_config)),
            LayerType::LogSoftmax => Box::new(LogSoftmax::default()),
            #[cfg(all(feature="cuda", not(feature="native")))]
            LayerType::Pooling(layer_config) => Box::new(Pooling::from_config(&layer_config)),
            LayerType::Sequential(layer_config) => Box::new(Sequential::from_config(backend, &layer_config)),
            LayerType::Softmax => Box::new(Softmax::default()),
            LayerType::ReLU => Box::new(ReLU),
            LayerType::Sigmoid => Box::new(Sigmoid),
            LayerType::NegativeLogLikelihood(layer_config) => Box::new(NegativeLogLikelihood::from_config(&layer_config)),
            LayerType::Reshape(layer_config) => Box::new(Reshape::from_config(&layer_config)),
        }
    }
}

/// A Layer in a Neural Network that can handle forward and backward of a computation step.
pub trait ILayer<B: IBackend> : ComputeOutput<f32, B> + ComputeInputGradient<f32, B> + ComputeParametersGradient<f32, B> {
    /// Initialize the layer for computation.
    ///
    /// Allows for layer-specific one time setup, e.g. precomputing constant values.
    fn init(&mut self, backend: Rc<B>) {}

    /// Adjust to shapes of the output blobs to fit the shapes of the input blobs.
    ///
    /// Should be called during Layer initalization, after [init][2].
    ///
    /// **Caution**: `input_data` should only be reshaped, but not resized.
    ///
    /// [2]: #method.init
    fn reshape(&mut self,
               backend: Rc<B>,
               input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {}

    /// Adjust size of shared workspace.
    ///
    /// Is used by layers that need a workspace.
    /// The layer should either:
    ///
    /// - leave the workspace as is if it bigger than required by this layer
    /// - resize the workspace to the required size if smaller
    /// - create the workspace if the `workspace` is `None`
    ///
    /// The reference to the workspace should be saved in the layer.
    fn resize_shared_workspace(&mut self, backend: Rc<B>, workspace: Option<ArcLock<SharedTensor<u8>>>) -> Option<ArcLock<SharedTensor<u8>>> {
        workspace
    }

    /// Compute the [feedforward][1] layer output using the provided Backend.
    /// [1]: https://en.wikipedia.org/wiki/Feedforward_neural_network
    ///
    /// Aquires read locks for the input tensors
    /// and write locks for the output tensors to ensure sequential computation,
    /// and then passes them to computation method specific function ([forward_cpu][4]).
    ///
    /// [3]: #method.forward_cpu
    #[cfg_attr(lint, allow(map_clone))]
    fn forward(&self,
               backend: &B,
               input_data: &[ArcLock<SharedTensor<f32>>],
               weights_data: &[ArcLock<SharedTensor<f32>>],
               output_data: &mut [ArcLock<SharedTensor<f32>>]) {
        // aquire all the locks
        let inp: Vec<_> = input_data.iter().map(|b| b.read().unwrap()).collect();
        let input_data_: Vec<&SharedTensor<f32>> = inp.iter().enumerate().map(|(_, val)| &**val).collect();

        let wgts: Vec<_> = weights_data.iter().map(|w| w.read().unwrap()).collect();
        let weights_data_: Vec<&SharedTensor<f32>> = wgts.iter().enumerate().map(|(_, val)| &**val).collect();

        let out_ref = output_data.iter().cloned().collect::<Vec<_>>();
        let mut out = &mut out_ref.iter().map(|b| b.write().unwrap()).collect::<Vec<_>>();
        let mut output_w = &mut out.iter_mut().map(|a| a).collect::<Vec<_>>();
        let mut output_data_: Vec<&mut SharedTensor<f32>> = output_w.iter_mut().enumerate().map(|(_, val)| &mut ***val).collect();

        self.compute_output(backend, &weights_data_, &input_data_, &mut output_data_);
    }

    /// Compute the [backpropagation][1] input gradient using the provided backend.
    /// [1]: https://en.wikipedia.org/wiki/Backpropagation
    ///
    /// Aquires write locks for the input blobs to ensure sequential computation,
    /// and then do a [compute_input_gradient][3].
    ///
    /// [3]: ./trait.ComputeInputGradient.html#method.compute_input_gradient
    #[cfg_attr(lint, allow(map_clone))]
    fn backward_input(&self,
                backend: &B,
                weights_data: &[ArcLock<SharedTensor<f32>>],
                output_data: &[ArcLock<SharedTensor<f32>>],
                output_gradients: &[ArcLock<SharedTensor<f32>>],
                input_data: &[ArcLock<SharedTensor<f32>>],
                input_gradients: &mut [ArcLock<SharedTensor<f32>>]) {
        let wgts_data: Vec<_> = weights_data.iter().map(|b| b.read().unwrap()).collect();
        let weights_data_: Vec<&SharedTensor<f32>> = wgts_data.iter().enumerate().map(|(_, val)| &**val).collect();
        let out_data: Vec<_> = output_data.iter().map(|b| b.read().unwrap()).collect();
        let output_data_: Vec<&SharedTensor<f32>> = out_data.iter().enumerate().map(|(_, val)| &**val).collect();
        let out_gradient: Vec<_> = output_gradients.iter().map(|b| b.read().unwrap()).collect();
        let output_gradients_: Vec<&SharedTensor<f32>> = out_gradient.iter().enumerate().map(|(_, val)| &**val).collect();
        let inp_data: Vec<_> = input_data.iter().map(|b| b.read().unwrap()).collect();
        let input_data_: Vec<&SharedTensor<f32>> = inp_data.iter().enumerate().map(|(_, val)| &**val).collect();
        let btm_gradient_ref = input_gradients.iter().cloned().collect::<Vec<_>>();
        let mut btm_gradient = &mut btm_gradient_ref.iter().map(|b| b.write().unwrap()).collect::<Vec<_>>();
        let mut input_gradient = &mut btm_gradient.iter_mut().map(|a| a).collect::<Vec<_>>();
        let mut input_gradients_: Vec<&mut SharedTensor<f32>> = input_gradient.iter_mut().enumerate().map(|(_, val)| &mut ***val).collect();

        self.compute_input_gradient(backend, &weights_data_, &output_data_, &output_gradients_, &input_data_, &mut input_gradients_);
    }

    /// Compute the [backpropagation][1] parameters gradient using the provided backend.
    /// [1]: https://en.wikipedia.org/wiki/Backpropagation
    ///
    /// Aquires write locks for the input blobs to ensure sequential computation,
    /// and then do a [compute_parameters_gradient][4].
    ///
    /// [4]: ./trait.ComputeParametersGradient.html#method.compute_parameters_gradient
    #[cfg_attr(lint, allow(map_clone))]
    fn backward_parameters(&self,
                backend: &B,
                output_data: &[ArcLock<SharedTensor<f32>>],
                output_gradients: &[ArcLock<SharedTensor<f32>>],
                input_data: &[ArcLock<SharedTensor<f32>>],
                weights_gradients: &mut [ArcLock<SharedTensor<f32>>]) {
        let out_data: Vec<_> = output_data.iter().map(|b| b.read().unwrap()).collect();
        let output_data_: Vec<&SharedTensor<f32>> = out_data.iter().enumerate().map(|(_, val)| &**val).collect();
        let out_gradients: Vec<_> = output_gradients.iter().map(|b| b.read().unwrap()).collect();
        let output_gradients_: Vec<&SharedTensor<f32>> = out_gradients.iter().enumerate().map(|(_, val)| &**val).collect();
        let inp_data: Vec<_> = input_data.iter().map(|b| b.read().unwrap()).collect();
        let input_data_: Vec<&SharedTensor<f32>> = inp_data.iter().enumerate().map(|(_, val)| &**val).collect();
        let wgt_gradient_ref = weights_gradients.iter().cloned().collect::<Vec<_>>();
        let mut wgt_gradient = &mut wgt_gradient_ref.iter().map(|b| b.write().unwrap()).collect::<Vec<_>>();
        let mut weights_gradient = &mut wgt_gradient.iter_mut().map(|a| a).collect::<Vec<_>>();
        let mut weights_gradients_: Vec<&mut SharedTensor<f32>> = weights_gradient.iter_mut().enumerate().map(|(_, val)| &mut ***val).collect();

        self.compute_parameters_gradient(backend, &output_data_, &output_gradients_, &input_data_, &mut weights_gradients_);
    }

    /// Return whether "anonymous" output blobs are created automatically for the layer.
    ///
    /// If this method returns true, Network::init will create enough "anonymous" output
    /// blobs to fulfill the requirement specified by [exact_num_output_blobs][1] or
    /// [min_output_blobs][2].
    /// [1]: #method.exact_num_output_blobs
    /// [2]: #method.min_output_blobs
    fn auto_output_blobs(&self) -> bool {
        false
    }
    /// Returns the minimum number of output blobs required by the layer,
    /// or 0 if no minimum number is required.
    ///
    /// This method should be overridden to return a positive value if your
    /// layer expects some minimum number of output blobs.
    fn min_output_blobs(&self) -> usize {
        0
    }
    /// Returns the exact number of output blobs required by the layer,
    /// or `None` if no exact number is required.
    ///
    /// This method should be overridden to return a positive value if your
    /// layer expects some exact number of output blobs.
    fn exact_num_output_blobs(&self) -> Option<usize> {
        None
    }
    /// Return whether weight blobs are created automatically for the layer.
    ///
    /// If this method returns true, Network::init will create a weight blob
    /// for every output blob.
    fn auto_weight_blobs(&self) -> bool {
        false
    }
    /// Returns the exact number of input blobs required by the layer,
    /// or `None` if no exact number is required.
    ///
    /// This method should be overridden to return a positive value if your
    /// layer expects some exact number of input blobs.
    fn exact_num_input_blobs(&self) -> Option<usize> {
        None
    }
    /// Return whether to allow force_backward for a given input blob index.
    ///
    /// If allow_force_backward(i) == false, we will ignore the force_backward
    /// setting and backpropagate to blob i only if it needs gradient information
    /// (as is done when force_backward == false).
    fn allow_force_backward(&self, input_id: usize) -> bool {
        true
    }
    /// Return wether a simple native backend should be used to [sync][1] instead of the default backend.
    /// [1]: #method.sync
    ///
    /// If `false` is returned the default backend will be used, otherwise a new native backend
    /// will be created and provided as argument to `sync`.
    fn sync_native(&self) -> bool {
        false
    }
    /// Return wether the computations of a layer should be done in-place (the output will be written where the input was read from).
    ///
    /// Doing computations in place reduces the memory required for layers.
    ///
    /// If `false` is returned the layer behaves as normal, otherwise
    /// if a layer is provided a identiacla "input" and "output", it will only be supplied an
    /// "output_data" when doing a `compute_output`.
    fn compute_in_place(&self) -> bool {
        false
    }

    /// Return wether the layer is a container.
    ///
    /// This turns of certain behaviour for containers which would lead to problems:
    /// - RwLocks will not be aquired for forward/backward since it would lead to deadlocks.
    fn is_container(&self) -> bool {
        false
    }

    /// Return the associated loss weight for a given output blob index.
    ///
    /// If loss_weight(i) == `None`, no loss will be calculated for the output blob.
    ///
    /// This is usually overridden by loss layers.
    fn loss_weight(&self, output_id: usize) -> Option<f32> {
        None
    }

    /// Return the input tensors of the layer.
    ///
    /// This should only be overridden by container layers,
    /// where the tensors are not easily exposable.
    fn inputs_data(&self) -> Option<Vec<ArcLock<SharedTensor<f32>>>> {
        None
    }

    /// Return the gradients of the input tensors of the layer.
    ///
    /// This should only be overridden by container layers,
    /// where the tensors are not easily exposable.
    fn inputs_gradients(&self) -> Option<Vec<ArcLock<SharedTensor<f32>>>> {
        None
    }

    /// Return the output tensors of the layer.
    ///
    /// This should only be overridden by container layers,
    /// where the tensors are not easily exposable.
    fn outputs_data(&self) -> Option<Vec<ArcLock<SharedTensor<f32>>>> {
        None
    }

    /// Return the gradients of the output tensors of the layer.
    ///
    /// This should only be overridden by container layers,
    /// where the tensors are not easily exposable.
    fn outputs_gradients(&self) -> Option<Vec<ArcLock<SharedTensor<f32>>>> {
        None
    }

    /// Return the learnable weights inside the layer.
    ///
    /// This should only be overridden by container layers,
    /// where the weights are not easily exposable.
    fn learnable_weights(&self) -> Option<Vec<ArcLock<SharedTensor<f32>>>> {
        None
    }

    /// Return the gradients for the learnable weights inside the layer.
    ///
    /// This should only be overridden by container layers,
    /// where the weights are not easily exposable.
    fn learnable_weights_gradients(&self) -> Option<Vec<ArcLock<SharedTensor<f32>>>> {
        None
    }

    /// Return the names of the learnable weights inside the layer.
    ///
    /// This should only be overridden by container layers,
    /// where the weights are not easily exposable.
    fn learnable_weights_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Return the learning rates for the learnable weights inside the layer.
    ///
    /// This should only be overridden by container layers,
    /// where the weights are not easily exposable.
    fn learnable_weights_lr(&self) -> Option<Vec<Option<f32>>> {
        None
    }
}

/// A Layer that can compute the output for a given input.
pub trait ComputeOutput<T, B: IBackend> {
    /// Compute output for given input and write them into `output_data`.
    fn compute_output(&self,
                      backend: &B,
                      weights_data: &[&SharedTensor<T>],
                      input_data: &[&SharedTensor<T>],
                      output_data: &mut [&mut SharedTensor<T>]);
}

/// A Layer that can compute the gradient with respect to its input.
pub trait ComputeInputGradient<T, B: IBackend> {
    /// Compute gradients with respect to the inputs and write them into `input_gradients`.
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights_data: &[&SharedTensor<T>],
                              output_data: &[&SharedTensor<T>],
                              output_gradients: &[&SharedTensor<T>],
                              input_data: &[&SharedTensor<T>],
                              input_gradients: &mut [&mut SharedTensor<T>]);
}

/// A Layer that can compute the gradient with respect to its parameters (= weights, bias, etc.).
pub trait ComputeParametersGradient<T, B: IBackend> {
    /// Compute gradients with respect to the parameters and write them into `parameters_gradients`.
    fn compute_parameters_gradient(&self,
                                   backend: &B,
                                   output_data: &[&SharedTensor<T>],
                                   output_gradients: &[&SharedTensor<T>],
                                   input_data: &[&SharedTensor<T>],
                                   parameters_gradients: &mut [&mut SharedTensor<T>]) {}
}

impl<B: IBackend> fmt::Debug for ILayer<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({})", "ILayer")
    }
}

#[derive(Debug, Clone)]
/// Layer Configuration Struct
pub struct LayerConfig {
    /// The name of the Layer
    pub name: String,

    /// The type of the Layer
    pub layer_type: LayerType,

    /// The name for each output Blob
    pub outputs: Vec<String>,

    /// The name for each input Blob
    pub inputs: Vec<String>,

    /// Specifies training configuration for each weight blob.
    pub params: Vec<WeightConfig>,

    /// Specifies on which inputs the backpropagation should be skipped.
    /// The size must be either 0 or equal to the number of inputs.
    pub propagate_down: Vec<bool>,
}

#[derive(Debug, Clone)]
/// The Layer Types
pub enum LayerType {
    // Common layers
    /// Convolution Layer
    #[cfg(all(feature="cuda", not(feature="native")))]
    Convolution(ConvolutionConfig),
    /// Linear Layer
    Linear(LinearConfig),
    /// LogSoftmax Layer
    LogSoftmax,
    /// Pooling Layer
    #[cfg(all(feature="cuda", not(feature="native")))]
    Pooling(PoolingConfig),
    /// Sequential Layer
    Sequential(SequentialConfig),
    /// Softmax Layer
    Softmax,
    // Activation layers
    /// ReLU Layer
    ReLU,
    /// Sigmoid Layer
    Sigmoid,
    // Loss layers
    /// NegativeLogLikelihood Layer
    NegativeLogLikelihood(NegativeLogLikelihoodConfig),
    // Utility layers
    /// Reshape Layer
    Reshape(ReshapeConfig),
}

impl LayerType {
    /// Returns wether the LayerType supports in-place operations.
    pub fn supports_in_place(&self) -> bool {
        match *self {
            #[cfg(all(feature="cuda", not(feature="native")))]
            LayerType::Convolution(_) => false,
            LayerType::Linear(_) => false,
            LayerType::LogSoftmax => false,
            #[cfg(all(feature="cuda", not(feature="native")))]
            LayerType::Pooling(_) => false,
            LayerType::Sequential(_) => false,
            LayerType::Softmax => false,
            #[cfg(all(feature="cuda", not(feature="native")))]
            LayerType::ReLU => true,
            #[cfg(feature="native")]
            LayerType::ReLU => false,
            #[cfg(all(feature="cuda", not(feature="native")))]
            LayerType::Sigmoid => true,
            #[cfg(feature="native")]
            LayerType::Sigmoid => false,
            LayerType::NegativeLogLikelihood(_) => false,
            LayerType::Reshape(_) => true,
        }
    }

}

impl<'a> CapnpWrite<'a> for LayerType {
    type Builder = capnp_layer_type::Builder<'a>;

    /// Write the LayerType into a capnp message.
    fn write_capnp(&self, builder: &mut Self::Builder) {
        match self {
            #[cfg(all(feature="cuda", not(feature="native")))]
            &LayerType::Convolution(ref cfg) => { let ref mut config = builder.borrow().init_convolution(); cfg.write_capnp(config); },
            &LayerType::Linear(ref cfg) => { let ref mut config = builder.borrow().init_linear(); cfg.write_capnp(config); },
            &LayerType::LogSoftmax => { builder.set_log_softmax(()) },
            #[cfg(all(feature="cuda", not(feature="native")))]
            &LayerType::Pooling(ref cfg) => { let ref mut config = builder.borrow().init_pooling(); cfg.write_capnp(config); },
            &LayerType::Sequential(ref cfg) => { let ref mut config = builder.borrow().init_sequential(); cfg.write_capnp(config); },
            &LayerType::Softmax => { builder.set_softmax(()) },
            #[cfg(all(feature="cuda", not(feature="native")))]
            &LayerType::ReLU => { builder.set_relu(()) },
            #[cfg(feature="native")]
            &LayerType::ReLU => { builder.set_relu(()) },
            #[cfg(all(feature="cuda", not(feature="native")))]
            &LayerType::Sigmoid => { builder.set_sigmoid(()) },
            #[cfg(feature="native")]
            &LayerType::Sigmoid => { builder.set_sigmoid(()) },
            &LayerType::NegativeLogLikelihood(ref cfg) => { let ref mut config = builder.borrow().init_negative_log_likelihood(); cfg.write_capnp(config); },
            &LayerType::Reshape(ref cfg) => { let ref mut config = builder.borrow().init_reshape(); cfg.write_capnp(config); },
        }
    }
}

impl<'a> CapnpRead<'a> for LayerType {
    type Reader = capnp_layer_type::Reader<'a>;

    fn read_capnp(reader: Self::Reader) -> Self {
        match reader.which().unwrap() {
            #[cfg(all(feature="cuda", not(feature="native")))]
            capnp_layer_type::Which::Convolution(read_config) => { let config = ConvolutionConfig::read_capnp(read_config.unwrap()); LayerType::Convolution(config) },
            #[cfg(not(all(feature="cuda", not(feature="native"))))]
            capnp_layer_type::Which::Convolution(_) => { panic!("Can not load Network because Convolution layer is not supported with the used feature flags.") },
            capnp_layer_type::Which::Linear(read_config) => { let config = LinearConfig::read_capnp(read_config.unwrap()); LayerType::Linear(config) },
            capnp_layer_type::Which::LogSoftmax(read_config) => { LayerType::LogSoftmax },
            #[cfg(all(feature="cuda", not(feature="native")))]
            capnp_layer_type::Which::Pooling(read_config) => { let config = PoolingConfig::read_capnp(read_config.unwrap()); LayerType::Pooling(config) },
            #[cfg(not(all(feature="cuda", not(feature="native"))))]
            capnp_layer_type::Which::Pooling(_) => { panic!("Can not load Network because Pooling layer is not supported with the used feature flags.") },
            capnp_layer_type::Which::Sequential(read_config) => { let config = SequentialConfig::read_capnp(read_config.unwrap()); LayerType::Sequential(config) },
            capnp_layer_type::Which::Softmax(_) => { LayerType::Softmax },
            capnp_layer_type::Which::Relu(_) => { LayerType::ReLU },
            capnp_layer_type::Which::Sigmoid(_) => { LayerType::Sigmoid },
            capnp_layer_type::Which::NegativeLogLikelihood(read_config) => { let config = NegativeLogLikelihoodConfig::read_capnp(read_config.unwrap()); LayerType::NegativeLogLikelihood(config) },
            capnp_layer_type::Which::Reshape(read_config) => { let config = ReshapeConfig::read_capnp(read_config.unwrap()); LayerType::Reshape(config) },
        }
    }
}

impl LayerConfig {
    /// Creates a new LayerConfig
    pub fn new<L: Into<LayerType>>(name: &str, layer_type: L) -> LayerConfig {
        LayerConfig {
            name: name.to_owned(),
            layer_type: layer_type.into(),

            outputs: Vec::new(),
            inputs: Vec::new(),

            params: Vec::new(),
            propagate_down: Vec::new(),
        }
    }

    /// Returns the Name of the requested output Blob
    pub fn output(&self, output_id: usize) -> Option<&String> {
        self.outputs.get(output_id)
    }

    /// Returns the number of output Blobs
    pub fn outputs_len(&self) -> usize {
        self.outputs.len()
    }

    /// Add a output by name
    pub fn add_output(&mut self, output_name: &str) {
        self.outputs.push(output_name.to_owned());
    }

    /// Returns the Name of the requested input Blob
    pub fn input(&self, input_id: usize) -> Option<&String> {
        self.inputs.get(input_id)
    }

    /// Returns the number of input Blobs
    pub fn inputs_len(&self) -> usize {
        self.inputs.len()
    }

    /// Add a input by name
    pub fn add_input(&mut self, input_name: &str) {
        self.inputs.push(input_name.to_owned());
    }

    /// Returns the requested WeightConfig
    pub fn param(&self, param_id: usize) -> Option<&WeightConfig> {
        self.params.get(param_id)
    }

    /// Returns the number of params
    pub fn params_len(&self) -> usize {
        self.params.len()
    }

    /// Check if the configured parameters make sense.
    pub fn validate(&self) -> Result<(), &'static str> {
        try!(self.validate_propagate_down_len());
        Ok(())
    }

    /// Checks if propagate down length makes sense.
    fn validate_propagate_down_len(&self) -> Result<(), &'static str> {
        if self.propagate_down.is_empty() || self.propagate_down.len() == self.inputs_len() {
            Ok(())
        } else {
            Err("propagate_down config must be specified either 0 or inputs_len times")
        }
    }
}

impl<'a> CapnpWrite<'a> for LayerConfig {
    type Builder = capnp_layer_config::Builder<'a>;

    /// Write the LayerConfig into a capnp message.
    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_name(&self.name);
        {
            let mut layer_type = builder.borrow().init_layer_type();
            self.layer_type.write_capnp(&mut layer_type);
        }
        {
            let mut outputs = builder.borrow().init_outputs(self.outputs.len() as u32);
            for (i, output) in self.outputs.iter().enumerate() {
                outputs.set(i as u32, &output);
            }
        }
        {
            let mut inputs = builder.borrow().init_inputs(self.inputs.len() as u32);
            for (i, input) in self.inputs.iter().enumerate() {
                inputs.set(i as u32, &input);
            }
        }
        {
            let mut params = builder.borrow().init_params(self.params.len() as u32);
            for (i, param) in self.params.iter().enumerate() {
                let ref mut capnp_param = params.borrow().get(i as u32);
                param.write_capnp(capnp_param);
            }
        }
        {
            let mut propagate_down = builder.borrow().init_propagate_down(self.propagate_down.len() as u32);
            for (i, input) in self.propagate_down.iter().enumerate() {
                propagate_down.set(i as u32, *input);
            }
        }
    }
}

impl<'a> CapnpRead<'a> for LayerConfig {
    type Reader = capnp_layer_config::Reader<'a>;

    fn read_capnp(reader: Self::Reader) -> Self {
        let name = reader.get_name().unwrap().to_owned();
        let layer_type = LayerType::read_capnp(reader.get_layer_type());

        let read_outputs = reader.get_outputs().unwrap();
        let mut outputs = Vec::new();
        for i in 0..read_outputs.len() {
            outputs.push(read_outputs.get(i).unwrap().to_owned())
        }
        let read_inputs = reader.get_inputs().unwrap();
        let mut inputs = Vec::new();
        for i in 0..read_inputs.len() {
            inputs.push(read_inputs.get(i).unwrap().to_owned())
        }

        let read_params = reader.get_params().unwrap();
        let mut params = Vec::new();
        for i in 0..read_params.len() {
            params.push(WeightConfig::read_capnp(read_params.get(i)))
        }

        let read_propagate_down = reader.get_propagate_down().unwrap();
        let mut propagate_down = Vec::new();
        for i in 0..read_propagate_down.len() {
            propagate_down.push(read_propagate_down.get(i))
        }

        LayerConfig {
            name: name,
            layer_type: layer_type,
            outputs: outputs,
            inputs: inputs,
            params: params,
            propagate_down: propagate_down,
        }
    }
}
