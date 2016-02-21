//! Provides the container of a Deep Learning Network
//!
//! Holds all the information about its Layers, how they are connected
//! and how the [forward][1] and [backward][2] steps should be
//! handeled and optimized (e.g. skipping layers).
//!
//! [1]: ./struct.Network.html#method.forward
//! [2]: ./struct.Network.html#method.backward
//!
//! If you are looking to train/test a network, [Solver][3] is usually a better
//! entry point.
//!
//! ## Development
//!
//! Currently only new networks can be created with [from_config][4].
//! In the future there should also be a way to load networks with saved
//! weights from a file.
//! [Issue #14][5].
//!
//! [3]: ../solver/index.html
//! [4]: #method.from_config
//! [5]: https://github.com/autumnai/leaf/issues/14
//! [6]: https://github.com/autumnai/leaf/issues/16
//!
//! ## Glossary
//! ### Input Layers / Blobs
//!
//! A input layer is the bottom-most layer of a network.</br>
//! During a forward step the data is put into the input layer,
//! passed through all the intermediate (hidden) layers and generates a
//! result in the output layer.
//!
//! The blobs in a input layer contain externally preprocessed data that has
//! been brought into a form suitable for consumption by a neural network.
use co::backend::IBackend;
use co::libraries::blas::IBlas;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use shared_memory::*;
use layer::{ILayer, Layer};
use layer::{LayerConfig, WeightConfig};
use phloem::Blob;
use std::rc::Rc;

#[derive(Debug)]
/// Defines a [Network][1] that contains the [Layers][2] and [Blobs][3] that store
/// the intermediate results between the layers which are generated by [forward][4]/[backward][5].
/// [1]: https://en.wikipedia.org/wiki/Artificial_neural_network
/// [2]: ../layer/struct.Layer.html
/// [3]: ../../phloem/blob/struct.Blob.html
/// [4]: ./struct.Network.html#method.forward
/// [5]: ./struct.Network.html#method.backward
///
/// It is also responsible for setting up the connections between the layers.
/// A Network is usually used together with a [Solver][6] to optimize the networks' weights.
///
/// [6]: ../solver/struct.Solver.html
pub struct Network<B: IBackend + IBlas<f32>> {
    /// Identifies the Network
    ///
    /// The name is mainly used for logging purposes.
    pub name: String,
    layers: Vec<Layer<B>>,

    blobs: Vec<ArcLock<HeapBlob>>, // the blobs storing intermediate results between the layer.
    blob_names: Vec<String>,

    input_blobs: Vec<ArcLock<HeapBlob>>,
    output_blobs: Vec<ArcLock<HeapBlob>>,

    weight_owners: Vec<Option<usize>>,
    weight_display_names: Vec<String>,
    weight_layer_indices: Vec<(usize, usize)>,
    weight_names_index: HashMap<String, usize>,

    /// Defines the [parameters/weights][1] of the network.
    /// [1]: https://en.wikipedia.org/wiki/Synaptic_weight
    ///
    /// Parameters are currently in the process of being renamed to weights throughout the codebase.
    /// [Issue #17](https://github.com/autumnai/leaf/issues/17)
    weights: Vec<ArcLock<HeapBlob>>,
    learnable_weights: Vec<ArcLock<HeapBlob>>,
    learnable_weight_ids: Vec<usize>,

    weights_lr: Vec<Option<f32>>,
    weights_weight_decay: Vec<Option<f32>>,
}

impl<B: IBackend + IBlas<f32>> Default for Network<B> {
    fn default() -> Network<B> {
        Network {
            name: "".to_owned(),
            layers: vec![],

            blobs: vec![],
            blob_names: vec![],

            input_blobs: vec![],
            output_blobs: vec![],

            weight_owners: vec![],
            weight_display_names: vec![],
            weight_layer_indices: vec![],
            weight_names_index: HashMap::<String, usize>::new(),

            weights: vec![],
            learnable_weights: vec![],
            learnable_weight_ids: vec![],

            weights_lr: vec![],
            weights_weight_decay: vec![],
        }
    }
}

impl<B: IBackend + IBlas<f32>> Network<B> {
    /// Creates a Network from a [NetworkConfig][1].
    /// [1]: ./struct.NetworkConfig.html
    ///
    /// ## Examples
    ///
    /// ```
    /// # extern crate collenchyma;
    /// # extern crate leaf;
    ///
    /// # use leaf::network::*;
    /// # use collenchyma::backend::{Backend, BackendConfig};
    /// # use collenchyma::frameworks::Native;
    /// # use collenchyma::framework::IFramework;
    /// # use std::rc::Rc;
    ///
    /// # fn main() {
    /// // create backend
    /// let framework = Native::new();
    /// let hardwares = framework.hardwares();
    /// let backend_config = BackendConfig::new(framework, hardwares);
    /// let backend = Rc::new(Backend::new(backend_config).unwrap());
    /// // create network
    /// let cfg = NetworkConfig::default();
    /// Network::from_config(backend, &cfg);
    /// # }
    /// ```
    pub fn from_config(backend: Rc<B>, param: &NetworkConfig) -> Network<B> {
        let mut network = Network::default();
        network.init(backend, param);
        network
    }

    /// Initializes a network.
    ///
    /// Sets up the whole structure of the network. It reads the supplied [NetworkConfig][1],
    /// appends the top and bottom blobs to each layer and determines if the backpropagation has
    /// to be executed for each blob and layer.
    ///
    /// [1]: ./struct.NetworkConfig.html
    fn init(&mut self, backend: Rc<B>, in_config: &NetworkConfig) {
        let config = in_config.clone();
        let registry = &mut HashMap::<String, ArcLock<HeapBlob>>::new();

        for (input_name, input_shape) in config.inputs.iter().zip(config.input_shapes.iter()) {
            self.init_input_blob(&input_name, input_shape, registry);
        }

        for layer_config in &config.layers {
            self.init_layer(backend.clone(), &layer_config, registry);
        }

        // Go through the net backwards to determine which blobs contribute to the
        // loss.  We can skip backward computation for blobs that don't contribute
        // to the loss.
        // Also checks if all bottom blobs don't need backward computation (possible
        // because the skip_propagate_down config) and so we can skip backward
        // computation for the entire layer
        let blobs_under_loss = &mut HashSet::<String>::new();
        let blobs_skip_backp = &mut HashSet::<String>::new();
        for layer in &mut self.layers {
            layer.init_backprop(blobs_under_loss, blobs_skip_backp);
        }

        if config.force_backward {
            for layer in &mut self.layers {
                layer.init_force_backward();
            }
        }

        // In the end, all remaining blobs are considered output blobs.
        for (blob_name, blob) in registry.iter() {
            info!("This network produces output {}", blob_name);
            self.output_blobs.push(blob.clone());
        }

        self.share_weights();

        info!("Network initialization done.");
    }

    /// Initializes a single layer of the network.
    ///
    /// Appends [top][1] and [bottom blobs][2] to the [Layer][3]. Apart from explicitly named
    /// top blobs it will also append anonymous top blobs that are required by the specific
    /// [Layer implemenations][4]. It also sets up the [loss weights],
    /// and backpropagation flags.
    ///
    /// [1]: ../layer/index.html
    /// [2]: ../layer/index.html
    /// [3]: ../layer/struct.Layer.html
    /// [4]: ../layers/index.html
    fn init_layer(&mut self,
                  backend: Rc<B>,
                  layer_config: &LayerConfig,
                  registry: &mut HashMap<String, ArcLock<HeapBlob>>) {
        // Caffe
        // bool share_from_root = !Caffe::root_solver()
        //     && root_net_->layers_[layer_id]->ShareInParallel();
        // // Inherit mode from net if unset.
        // if (!param.layer(layer_id).has_mode()) {
        //   param.mutable_layer(layer_id)->set_mode(mode_);
        // }

        // Setup layer.
        if let Err(e) = layer_config.validate() {
            error!("{}", e);
        }

        info!("Creating Layer {}", layer_config.name.clone());
        let mut layer = Layer::from_config(backend, &layer_config);

        // Figure out this layer's input and output
        // self.layers.last_mut().unwrap().connect(registry);
        layer.connect(registry);

        for (weight_id, _) in layer.blobs.iter().enumerate() {
            let layer_id = self.layers.len();
            self.append_weight(layer_id, weight_id);
        }

        self.layers.push(layer);
    }

    /// Share weights among multiple layers.
    ///
    /// Shared weights are usually used for [Siamese networks][1]
    ///
    /// [1]: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.28.4792
    fn share_weights(&mut self) {
        // Caffe / not sure if ported correctly
        // for (int i = 0; i < params_.size(); ++i) {
        //     if (param_owners_[i] < 0) { continue; }
        //     params_[i]->ShareData(*params_[param_owners_[i]]);
        //     params_[i]->ShareDiff(*params_[param_owners_[i]]);
        // }
        for (i, _) in self.weights.clone().iter().enumerate() {
            if let Some(j) = self.weight_owners[i] {
                assert!(self.weights[i].read().unwrap().capacity() == self.weights[j].read().unwrap().capacity());
                self.weights[i] = self.weights[j].clone(); // sharing whole blob?
            }
        }
    }

    /// Initialize input blobs for the Network.
    ///
    /// Appends a input blob to the network, so the bottom-most [Layer][1] can
    /// [connect][2] to them.
    ///
    /// Used during initialization of the Network.
    /// [1]: ../layer/struct.Layer.html
    /// [2]: ../layer/struct.Layer.html#method.connect
    #[cfg_attr(lint, allow(ptr_arg))]
    fn init_input_blob(&mut self,
                       blob_name: &str,
                       input_shape: &Vec<usize>,
                       registry: &mut HashMap<String, ArcLock<HeapBlob>>) {

        if registry.contains_key(blob_name) {
            // If we are not doing in-place computation but have duplicated blobs, raise an
            // error.
            error!("Top blob {} produced by multiple sources.", blob_name);
            return;
        } else {
            // if (Caffe::root_solver()) {
            {
                info!("Input {} -> {}", self.input_blobs.len(), blob_name);
            }

            let blob: ArcLock<HeapBlob> = Arc::new(RwLock::new(Box::new(Blob::new())));
            let blob_id = self.blobs.len();
            self.blobs.push(blob.clone());
            self.blob_names.push(blob_name.to_owned());

            // Set the (explicitly specified) dimensions of the input blob.
            // let input_shape = config.input_shape(top_id).unwrap().clone();
            blob.write().unwrap().reshape(&input_shape.clone());

            self.input_blobs.push(blob.clone());
            registry.insert(blob_name.to_owned(), blob);
        }
    }

    /// Append a weight blob to the network.
    ///
    /// During network initalization weight blobs are appended to the correct layers.
    /// If a layer's [LayerConfig][1] states that the weights are shared,
    /// this function also makes sure to set a reference to the other weight blob instead of
    /// allocating a new one.
    ///
    /// [1]: ../layer/struct.LayerConfig.html
    fn append_weight(&mut self, layer_id: usize, weight_id: usize) {
        let layer_config = self.layers[layer_id].config.clone();
        let weights_len = self.weights.len();
        let weight_name = if weights_len > weight_id {
            layer_config.param(weight_id).unwrap().name.clone()
        } else {
            "".to_owned()
        };

        // use weight_name (or weight_id as a fallback) as display_name
        if !weight_name.is_empty() {
            self.weight_display_names.push(weight_name.clone());
        } else {
            self.weight_display_names.push(format!("{}", weight_id));
        }

        // add to tracking vectors
        let net_weight_id = weights_len;
        self.weights.push(self.layers[layer_id].blobs[weight_id].clone());
        self.weight_layer_indices.push((layer_id, weight_id));

        let mut weight_config = &WeightConfig::default();
        if layer_config.params_len() > weight_id {
            weight_config = layer_config.param(weight_id).unwrap();
        }
        // This layer "owns" this weight blob -- it is either anonymous
        // (i.e., not given a weight_name) or explicitly given a name that we
        // haven't already seen.
        if weight_name.is_empty() || !self.weight_names_index.contains_key(&weight_name) {
            self.weight_owners.push(None);
            if !weight_name.is_empty() {
                self.weight_names_index.insert(weight_name.clone(), net_weight_id);
            }
            let learnable_weight_id = self.learnable_weights.len();
            self.learnable_weights.push(self.weights[net_weight_id].clone());
            self.learnable_weight_ids.push(learnable_weight_id);
            self.weights_lr.push(weight_config.lr_mult.clone());
            self.weights_weight_decay.push(weight_config.decay_mult.clone());
        } else {
            // Named weight blob with name we've seen before: share weights

            let owner_net_weight_id = *self.weight_names_index.get(&weight_name).unwrap();
            self.weight_owners.push(Some(owner_net_weight_id));
            let (owner_layer_id, owner_weight_id) = self.weight_layer_indices[owner_net_weight_id];
            info!("Sharing weights '{}' owned by layer '{}', weight index {}",
                  weight_name.clone(),
                  self.layers[owner_layer_id].name,
                  owner_weight_id);
            let this_blob = self.layers[layer_id].blobs[weight_id].clone();
            let owner_blob = self.layers[owner_layer_id].blobs[owner_weight_id].clone();
            // can only share weights if blobs match by shape or capacity
            if weights_len > weight_id {
                if let Err(e) = layer_config.param(weight_id)
                                            .unwrap()
                                            .check_dimensions(&this_blob.read().unwrap(),
                                                              &owner_blob.read().unwrap(),
                                                              weight_name.clone(),
                                                              self.layers[owner_layer_id].name.clone(),
                                                              self.layers[layer_id].name.clone()) {
                    error!("{}", e)
                }
            }

            let learnable_weight_id = self.learnable_weight_ids[owner_net_weight_id];
            self.learnable_weight_ids.push(learnable_weight_id);
            // can only share parameters if both have same lr_mult
            if let Some(lr_mult) = weight_config.lr_mult {
                if let Some(owner_lr_mult) = self.weights_lr[learnable_weight_id] {
                    if !lr_mult.eq(&owner_lr_mult) {
                        error!("Shared param '{}' has mismatched lr_mult.",
                               weight_name.clone());
                    }
                } else {
                    self.weights_lr[learnable_weight_id] = weight_config.lr_mult;
                }
            }
            // can only share weights if both have same decay_mult
            if let Some(decay_mult) = weight_config.decay_mult {
                if let Some(owner_decay_mult) = self.weights_weight_decay[learnable_weight_id] {
                    if !decay_mult.eq(&owner_decay_mult) {
                        error!("Shared param '{}' has mismatched decay_mult.",
                               weight_name.clone());
                    }
                } else {
                    self.weights_weight_decay[learnable_weight_id] = weight_config.decay_mult;
                }
            }
        }
    }

    /// Computes [forward][1] and [backward][2] step for the network and returns [the total loss.][3]
    /// [1]: #method.forward
    /// [2]: #method.backward
    /// [3]: http://caffe.berkeleyvision.org/tutorial/loss.html
    ///
    /// Used by the [Solver][4] to conveniently compute one [forward- and one backward-propagation
    /// step][5] together, which is all the network has to do while training it.
    ///
    /// [4]: ../solver/struct.Solver.html
    /// [5]: https://en.wikipedia.org/wiki/Backpropagation#Phase_1:_Propagation
    pub fn forward_backward(&mut self, bottom: &[ArcLock<HeapBlob>]) -> f32 {
        let loss = &mut 0f32;

        self.forward(bottom, loss);
        self.backward();

        *loss
    }

    /// Copies supplied [input Blobs][1] into the network, computes [forward step][2] for the
    /// network and returns [the output blobs.][3].
    /// [1]: ./index.html#input-layers--blobs
    /// [2]: https://en.wikipedia.org/wiki/Feedforward_neural_network
    /// [3]: http://caffe.berkeleyvision.org/tutorial/loss.html
    ///
    /// Does not actually copy data, only references to the input blobs.
    ///
    /// This is the go-to if you just want to feed data to your network and get the corresponding
    /// output.
    pub fn forward(&mut self, input: &[ArcLock<HeapBlob>], loss: &mut f32) -> &Vec<ArcLock<HeapBlob>> {
        for (i, inp) in input.iter().enumerate() {
            self.input_blobs[i] = inp.clone();
        }

        self.forward_prefilled(Some(loss))
    }

    /// Computes [forward step][1] for a network whose [input blob][2] references have been set
    /// and returns [the output blobs.][3]
    /// [1]: https://en.wikipedia.org/wiki/Feedforward_neural_network
    /// [2]: ./index.html#input-layers--blobs
    /// [3]: http://caffe.berkeleyvision.org/tutorial/loss.html
    ///
    /// Can be used if you need more control over how to put data into the network (debugging),
    /// otherwise [forward][4] is the prefered method to forward through the whole network.
    ///
    /// [4]: #method.forward
    pub fn forward_prefilled(&mut self, loss: Option<&mut f32>) -> &Vec<ArcLock<HeapBlob>> {
        let end = self.layers.len() - 1;
        match loss {
            Some(loss_result) => {
                // not sure if loss_result will really be changed
                *loss_result = self.forward_from_to(0, end);
            }
            None => {
                self.forward_from_to(0, end);
            }
        }

        &self.output_blobs
    }

    /// Compute [forward step][1] for a part of (or the whole) network and returns the [total loss][2].
    /// [1]: https://en.wikipedia.org/wiki/Feedforward_neural_network
    /// [2]: http://caffe.berkeleyvision.org/tutorial/loss.html
    ///
    /// Computes the forward step from the layer with index `start` to the layer with index `end`
    /// and return the total [scalar loss][2] over all loss layers.
    ///
    /// If you want to compute a foward step for the whole network
    /// you should use [forward_prefilled][3].
    /// Computing a forward on a part of the network is usually only done for debugging purposes.
    ///
    /// [3]: #method.forward_prefilled
    pub fn forward_from_to(&mut self, start: usize, end: usize) -> f32 {
        assert!(end < self.layers.len());

        let mut loss = 0f32;

        for i in start..end {
            loss += self.layers[i].forward();
        }

        loss
    }

    /// Computes a [backpropagation][1] step for the whole network using the currently set output blobs.
    /// [1]: https://en.wikipedia.org/wiki/Backpropagation
    ///
    /// Computes the backpropagation step for each layer of the Network using [backward_from_to][2].
    /// [2]: #method.backward_from_to
    ///
    /// Called directly only for debugging purposes.
    /// Backpropagating a network is only useful during training and handled by a [Solver][3]
    /// [3]: ../solver/index.html
    pub fn backward(&mut self) {
        let start = self.layers.len() - 1;
        self.backward_from_to(start, 0);
    }

    /// Compute [backpropagation][1] step for a part of (or the whole) network.
    /// [1]: https://en.wikipedia.org/wiki/Backpropagation
    ///
    /// Computes the backpropagation step from the layer with index `start` to the layer with index `end`,
    /// skipping layers that have been flagged to be skipped (usually in [init_backprop][2]).
    /// [2]: #method.init_backprop
    ///
    /// If you want to compute a foward step for the whole network you should use [backward][3].
    /// Computing a backward on a part of the network is usually only done for debugging purposes.
    /// [3]: #method.backward
    pub fn backward_from_to(&mut self, start: usize, end: usize) {
        assert!(start < self.layers.len());

        for i in start..end {
            self.layers[i].backward();
        }
    }

    /// Clears the [weights][1] diffs and zero-inits them.
    /// [1]: https://en.wikipedia.org/wiki/Synaptic_weight
    ///
    /// The diffs for the weights accumulate over the backpropagation steps of
    /// a [Solver][2] minibatch and are cleared between each minibatch
    /// to start over with a clean slate.
    ///
    /// [2]: ../solver/struct.Solver.html
    pub fn clear_weight_diffs(&mut self) {
        for weight_blob in &mut self.learnable_weights.iter() {
            // TODO
            // for p in weight_blob.write().unwrap().mut_diff().iter_mut() {
            //     *p = 0f32;
            // }
        }
    }

    /// Updates the [weights][1] with the weight update computed by the [Solver][2].
    /// [1]: https://en.wikipedia.org/wiki/Synaptic_weight
    /// [2]: ../solver/struct.Solver.html
    ///
    /// Updating the weights is the last step of computing a [Solver][2] minibatch.
    /// The update value is computed in previous steps according to the [learning rate policy][3]
    ///
    /// [3]: ../solver/enum.LRPolicy.html
    pub fn update_weights(&mut self) {
        for weight_blob in &self.learnable_weights {
            weight_blob.write().unwrap().apply_diff()
        }
    }

    #[allow(missing_docs)]
    pub fn learnable_weights(&self) -> &Vec<ArcLock<HeapBlob>> {
        &self.learnable_weights
    }

    #[allow(missing_docs)]
    pub fn weights_weight_decay(&self) -> &Vec<Option<f32>> {
        &self.weights_weight_decay
    }

    #[allow(missing_docs)]
    pub fn weights_lr(&self) -> &Vec<Option<f32>> {
        &self.weights_lr
    }
}

#[derive(Debug, Clone)]
/// Defines the configuration of a network.
///
/// TODO: [DOC] When and why would you use this?
/// TODO: [DOC] What is the purpose of this configuration type?
///
/// TODO: [DOC] <Now-What> Examples
pub struct NetworkConfig {
    /// Defines the name the network.
    pub name: String,

    /// Defines the names of the [input blobs][1].
    /// [1]: ./index.html#input-layers--blobs
    ///
    /// The input blobs are identified by name so they can be referenced as [bottom blobs][2]
    /// in a [LayerConfig][3].
    ///
    /// [2]: ../layer/index.html
    /// [3]: ../layer/struct.LayerConfig.html
    pub inputs: Vec<String>,

    /// Defines the [shape][1] of the [input blobs][2].
    /// [1]: ???
    /// [2]: ./index.html#input-layers--blobs
    ///
    /// The number of input_shapes supplied should match the number of inputs supplied.
    /// The shape of the input blobs has to be known so that the right connections to the
    /// upper layers can be set up.
    pub input_shapes: Vec<Vec<usize>>,

    /// Defines if the network will force every layer to do [backpropagation][1].
    /// [1]: https://en.wikipedia.org/wiki/Backpropagation
    ///
    /// If set to `false`, then the execution of backpropagation is determined automatically
    /// according to the net structure and learning rates.
    ///
    /// Default: `false`
    pub force_backward: bool,

    /// Defines the [state][1] of the network.
    /// [1]: ../struct.NetworkState.html
    ///
    /// Some layers may be included/excluded depending on this state and the states
    /// specified in the layers' include and exclude fields.
    pub state: NetworkState,

    /// Defines if the network will print debugging information about results
    ///
    /// Default: `false`
    pub debug_info: bool,

    /// Defines the layers of the network via [LayerConfig][1]s.
    /// [1]: ../layer/struct.LayerConfig.html
    pub layers: Vec<LayerConfig>,
}

impl Default for NetworkConfig {
    fn default() -> NetworkConfig {
        NetworkConfig {
            name: "".to_owned(),
            inputs: Vec::new(),
            input_shapes: Vec::new(),

            force_backward: false,
            debug_info: false,

            layers: Vec::new(),
            state: NetworkState::default(),
        }
    }
}

impl NetworkConfig {
    #[allow(missing_docs)]
    pub fn layer(&self, layer_id: usize) -> Option<&LayerConfig> {
        self.layers.get(layer_id)
    }

    #[allow(missing_docs)]
    pub fn input(&self, input_id: usize) -> Option<&String> {
        self.inputs.get(input_id)
    }

    #[allow(missing_docs)]
    pub fn input_shape(&self, input_id: usize) -> Option<&Vec<usize>> {
        self.input_shapes.get(input_id)
    }
}

#[derive(Debug, Clone)]
/// Defines the state of a network.
pub struct NetworkState {
    /// Defines the current mode of the network.
    ///
    /// Default: Test
    pub mode: NetworkMode,
    /// TODO: [DOC] what does this do?
    /// TODO: [DOC] could it be of type usize?
    ///
    /// Default: 0
    pub level: isize,
    /// TODO: [DOC] what does this do?
    ///
    /// Default: vec![]
    pub stage: Vec<String>,
}

impl Default for NetworkState {
    fn default() -> NetworkState {
        NetworkState {
            mode: NetworkMode::Test,
            level: 0,
            stage: vec![],
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// Defines the possible modes that a network can be in.
pub enum NetworkMode {
    #[allow(missing_docs)]
    Train,
    #[allow(missing_docs)]
    Test,
}
