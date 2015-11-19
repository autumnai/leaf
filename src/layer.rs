//! Provides the generics and interfaces for the specific [Layers][layers].
//! [layers]: ../layers/index.html
use math::*;
use phloem::{Blob, Numeric};
use shared_memory::{ArcLock, HeapBlob};
use layers::*;
use std::fmt;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use std::sync::{RwLockReadGuard, RwLockWriteGuard};

/// Secures sequential execution as bottom Blob for a forward and as top Blob for a backward
/// operation.
///
/// Ensures that no layer is reading the HeapBlob, while the current layer is still writing.
/// The RwLockReadGuard unlocks automatically as soon as the {forward, backward} operation of
/// the layer is finished and allows for a quick operation transition to the following layer.
/// Is automatically created by the {forward, backward} method of a [Layer][1] and passed to the
/// specific [forward_{cpu, gpu}][2] implementation.
/// [1]: ./trait.ILayer.html#method.forward
/// [2]: ./trait.ILayer.html#tymethod.forward_cpu
///
/// ## Example
///
/// Creates a ReadBlob for seldom scenarios such as testing.
///
/// ```
/// extern crate phloem;
/// # extern crate leaf;
/// use phloem::Blob;
/// use std::sync::{RwLock, RwLockReadGuard};
/// # use leaf::layer::ReadBlob;
///
/// # fn main() {
/// let lock = RwLock::new(Box::new(Blob::<f32>::of_shape(vec![3])));
/// let read_blob: ReadBlob = lock.read().unwrap();
/// # }
/// ```
pub type ReadBlob<'_> = RwLockReadGuard<'_, HeapBlob>;

/// Secures sequential execution as top Blob for a forward and as bottom Blob for a backward
/// operation.
///
/// Ensures that no layer is writing to the HeapBlob, while the current layer is still reading it.
/// The RwLockWriteGuard unlocks automatically as soon as the {forward, backward} operation of
/// the layer is finished and allows for a quick operation transition to the following layer.
/// Is automatically created by the {forward, backward} method of a [Layer][1] and passed to the
/// specific [forward_{cpu, gpu}][2] implementation.
/// [1]: ./trait.ILayer.html#method.forward
/// [2]: ./trait.ILayer.html#tymethod.forward_cpu
///
/// ## Example
///
/// Creates a ReadBlob for seldom scenarios such as testing.
///
/// ```
/// extern crate phloem;
/// # extern crate leaf;
/// use phloem::Blob;
/// use std::sync::{RwLock, RwLockWriteGuard};
/// # use leaf::layer::WriteBlob;
///
/// # fn main() {
/// let lock = RwLock::new(Box::new(Blob::<f32>::of_shape(vec![3])));
/// let read_blob: WriteBlob = lock.write().unwrap();
/// # }
/// ```
pub type WriteBlob<'_> = RwLockWriteGuard<'_, HeapBlob>;

#[derive(Debug)]
/// The generic Layer
pub struct Layer {
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
    pub worker: Box<ILayer>,

    /// Determies if layer will skip comutations for [backward][1] step.
    /// [1]: ./trait.ILayer.html#method.backward
    needs_backward: bool,

    /// The vector that stores shared references to the weights in the form of blobs.
    pub blobs: Vec<ArcLock<HeapBlob>>,

    /// The vector that indicates whether each top blob contributes to
    /// the [loss][1] of the network and with which weight.
    /// [1]: http://caffe.berkeleyvision.org/tutorial/loss.html
    loss: Vec<f32>,

    /// Vector indicating whether to compute the diff of each weight blob.
    ///
    /// You can safely ignore false values and always compute gradients
    /// for all weights, but possibly with wasteful computation.
    ///
    /// Can be used by some [Layer implementations][1] to optimize performance.
    /// [1]: ../layers/index.html
    weight_propagate_down: Vec<bool>,

    /// References to all the bottom blobs of the layer.
    pub bottom_blobs: Vec<ArcLock<HeapBlob>>,
    bottom_blob_names: HashMap<String, (usize, ArcLock<HeapBlob>)>,
    bottom_need_backwards: Vec<bool>,

    /// References to all the top blobs of the layer.
    pub top_blobs: Vec<ArcLock<HeapBlob>>,
    top_blob_names: HashMap<String, (usize, ArcLock<HeapBlob>)>,

    /// All the blobs of the layer that can be addressed by name.
    ///
    /// Does not contain anonymous blobs.
    pub blob_names: HashMap<String, ArcLock<HeapBlob>>,
}

impl Layer {
    /// Creates a new Layer from a [LayerConfig][1].
    /// [1]: ./struct.LayerConfig.html
    ///
    /// Used during [Network][2] initalization.
    ///
    /// [2]: ../network/struct.Network.html
    pub fn from_config(config: &LayerConfig) -> Layer {
        let cl = config.clone();
        let cfg = Box::<LayerConfig>::new(cl);
        Layer {
            name: cfg.name.clone(),

            needs_backward: true,

            blobs: Vec::new(),
            loss: Vec::new(),
            weight_propagate_down: Vec::new(),

            bottom_blobs: Vec::new(),
            bottom_blob_names: HashMap::new(),
            bottom_need_backwards: Vec::new(),

            top_blobs: Vec::new(),
            top_blob_names: HashMap::new(),

            blob_names: HashMap::new(),

            worker: Layer::worker_from_config(&cfg),
            config: cfg,
        }
    }

    /// Helper for [from_config] to match a [LayerType][2] to its [implementation][3].
    /// [1]: #method.from_config
    /// [2]: ./enum.LayerType.html
    /// [3]: ../layers/index.html
    fn worker_from_config(config: &LayerConfig) -> Box<ILayer> {
        match config.layer_type {
            LayerType::Sigmoid => Box::new(Sigmoid),
        }
    }

    /// Connect layer to the other layers in a [Network][1] and set up Blobs.
    /// [1]: ../network/struct.Network.html
    ///
    /// Connects to the bottoms provided by other layers via the `registry`.
    /// Adds top blobs to the layer and then adds them to the `registry`, so the next
    /// layers can connect them as their bottoms.
    /// In the end it intializes the underlying [layer implementation][2].
    ///
    /// [2]: ./trait.ILayer.html
    ///
    /// Called during [Network][1] initialization.
    pub fn connect(&mut self, registry: &mut HashMap<String, ArcLock<HeapBlob>>) {
        // connect to all required bottoms
        for bottom_name in &self.config.bottoms.clone() {
            self.connect_bottom(bottom_name, registry)
        }
        // setup tops
        for (top_id, _) in self.config.tops.clone().iter().rev().enumerate() {
            self.append_top(top_id, registry);
        }

        // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
        // specified fewer than the required number (as specified by
        // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
        let auto_top_blobs = self.worker.auto_top_blobs();
        let min_top_blobs = self.worker.min_top_blobs();
        let exact_num_top_blobs = self.worker.exact_num_top_blobs();
        if auto_top_blobs {
            let needed_num_top = cmp::max(min_top_blobs, exact_num_top_blobs);
            for _ in 0..(needed_num_top - self.top_blobs.len()) {
                // Add "anonymous" top blobs -- do not add to registry
                // as we don't want these blobs to be usable as input
                // to other layers.
                info!("Adding anonymous top blob");
                self.create_anonymous_top();
            }
        }

        self.worker.init();
    }

    /// Append blob as [bottom blob][1] to the Layer.
    /// [1]: ../layer/index.html
    ///
    /// During network initalization the blobs will be appended to the Layers as per their
    /// [LayerConfig][3]. It is also determined if a bottom blob skips backpropagation
    /// from [LayerConfig.propagate_down][3] (see also [init_backprop][5]).
    ///
    /// [3]: ../layer/struct.LayerConfig.html
    /// [5]: #method.init_backprop
    fn connect_bottom(&mut self, blob_name: &str, available_blobs: &mut HashMap<String, ArcLock<HeapBlob>>) {
        let bottom_id = self.config.bottoms.iter().position(|bottom_name| bottom_name == blob_name).unwrap();

        if !available_blobs.contains_key(&*blob_name) {
            error!("Unknown bottom blob {} (layer '{}', bottom_id: {})",
                   blob_name,
                   self.name,
                   bottom_id);
        }
        info!("{} <- {}", self.name, blob_name);

        self.bottom_blob_names.insert(blob_name.to_owned(), (self.bottom_blobs.len(), available_blobs[&*blob_name].clone()));
        self.bottom_blobs.push(available_blobs[&*blob_name].clone());
        available_blobs.remove(&*blob_name);

        let mut propagate_down = true;
        // Check if the backpropagation on bottom_id should be skipped
        if !self.config.propagate_down.is_empty() {
            propagate_down = self.config.propagate_down[bottom_id];
        }
        let need_backward = propagate_down;
        self.bottom_need_backwards.push(need_backward);
    }

    /// Append blob as [top blob][1] to the Layer.
    /// [1]: ../layer/index.html
    ///
    /// During network initalization the blobs will be appended to the Layers as per their
    /// [LayerConfig][2]. It is also determined if computations can be done in-place, in which
    /// no additional Blob will be allocated.</br>
    /// Finally, the new blob will be added to the registry, so that the other layers can
    /// connect it as their bottom.
    /// [2]: ../layer/struct.LayerConfig.html
    fn append_top(&mut self,
                  top_id: usize,
                  registry: &mut HashMap<String, ArcLock<HeapBlob>>) {
        let layer_config = &self.config;

        let blob_name = layer_config.top(top_id).unwrap().clone();
        let blob: ArcLock<HeapBlob>;

        if layer_config.bottom(top_id).is_some() && *layer_config.bottom(top_id).unwrap() == blob_name {
            info!("{} -> {} (in-place)", layer_config.name, blob_name);
            blob = registry[&blob_name].clone();
        } else if registry.contains_key(&blob_name) {
            // If we are not doing in-place computation but have duplicated blobs, raise an
            // error.
            error!("Top blob {} produced by multiple sources.", blob_name);
            return
        } else {
            // if (Caffe::root_solver()) {
            {
                info!("{} -> {}", layer_config.name, blob_name);
                info!("Input {} -> {}", top_id, blob_name);
            }

            blob = Arc::new(RwLock::new(Box::new(Blob::new())));
        }
        self.top_blob_names.insert(blob_name.clone(), (self.top_blobs.len(),blob.clone()));
        self.top_blobs.push(blob.clone());
        self.blob_names.insert(blob_name.clone(), blob.clone());
        registry.insert(blob_name.clone(), blob.clone());
    }

    /// Append anonymous blob as [top blob][1] to the Layer.
    /// [1]: ../layer/index.html
    ///
    /// [Layer implementations][2] may request creation of anonymous top blobs
    /// via [auto_top_blobs][3]. Since the blobs are not named, other layers can
    /// not use them as their bottom blobs.
    /// [2]: ./trait.ILayer.html
    /// [3]: ./trait.ILayer.html#method.auto_top_blobs
    fn create_anonymous_top(&mut self) {
        let blob_name = "(automatic)".to_owned();

        info!("{} -> {}", self.name, blob_name);

        let blob: ArcLock<HeapBlob> = Arc::new(RwLock::new(Box::new(Blob::new())));
        self.top_blobs.push(blob);
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
        for (top_id, top_blob) in self.top_blobs.iter().enumerate() {
            let blob_name = self.name_for_blob(top_blob);

            // layer is a loss layer or under a loss layer
            if self.loss(top_id).is_some() || blobs_under_loss.contains(blob_name) {
                layer_contributes_loss = true;
            }
            // layer is not marked to skip backpropagation
            if !blobs_skip_backp.contains(blob_name) {
                layer_skip_propagate_down = false;
            }
            // layer contributes loss to some
            if layer_contributes_loss && !layer_skip_propagate_down {
                break;
            }
        }

        // If this layer can skip backward computation, also all his bottom blobs
        // don't need backpropagation
        if self.needs_backward && layer_skip_propagate_down {
            self.needs_backward = false;
            for (bottom_id, _) in self.bottom_blobs.iter().enumerate() {
                self.bottom_need_backwards[bottom_id] = false;
            }
        }
        // layer doesn't contribute loss so it does not need to be backpropagated
        if !layer_contributes_loss {
            self.needs_backward = false;
        }
        // if (Caffe::root_solver()) { // Caffe
        {
            info!("{} needs backward computation: {}",
                  self.name,
                  self.needs_backward);
        }

        for (bottom_name, (bottom_id, _)) in self.bottom_blob_names.clone() {
            if layer_contributes_loss {
                blobs_under_loss.insert(bottom_name.clone());
            } else {
                self.bottom_need_backwards[bottom_id] = false;
            }
            if !self.bottom_need_backwards[bottom_id] {
                blobs_skip_backp.insert(bottom_name.clone());
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
        for (bottom_id, _) in self.bottom_need_backwards.clone().iter().enumerate() {
            self.bottom_need_backwards[bottom_id] =
                *self.bottom_need_backwards
                     .get(bottom_id)
                     .unwrap_or(&self.worker.allow_force_backward(bottom_id));
        }
        for (weight_id, _) in self.blobs.clone().iter().enumerate() {
            self.set_weight_propagate_down(weight_id, true);
        }
    }

    /// Uses the underlying layer implementation to compute a forward step.
    ///
    /// See [ILayer.forward](./trait.ILayer.html#method.forward)
    pub fn forward(&mut self) -> f32 {
        self.worker.forward(&self.bottom_blobs, &mut self.top_blobs)
    }

    /// Uses the underlying layer implementation to compute a backward step.
    ///
    /// See [ILayer.backward](./trait.ILayer.html#method.backward)
    pub fn backward(&mut self) {
        if self.needs_backward {
            self.worker.backward(&self.top_blobs, &self.bottom_need_backwards, &mut self.bottom_blobs)
        }
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

    /// Returns the [loss weight][1] associated with the weight blob
    /// with id `weight_id`.
    /// [1]: http://caffe.berkeleyvision.org/tutorial/loss.html
    pub fn loss(&self, weight_id: usize) -> Option<&f32> {
        self.loss.get(weight_id)
    }

    /// Find the name for a supplied blob.
    fn name_for_blob(&self, blob: &ArcLock<HeapBlob>) -> &str {
        // let (res, _) = self.blob_names.iter().find(|&(_, b)| blob == b).unwrap();
        //
        // res
        unimplemented!();
    }
}

/// A Layer in a [Neural Network][1] that can handle forward and backward of a computation step.
/// [1]: ../network/index.html
pub trait ILayer {
    /// Initialize the layer for computation.
    ///
    /// Allows for layer-specific one time setup, e.g. precomputing constant values.
    ///
    /// Is called during [Network][1] initalization
    /// [1]: ../network/type.Network.html
    fn init(&mut self) {}

    /// Compute the [feedforward][1] layer output.
    /// Uses the CPU.
    /// [1]: https://en.wikipedia.org/wiki/Feedforward_neural_network
    fn forward_cpu(&self, bottom: &[ReadBlob], top: &mut Vec<&mut WriteBlob>);
    /// Compute the gradients for the bottom blobs
    /// if the corresponding value of `propagate_down` is true.
    /// Uses the CPU.
    fn backward_cpu(&self, top: &[ReadBlob], propagate_down: &[bool], bottom: &mut Vec<&mut WriteBlob>);

    /// Compute the [feedforward][1] layer output using the currently set computation method.
    /// [1]: https://en.wikipedia.org/wiki/Feedforward_neural_network
    ///
    /// Aquires read locks for the bottom blobs ([ReadBlob][2])
    /// and write locks for the top blobs ([WriteBlob][3]) to ensure sequential computation,
    /// and then passes them to computation method specific function ([forward_cpu][4]).
    ///
    /// [2]: ./type.ReadBlob.html
    /// [3]: ./type.WriteBlob.html
    /// [3]: #method.forward_cpu
    #[allow(map_clone)]
    fn forward(&self, bottom: &[ArcLock<HeapBlob>], top: &mut Vec<ArcLock<HeapBlob>>) -> f32 {
        // Lock();
        // Reshape(bottom, top); // Reshape the layer to fit top & bottom blob
        let mut loss = 0f32;

        let btm: Vec<_> = bottom.iter().map(|b| b.read().unwrap()).collect();
        let tp_ref = top.iter().cloned().collect::<Vec<_>>();
        let mut tp = &mut tp_ref.iter().map(|b| b.write().unwrap()).collect::<Vec<_>>();
        let mut top_w = &mut tp.iter_mut().map(|a| a).collect::<Vec<_>>();
        self.forward_cpu(&btm, top_w);

        for (top_id, top_layer) in top.iter().enumerate() {
            // if (!this->loss(top_id)) { continue; } // Caffe
            // if !self.loss(top_id) { continue; }

            let top_blob = top_layer.read().unwrap();

            let data = top_blob.cpu_data();
            let loss_weights = top_blob.cpu_diff();

            loss += leaf_cpu_dot(data, loss_weights);
        }

        // Unlock();

        loss
    }

    /// Compute the [backpropagation][1] layer output and gradient using the currently set computation method.
    /// [1]: https://en.wikipedia.org/wiki/Backpropagation
    ///
    /// Aquires read locks for the top blobs ([ReadBlob][2])
    /// and write locks for the bottom blobs ([WriteBlob][3]) to ensure sequential computation,
    /// and then passes them to computation method specific function ([backward_cpu][4]).
    ///
    /// [2]: ./type.ReadBlob.html
    /// [3]: ./type.WriteBlob.html
    /// [3]: #method.backward_cpu
    #[allow(map_clone)]
    fn backward(&self, top: &[ArcLock<HeapBlob>], propagate_down: &[bool], bottom: &mut Vec<ArcLock<HeapBlob>>) {
        let tp: Vec<_> = top.iter().map(|b| b.read().unwrap()).collect();
        let bt_ref = bottom.iter().cloned().collect::<Vec<_>>();
        let mut bt = &mut bt_ref.iter().map(|b| b.write().unwrap()).collect::<Vec<_>>();
        let mut btm = &mut bt.iter_mut().map(|a| a).collect::<Vec<_>>();
        self.backward_cpu(&tp, propagate_down, btm);
    }

    /// Return whether "anonymous" top blobs are created automatically for the layer.
    ///
    /// If this method returns true, Network::init will create enough "anonymous" top
    /// blobs to fulfill the requirement specified by [exact_num_top_blobs][1] or
    /// [min_top_blobs][2].
    /// [1]: #method.exact_num_top_blobs
    /// [2]: #method.min_top_blobs
    fn auto_top_blobs(&self) -> bool {
        false
    }
    /// Returns the minimum number of top blobs required by the layer,
    /// or 0 if no minimum number is required.
    ///
    /// This method should be overridden to return a positive value if your
    /// layer expects some minimum number of top blobs.
    fn min_top_blobs(&self) -> usize {
        0
    }
    /// Returns the exact number of top blobs required by the layer,
    /// or 0 if no exact number is required.
    ///
    /// This method should be overridden to return a positive value if your
    /// layer expects some exact number of top blobs.
    fn exact_num_top_blobs(&self) -> usize {
        0
    }
    /// Returns the exact number of bottom blobs required by the layer,
    /// or 0 if no exact number is required.
    ///
    /// This method should be overridden to return a positive value if your
    /// layer expects some exact number of bottom blobs.
    fn exact_num_bottom_blobs(&self) -> usize {
        0
    }
    /// Return whether to allow force_backward for a given bottom blob index.
    ///
    /// If AllowForceBackward(i) == false, we will ignore the force_backward
    /// setting and backpropagate to blob i only if it needs gradient information
    /// (as is done when force_backward == false).
    fn allow_force_backward(&self, bottom_id: usize) -> bool {
        true
    }
}

impl fmt::Debug for ILayer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", "foo", "bar")
    }
}

#[derive(Debug, Clone)]
/// Layer Configuration Struct
pub struct LayerConfig {
    /// The name of the Layer
    pub name: String,

    /// The type of the Layer
    pub layer_type: LayerType,

    /// The name for each top Blob
    pub tops: Vec<String>,

    /// The name for each bottom Blob
    pub bottoms: Vec<String>,

    /// Specifies training configuration for each weight blob.
    pub params: Vec<WeightConfig>,

    /// Specifies on which bottoms the backpropagation should be skipped.
    /// The size must be either 0 or equal to the number of bottoms.
    pub propagate_down: Vec<bool>,
}

#[derive(Debug, Copy, Clone)]
/// The Layer Types
pub enum LayerType {
    /// Sigmoid Layer
    Sigmoid,
}

impl LayerConfig {
    /// Creates a new LayerConfig
    pub fn new(name: String, layer_type: LayerType) -> LayerConfig {
        LayerConfig {
            name: name,
            layer_type: layer_type,

            tops: Vec::new(),
            bottoms: Vec::new(),

            params: Vec::new(),
            propagate_down: Vec::new(),
        }
    }

    /// Returns the Name of the requested top Blob
    pub fn top(&self, top_id: usize) -> Option<&String> {
        self.tops.get(top_id)
    }

    /// Returns the number of top Blobs
    pub fn tops_len(&self) -> usize {
        self.tops.len()
    }

    /// Returns the Name of the requested bottom Blob
    pub fn bottom(&self, bottom_id: usize) -> Option<&String> {
        self.bottoms.get(bottom_id)
    }

    /// Returns the number of bottom Blobs
    pub fn bottoms_len(&self) -> usize {
        self.bottoms.len()
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
        if self.propagate_down.is_empty() || self.propagate_down.len() == self.bottoms.len() {
            Ok(())
        } else {
            Err("propagate_down config must be specified either 0 or bottom_size times")
        }
    }

    // /// Checks if propagate down length is sane
    // pub fn check_propagate_down_len(&self) -> bool {
    //     self.propagate_down.is_empty() || self.propagate_down.len() == self.bottoms.len()
    // }
}


#[derive(Debug, Clone)]
/// Specifies training configuration for a weight blob.
pub struct WeightConfig {
    /// The name of the weight blob -- useful for sharing weights among
    /// layers, but never required otherwise. To share a weight between two
    /// layers, give it a (non-empty) name.
    ///
    /// Default: ""
    pub name: String,
    /// Whether to require shared weights to have the same shape, or just the same
    /// count
    ///
    /// Default: DimCheckMode::Strict
    pub share_mode: DimCheckMode,

    /// The multiplier on the global learning rate for this parameter.
    ///
    /// Default: 1.0f32
    pub lr_mult: Option<f32>,

    /// The multiplier on the global weight decay for this parameter.
    ///
    /// Default: 1.0f32
    pub decay_mult: Option<f32>,
}

impl Default for WeightConfig {
    fn default() -> WeightConfig {
        WeightConfig {
            name: "".to_owned(),
            share_mode: DimCheckMode::Strict,
            lr_mult: None,
            decay_mult: None,
        }
    }
}

impl WeightConfig {
    /// Checks dimensions of two blobs according to the `share_mode`.
    /// Returns an error if there is a count/shape mismatch.
    pub fn check_dimensions<T: Numeric>(&self,
                                        blob_one: &Blob<T>,
                                        blob_two: &Blob<T>,
                                        param_name: String,
                                        owner_name: String,
                                        layer_name: String)
                                        -> Result<(), String> {
        match self.share_mode {
            // Permissive dimension checking -- only check counts are the same.
            DimCheckMode::Permissive => {
                if blob_one.capacity() != blob_two.capacity() {
                    return Err(format!("Cannot share weight '{}' owned by layer '{}' with layer '{}';
                                count mismatch.
                                Owner layer weight shape is {};
                                Sharing layer weight shape is {}",
                                       param_name,
                                       owner_name,
                                       layer_name,
                                       blob_two.shape_string(),
                                       blob_one.shape_string()));
                }
            }
            // Strict dimension checking -- all dims must be the same.
            DimCheckMode::Strict => {
                if blob_one.shape() != blob_two.shape() {
                    return Err(format!("Cannot share weight '{}' owned by layer '{}' with layer '{}';
                                shape mismatch.
                                Owner layer weight shape is {};
                                Sharing layer expects weight shape {}",
                                       param_name,
                                       owner_name,
                                       layer_name,
                                       blob_two.shape_string(),
                                       blob_one.shape_string()));
                }
            }
        }
        Ok(())
    }

    /// The multiplier on the global learning rate for this weight blob.
    pub fn lr_mult(&self) -> f32 {
        match self.lr_mult {
            Some(val) => val,
            None => 1.0f32,
        }
    }

    /// The multiplier on the global weight decay for this weight blob.
    pub fn decay_mult(&self) -> f32 {
        match self.decay_mult {
            Some(val) => val,
            None => 1.0f32,
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// Enum for specifing the shared weights behaviour
pub enum DimCheckMode {
    /// Strict requires that shapes match.
    Strict,
    /// Permissive requires only the count of weights to match.
    Permissive,
}
