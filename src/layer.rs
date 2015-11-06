use math::*;
use phloem::{Blob, Numeric};
use shared_memory::{ArcLock, HeapBlob};
use layers::*;
use std::fmt;

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
pub struct Layer<'a> {
    /// The configuration of the Layer
    pub config: Box<&'a LayerConfig>,
    /// The [implementation][1] of the Layer.
    /// [1]: ../layers/index.html
    ///
    /// This is the part that does most of the work ([forward][2]/[backward][3]).
    /// [2]: ./trait.ILayer.html#method.forward
    /// [3]: ./trait.ILayer.html#method.backward
    pub worker: Box<ILayer>,

    /// The vector that indicates whether each top blob contributes to
    /// the [loss][1] of the network and with which weight.
    /// [1]: http://caffe.berkeleyvision.org/tutorial/loss.html
    loss: Vec<f32>,

    /// The vector that stores shared references to the weights in the form of blobs.
    pub blobs: Vec<ArcLock<HeapBlob>>,

    /// Vector indicating whether to compute the diff of each weight blob.
    ///
    /// You can safely ignore false values and always compute gradients
    /// for all weights, but possibly with wasteful computation.
    ///
    /// Can be used by some [Layer implementations][1] to optimize performance.
    /// [1]: ../layers/index.html
    weight_propagate_down: Vec<bool>,
}

impl<'a> Layer<'a> {
    /// Creates a new Layer from a [LayerConfig][1].
    /// [1]: ./struct.LayerConfig.html
    ///
    /// Used during [Network][2] initalization.
    ///
    /// [2]: ../network/struct.Network.html
    pub fn from_config(config: &'a LayerConfig) -> Layer {
        let cl = config.clone();
        let cfg = Box::<&'a LayerConfig>::new(cl);
        Layer {
            loss: Vec::new(),
            blobs: Vec::new(),

            weight_propagate_down: Vec::new(),

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
}

/// A Layer in a [Neural Network][1] that can handle forward and backward of a computation step.
/// [1]: ../network/index.html
pub trait ILayer {
    /// Compute the [feedforward][1] layer output.
    /// Uses the CPU.
    /// [1]: https://en.wikipedia.org/wiki/Feedforward_neural_network
    fn forward_cpu(&self, bottom: &[ReadBlob], top: &mut Vec<&mut WriteBlob>);
    /// Compute the gradients for the bottom blobs
    /// if the corresponding value of propagate_down is true.
    /// Uses the CPU.
    fn backward_cpu(&self, top: &[HeapBlob], propagate_down: &[bool], bottom: &mut Vec<HeapBlob>);

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
    fn forward(&self, bottom: &[ArcLock<HeapBlob>], top: &mut Vec<ArcLock<HeapBlob>>) -> f32 {
        // Lock();
        // Reshape(bottom, top); // Reshape the layer to fit top & bottom blob
        let mut loss = 0f32;

        let btm: Vec<_> = bottom.iter().map(|b| b.read().unwrap()).collect();
        // let tp: Vec<_> = top.iter().map(|b| b.write().unwrap()).collect();
        let tp_ref = top.iter().map(|t| t.clone()).collect::<Vec<_>>();
        let mut tp = &mut tp_ref.iter().map(|b| b.write().unwrap()).collect::<Vec<_>>();
        let mut tpo = &mut tp.iter_mut().map(|a| a).collect::<Vec<_>>();
        self.forward_cpu(&btm, tpo);
        // self.forward_cpu(bottom, top);

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

    /// Return whether "anonymous" top blobs are created automatically for the layer.
    ///
    /// If this method returns true, Network::init will create enough "anonymous" top
    /// blobs to fulfill the requirement specified by exact_num_top_blobs() or
    /// min_top_blobs().
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

#[derive(Debug)]
/// Layer Configuration Struct
pub struct LayerConfig {
    /// The name of the Layer
    pub name: String,

    /// The type of the Layer
    layer_type: LayerType,

    /// The name for each top Blob
    tops: Vec<String>,

    /// The name for each bottom Blob
    bottoms: Vec<String>,

    /// Specifies training configuration for each weight blob.
    params: Vec<WeightConfig>,

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

    /// Checks if propagate down length is sane
    pub fn check_propagate_down_len(&self) -> bool {
        self.propagate_down.is_empty() || self.propagate_down.len() == self.bottoms.len()
    }
}


#[derive(Debug)]
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
