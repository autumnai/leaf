use math::*;
use shared_memory::*;
use layers::sigmoid_layer::SigmoidLayer;

pub struct Layer<'a> {
    // pub config: Box<LayerConfig>,
    pub config: Box<&'a LayerConfig>,
    pub worker: Box<ILayer>,

    /// The vector that indicates whether each top blob has a non-zero weight in
    /// the objective function.
    loss: Vec<f32>,

    /// The vector that stores shared references to the parameters in the form of blobs.
    pub blobs: Vec<ArcLock<HeapBlob>>,

    /// Vector indicating whether to compute the diff of each param blob. */
    param_propagate_down: Vec<bool>,
}

impl<'a> Layer<'a> {
    pub fn from_config(config: &'a LayerConfig) -> Layer {
        // let ref cl = config.clone();
        let cl = config.clone();
        let cfg = Box::<&'a LayerConfig>::new(cl);
        Layer {
            loss: Vec::new(),
            // top: Vec::new(),
            blobs: Vec::new(),

            param_propagate_down: Vec::new(),

            worker: Layer::worker_from_config(&cfg),
            config: cfg,
        }
    }

    fn worker_from_config(config: &LayerConfig) -> Box<ILayer> {
        match config.layer_type {
            LayerType::SigmoidLayer => Box::new(SigmoidLayer),
        }
    }

    /// Sets whether the layer should compute gradients w.r.t. a
    /// parameter at a particular index given by param_id.
    pub fn set_param_propagate_down(&mut self, param_id: usize, value: bool) {
        if self.param_propagate_down.len() <= param_id {
            self.param_propagate_down.resize(param_id + 1, true);
        }
        self.param_propagate_down[param_id] = value;

    }

    pub fn loss(&self, id: usize) -> Option<&f32> {
        self.loss.get(id)
    }
}

/// A Layer in a Neural Network that can handle forward and backward of a computation step.
pub trait ILayer {
    /// Compute the layer output.
    /// Uses the CPU.
    fn forward_cpu(&self, bottom: &[HeapBlob], top: &mut Vec<HeapBlob>);
    /// Compute the gradients for the bottom blobs
    /// if the corresponding value of propagate_down is true.
    /// Uses the CPU.
    fn backward_cpu(&self, top: &[HeapBlob], propagate_down: &[bool], bottom: &mut Vec<HeapBlob>);

    /// Compute the layer output using the currently set computation method (CPU).
    fn forward(&self, bottom: &[HeapBlob], top: &mut Vec<HeapBlob>) -> f32 {
        // Lock();
        // Reshape(bottom, top); // Reshape the layer to fit top & bottom blob
        let mut loss = 0f32;

        self.forward_cpu(bottom, top);

        for top_layer in top {
            // if (!this->loss(top_id)) { continue; } // Caffe
            // if !self.loss(top_layer) { continue; }
            // let count = (**top_layer).len();
            let data = top_layer.cpu_data();
            let loss_weights = top_layer.cpu_diff();

            loss += leaf_cpu_dot(data, loss_weights);
            // loss += leaf_cpu_dot(count, data, loss_weights);
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

pub struct LayerConfig {
    pub name: String, // the layer name
    layer_type: LayerType, // the layer type

    tops: Vec<String>, // the name of each top blob
    bottoms: Vec<String>, // the name of each bottom blob

    /// Specifies training parameters (multipliers on global learning constants,
    /// and the name and other settings used for weight sharing).
    params: Vec<ParamConfig>,

    /// Specifies on which bottoms the backpropagation should be skipped.
    /// The size must be either 0 or equal to the number of bottoms.
    pub propagate_down: Vec<bool>,

    // minimal, a lot of Caffe not ported yet
}

pub enum LayerType {
    SigmoidLayer,
}

impl LayerConfig {
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

    pub fn top(&self, top_id: usize) -> Option<&String> {
        self.tops.get(top_id)
    }
    pub fn tops_len(&self) -> usize {
        self.tops.len()
    }
    pub fn bottom(&self, bottom_id: usize) -> Option<&String> {
        self.bottoms.get(bottom_id)
    }
    pub fn bottoms_len(&self) -> usize {
        self.bottoms.len()
    }

    pub fn param(&self, param_id: usize) -> Option<&ParamConfig> {
        self.params.get(param_id)
    }
    pub fn params_len(&self) -> usize {
        self.params.len()
    }

    pub fn check_propagate_down_len(&self) -> bool {
        self.propagate_down.is_empty() || self.propagate_down.len() == self.bottoms.len()
    }
}


/// Specifies training parameters (multipliers on global learning constants,
/// and the name and other settings used for weight sharing).
pub struct ParamConfig {
    /// The names of the parameter blobs -- useful for sharing parameters among
    /// layers, but never required otherwise.  To share a parameter between two
    /// layers, give it a (non-empty) name.
    pub name: String,
    /// Whether to require shared weights to have the same shape, or just the same
    /// count
    pub share_mode: DimCheckMode,

    /// The multiplier on the global learning rate for this parameter.
    // optional float lr_mult = 3 [default = 1.0]; // Caffe
    pub lr_mult: f32,

    /// The multiplier on the global weight decay for this parameter.
    // optional float decay_mult = 4 [default = 1.0]; // Caffe
    pub decay_mult: f32,
}

pub enum DimCheckMode {
    /// Strict requires that shapes match.
    Strict,
    /// Permissive requires only the count of weights to match.
    Permissive,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_layer_config() -> LayerConfig {
        LayerConfig::new("foo".to_owned(), LayerType::SigmoidLayer)
    }

    #[test]
    fn new_layer() {
        let cfg = new_layer_config();
        let layer = Layer::from_config(&cfg);
    }
}
