//! Provides configuration of weights and their initialization.
use rand;
use rand::distributions::{IndependentSample, Range};
use co::{ITensorDesc, SharedTensor};
use co::plugin::numeric_helpers::Float;
// use shared_memory::*;
use util::native_backend;

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

    /// The filler that initializes the weights in the weight blob.
    ///
    /// Default: None
    pub filler: Option<FillerType>,
}

impl Default for WeightConfig {
    fn default() -> WeightConfig {
        WeightConfig {
            name: "".to_owned(),
            share_mode: DimCheckMode::Strict,
            lr_mult: None,
            decay_mult: None,
            filler: None,
        }
    }
}

impl WeightConfig {
    /// Checks dimensions of two blobs according to the `share_mode`.
    /// Returns an error if there is a count/shape mismatch.
    pub fn check_dimensions<T: Float>(&self,
                                        tensor_one: &SharedTensor<T>,
                                        tensor_two: &SharedTensor<T>,
                                        param_name: String,
                                        owner_name: String,
                                        layer_name: String)
                                        -> Result<(), String> {
        match self.share_mode {
            // Permissive dimension checking -- only check counts are the same.
            DimCheckMode::Permissive => {
                if tensor_one.desc().size() != tensor_two.desc().size() {
                    return Err(format!("Cannot share weight '{}' owned by layer '{}' with layer '{}';
                                count mismatch.
                                Owner layer weight shape is {:?};
                                Sharing layer weight shape is {:?}",
                                       param_name,
                                       owner_name,
                                       layer_name,
                                       tensor_two.desc(),
                                       tensor_one.desc()));
                }
            }
            // Strict dimension checking -- all dims must be the same.
            DimCheckMode::Strict => {
                if tensor_one.desc().size() != tensor_two.desc().size() {
                    return Err(format!("Cannot share weight '{}' owned by layer '{}' with layer '{}';
                                shape mismatch.
                                Owner layer weight shape is {:?};
                                Sharing layer expects weight shape {:?}",
                                       param_name,
                                       owner_name,
                                       layer_name,
                                       tensor_two.desc(),
                                       tensor_one.desc()));
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

#[derive(Debug, Copy, Clone)]
/// Enum for specifing the type of Filler.
pub enum FillerType {
    /// Fills the weight blob with a constant `value` (all values are the same).
    Constant {
        /// The value that will be used to fill the blob.
        value: f32
    },
    /// Fills the weight blobs based on the paper:
    ///
    /// `[Bengio and Glorot 2010]: Understanding the difficulty of training deep feedforward neural networks.`
    ///
    /// Also known as Xavier filler.
    Glorot {
        /// Number of input nodes for each output.
        input_size: usize,
        /// Number of output nodes for each input.
        output_size: usize,
    },
}

impl FillerType {
    /// Uses a filler as specified by this FillerType to fill the values in a SharedTensor
    ///
    /// This filling of weights is usually done directly after creation of the weight blob.
    pub fn fill(&self, weight: &mut SharedTensor<f32>) {
        let native = native_backend();
        let native_device = native.device();
        let actual_device = weight.latest_device().clone();
        // sync to native so we can fill
        match weight.add_device(native_device) { _ => weight.sync(native_device).unwrap() }

        match *self {
            FillerType::Constant { value } => Self::fill_constant(weight, value),
            FillerType::Glorot { input_size, output_size } => Self::fill_glorot(weight, input_size, output_size),
        }

        // sync back to the actual device
        weight.sync(&actual_device).unwrap();
    }

    /// Directly use the [Constant Filler](#variant.Constant).
    pub fn fill_constant(weight: &mut SharedTensor<f32>, value: f32) {
        let native = native_backend();
        let native_weight = weight.get_mut(native.device()).unwrap().as_mut_native().unwrap();

        for e in native_weight.as_mut_slice::<f32>() {
            *e = value;
        }
    }

    /// Directly use the [Glorot Filler](#variant.Glorot).
    pub fn fill_glorot(weight: &mut SharedTensor<f32>, num_inputs: usize, num_outputs: usize) {
        let native = native_backend();
        let native_weight = weight.get_mut(native.device()).unwrap().as_mut_native().unwrap();

        let init_range = (6.0f32 / (num_inputs as f32 + num_outputs as f32)).sqrt();

        let between = Range::new(-init_range, init_range);
        let mut rng = rand::thread_rng();
        for e in native_weight.as_mut_slice::<f32>() {
            *e = between.ind_sample(&mut rng);
        }
    }
}
