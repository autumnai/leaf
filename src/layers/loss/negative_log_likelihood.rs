//! TODO: DOC
//!
use co::{IBackend, ITensorDesc, SharedTensor};
use layer::*;
use util::{ArcLock, native_backend};

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// NegativeLogLikelihood Loss Layer
pub struct NegativeLogLikelihood {
    num_classes: usize,
}

impl NegativeLogLikelihood {
    /// Create a NegativeLogLikelihood layer from a NegativeLogLikelihoodConfig.
    pub fn from_config(config: &NegativeLogLikelihoodConfig) -> NegativeLogLikelihood {
        NegativeLogLikelihood {
            num_classes: config.num_classes,
        }
    }

    fn calculate_outer_num(softmax_axis: usize, input_shape: &[usize]) -> usize {
        input_shape.iter().take(softmax_axis + 1).fold(1, |prod, i| prod * i)
    }

    fn calculate_inner_num(softmax_axis: usize, input_shape: &[usize]) -> usize {
        input_shape.iter().skip(softmax_axis + 1).fold(1, |prod, i| prod * i)
    }

    fn batch_size(input_shape: &[usize]) -> usize {
        match input_shape.len() {
            1 => 1,
            2 => input_shape[0],
            _ => panic!("NegativeLogLikelihood layer only supports 1D/2D inputs")
        }
    }
}

impl<B: IBackend> ILayer<B> for NegativeLogLikelihood {
    impl_ilayer_loss!();

    fn sync_native(&self) -> bool {
        true
    }

    fn reshape(&mut self,
               backend: ::std::rc::Rc<B>,
               input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {
        let data = input_data[0].read().unwrap();
        let label = input_data[1].read().unwrap();

        input_gradient[0].write().unwrap().resize(data.desc()).unwrap();
        output_data[0].write().unwrap().resize(label.desc()).unwrap();
    }
}

impl<B: IBackend> ComputeOutput<f32, B> for NegativeLogLikelihood {
    fn compute_output(&self,
                      backend: &B,
                      _weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        let probabilities = input_data[0];
        let labels = input_data[1];

        let batch_size = Self::batch_size(labels.desc());

        let native = native_backend();
        let native_labels = labels.get(native.device()).unwrap().as_native().unwrap().as_slice::<f32>();
        let native_probabilities = probabilities.get(native.device()).unwrap().as_native().unwrap().as_slice::<f32>();

        let mut writable_loss = Vec::<f32>::new();
        for &label_value in native_labels {
            let probability_value = native_probabilities[label_value as usize];
            writable_loss.push(-probability_value);
        }

        let mut loss = writable_loss.iter().fold(0f32, |sum, &val| sum + val);
        loss = loss / (batch_size as f32);
        writable_loss = vec![loss];

        ::util::write_to_memory(output_data[0].get_mut(native.device()).unwrap(), &writable_loss);
    }
}

impl<B: IBackend> ComputeInputGradient<f32, B> for NegativeLogLikelihood {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights_data: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        let labels = input_data[1];
        let batch_size = Self::batch_size(input_data[0].desc());
        let num_classes = self.num_classes;

        let native = native_backend();
        let native_labels = labels.get(native.device()).unwrap().as_native().unwrap().as_slice::<f32>();
        let mut writable_gradient = vec![0f32; input_gradients[0].desc().size()];

        for (batch_n, &label_value) in native_labels.iter().enumerate() {
            let index = (num_classes * batch_n) + label_value as usize;
            writable_gradient[index] = -1f32;
        }
        input_gradients[0].sync(native.device()).unwrap();
        ::util::write_to_memory(input_gradients[0].get_mut(native.device()).unwrap(), &writable_gradient);
    }
}

impl<B: IBackend> ComputeParametersGradient<f32, B> for NegativeLogLikelihood { }

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// Specifies configuration parameters for a NegativeLogLikelihood Layer.
pub struct NegativeLogLikelihoodConfig {
    /// How many different classes can be classified.
    pub num_classes: usize,
}
