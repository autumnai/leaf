//! A container layer that runs operations sequentially on the contained layers.
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use co::{IBackend, SharedTensor};
use layer::*;
use util::{ArcLock, LayerOps};
use leaf_capnp::sequential_config as capnp_config;
use leaf_capnp::shaped_input as capnp_shaped_input;
use capnp_util::*;

#[derive(Debug)] /// Sequential Layer
pub struct Sequential<B: IBackend + LayerOps<f32>> {
    layers: Vec<RefCell<Layer<B>>>,

    input_tensor_names: Vec<String>,
    input_data_tensors: Vec<ArcLock<SharedTensor<f32>>>,
    input_gradient_tensors: Vec<ArcLock<SharedTensor<f32>>>,

    output_data_tensors: Vec<ArcLock<SharedTensor<f32>>>,
    output_gradient_tensors: Vec<ArcLock<SharedTensor<f32>>>,

    registry: HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>)>,
}

impl<B: IBackend + LayerOps<f32> + 'static> Sequential<B> {
    /// Create a empty Sequential container layer.
    pub fn empty() -> Sequential<B> {
        Sequential {
            layers: vec![],

            input_tensor_names: vec![],
            input_data_tensors: vec![],
            input_gradient_tensors: vec![],

            output_data_tensors: vec![],
            output_gradient_tensors: vec![],

            registry: HashMap::new(),
        }
    }

    /// Create a Sequential layer from a SequentialConfig.
    pub fn from_config(backend: Rc<B>, config: &SequentialConfig) -> Sequential<B> {
        let mut layer = Self::empty();

        layer.init_layers(backend, &config.clone());

        layer
    }

    /// Initializes a sequential container.
    ///
    /// Sets up the structure of the sequential container. It reads the supplied [SequentialConfig][1],
    /// connects the input and output blobs of each layer and determines if the backpropagation has
    /// to be executed for each tensor and layer.
    ///
    /// [1]: ./struct.SequentialConfig.html
    pub fn init_layers(&mut self, backend: Rc<B>, in_config: &SequentialConfig) {
        let mut config = in_config.clone();
        let mut registry = HashMap::<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>)>::new();
        let weight_registry = &mut HashMap::<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>, Option<f32>, Option<f32>)>::new();

        for (input_name, input_shape) in config.inputs.clone() {
            self.init_input_blob(backend.clone(), &input_name, &input_shape, &mut registry);
        }

        // add input names to first layer so they correctly connect
        if let Some(first_layer) = config.layers.first_mut() {
            for container_input in &self.input_tensor_names {
                first_layer.add_input(&container_input);
            }
        }
        // connect each layer to the next one
        for (i, _) in config.layers.clone().iter().enumerate() {
            match i == (config.layers.len() - 1) {
                false => {
                    // layers have already been manually connected
                    if config.layers[i].outputs.get(0).is_some() && config.layers[i + 1].inputs.get(0).is_some() &&
                       config.layers[i].outputs.get(0) == config.layers[i + 1].inputs.get(0) {
                        continue;
                    }
                    if let Some(in_place) = config.find_in_place_output(i) {
                        config.layers[i].add_output(&in_place);
                        config.layers[i + 1].add_input(&in_place);
                    } else {
                        config.layers[i].add_output(&format!("SEQUENTIAL_{}", i));
                        config.layers[i + 1].add_input(&format!("SEQUENTIAL_{}", i));
                    }
                },
                // last layer
                true => {
                    config.layers[i].add_output(&format!("SEQUENTIAL_OUTPUT_{}", i));
                },
            }
        }

        let mut shared_workspace = None;
        for layer_config in &config.layers {
            self.init_layer(backend.clone(), &layer_config, &mut registry, weight_registry);
            shared_workspace = self.resize_shared_workspace(backend.clone(), shared_workspace);
        }

        // Go through the net backwards to determine which blobs contribute to the
        // loss.  We can skip backward computation for blobs that don't contribute
        // to the loss.
        // Also checks if all bottom blobs don't need backward computation (possible
        // because the skip_propagate_down config) and so we can skip backward
        // computation for the entire layer
        let blobs_under_loss = &mut HashSet::<String>::new();
        let blobs_skip_backp = &mut HashSet::<String>::new();
        for layer in &mut self.layers.iter_mut().rev() {
            layer.borrow_mut().init_backprop( blobs_under_loss, blobs_skip_backp);
        }

        if config.force_backward {
            for layer in &mut self.layers {
                layer.borrow_mut().init_force_backward();
            }
        }

        // Outputs of the last layer are considered output of the container
        if let Some(last_layer) = self.layers.last() {
            for data_tensor in &last_layer.borrow().output_blobs_data {
                self.output_data_tensors.push(data_tensor.clone());
            }
            for gradient_tensor in &last_layer.borrow().output_blobs_gradient {
                self.output_gradient_tensors.push(gradient_tensor.clone());
            }
        }

        self.registry = registry;

        info!("Sequential container initialization done.");
    }

    /// Initialize a input tensor for the Sequential container.
    ///
    /// Appends a input blob to the network, so the first [Layer][1] can
    /// [connect][2] to them.
    ///
    /// Used during initialization of the Sequential container.
    /// [1]: ../layer/struct.Layer.html
    /// [2]: ../layer/struct.Layer.html#method.connect
    fn init_input_blob(&mut self,
                  backend: Rc<B>,
                  tensor_name: &str,
                  input_shape: &[usize],
                  registry: &mut HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>)> ) {

        if registry.contains_key(tensor_name) {
            // If we are not doing in-place computation but see two layers trying
            // to produce the same tensor, raise an error.
            error!("Output tensor {} produced by multiple sources.", tensor_name);
            return
        } else {
            info!("Input {} -> {}", self.input_data_tensors.len(), tensor_name);

            let ibackend: Rc<IBackend<F=B::F>> = backend;
            let data_tensor: ArcLock<SharedTensor<f32>> = Arc::new(RwLock::new(SharedTensor::new(ibackend.device(), &input_shape).unwrap()));
            let gradient_tensor: ArcLock<SharedTensor<f32>> = Arc::new(RwLock::new(SharedTensor::new(ibackend.device(), &input_shape).unwrap()));

            self.input_data_tensors.push(data_tensor.clone());
            self.input_gradient_tensors.push(gradient_tensor.clone());
            self.input_tensor_names.push(tensor_name.to_owned());
            registry.insert(tensor_name.to_owned(), (data_tensor, gradient_tensor));
        }
    }

    /// Initializes a single layer of the Sequential container.
    ///
    /// Appends input and output tensors to the [Layer][3]. Apart from explicitly named
    /// output tensors it will also append anonymous output tensors that are required by the specific
    /// [Layer implemenations][4]. It also sets up the backpropagation flags.
    ///
    /// [3]: ../layer/struct.Layer.html
    /// [4]: ../layers/index.html
    fn init_layer(&mut self,
                  backend: Rc<B>,
                  layer_config: &LayerConfig,
                  registry: &mut HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>)>,
                  weight_registry: &mut HashMap<String, (ArcLock<SharedTensor<f32>>, ArcLock<SharedTensor<f32>>, Option<f32>, Option<f32>)>) {
        // Setup layer.
        if let Err(e) = layer_config.validate() {
            error!("{}", e);
        }

        info!("Creating Layer {}", &layer_config.name);
        let mut layer = Layer::from_config(backend, &layer_config);

        // Figure out this layer's input and output
        layer.connect(registry, weight_registry);

        self.layers.push(RefCell::new(layer));
    }
}

impl<B: IBackend + LayerOps<f32> + 'static> ILayer<B> for Sequential<B> {
    fn is_container(&self) -> bool {
        true
    }

    fn inputs_data(&self) -> Option<Vec<ArcLock<SharedTensor<f32>>>> {
        Some(self.input_data_tensors.clone())
    }

    fn inputs_gradients(&self) -> Option<Vec<ArcLock<SharedTensor<f32>>>> {
        Some(self.input_gradient_tensors.clone())
    }

    fn outputs_data(&self) -> Option<Vec<ArcLock<SharedTensor<f32>>>> {
        Some(self.output_data_tensors.clone())
    }

    fn outputs_gradients(&self) -> Option<Vec<ArcLock<SharedTensor<f32>>>> {
        Some(self.output_gradient_tensors.clone())
    }

    fn learnable_weights(&self) -> Option<Vec<ArcLock<SharedTensor<f32>>>> {
        let weights = self.layers.iter().flat_map(|layer| layer.borrow().learnable_weights_data()).collect();
        Some(weights)
    }

    fn learnable_weights_gradients(&self) -> Option<Vec<ArcLock<SharedTensor<f32>>>> {
        let gradients = self.layers.iter().flat_map(|layer| layer.borrow().learnable_weights_gradients()).collect();
        Some(gradients)
    }

    fn learnable_weights_names(&self) -> Option<Vec<String>> {
        let names = self.layers.iter().flat_map(|layer| layer.borrow().learnable_weights_names()).collect();
        Some(names)
    }

    fn resize_shared_workspace(&mut self, backend: Rc<B>, workspace: Option<ArcLock<SharedTensor<u8>>>) -> Option<ArcLock<SharedTensor<u8>>> {
        debug!("Resizing shared workspace {:?}", workspace.is_some());
        let mut shared_workspace = workspace;

        for layer in &self.layers {
            shared_workspace = layer.borrow_mut().worker.resize_shared_workspace(backend.clone(), shared_workspace);
        }

        shared_workspace
    }

    fn forward(&self,
               backend: &B,
               input_data: &[ArcLock<SharedTensor<f32>>],
               weights_data: &[ArcLock<SharedTensor<f32>>],
               output_data: &mut [ArcLock<SharedTensor<f32>>]) {
        for layer in &self.layers {
            for (i, (input, input_name)) in input_data.iter().zip(self.input_tensor_names.iter()).enumerate() {
                if &layer.borrow().input_blob_names[i] == input_name {
                    layer.borrow_mut().input_blobs_data[i] = input.clone();
                }
            }
            layer.borrow_mut().forward(&[]);
        }
        if let Some(last_layer) = self.layers.last() {
            last_layer.borrow_mut().synchronize();
        }
    }

    fn backward_input(&self,
                backend: &B,
                weights_data: &[ArcLock<SharedTensor<f32>>],
                output_data: &[ArcLock<SharedTensor<f32>>],
                output_gradients: &[ArcLock<SharedTensor<f32>>],
                input_data: &[ArcLock<SharedTensor<f32>>],
                input_gradients: &mut [ArcLock<SharedTensor<f32>>]) {
        if let Some(last_layer) = self.layers.last() {
            for (i, output_gradient) in output_gradients.iter().enumerate() {
                last_layer.borrow_mut().output_blobs_gradient[i] = output_gradient.clone();
            }
        }
        for layer in self.layers.iter().rev() {
            layer.borrow_mut().backward_input(&[]);
        }
        if let Some(first_layer) = self.layers.iter().rev().last() {
            first_layer.borrow_mut().synchronize();
        }
    }

    fn backward_parameters(&self,
                backend: &B,
                output_data: &[ArcLock<SharedTensor<f32>>],
                output_gradients: &[ArcLock<SharedTensor<f32>>],
                input_data: &[ArcLock<SharedTensor<f32>>],
                weights_gradients: &mut [ArcLock<SharedTensor<f32>>]) {
        for layer in self.layers.iter().rev() {
            layer.borrow_mut().backward_parameters();
        }
        if let Some(first_layer) = self.layers.iter().rev().last() {
            first_layer.borrow_mut().synchronize();
        }
    }
}

impl<B: IBackend + LayerOps<f32> + 'static> ComputeOutput<f32, B> for Sequential<B> {
    // we are overriding `forward` and not calling `compute_output`
    fn compute_output(&self,
                      backend: &B,
                      weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) { }
}

impl<B: IBackend + LayerOps<f32> + 'static> ComputeInputGradient<f32, B> for Sequential<B> {
    // we are overriding `backward_input` and not calling `compute_input_gradient`
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights_data: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) { }
}

impl<B: IBackend + LayerOps<f32> + 'static> ComputeParametersGradient<f32, B> for Sequential<B> {
    // we are overriding `backward_parameters` and not calling `compute_parameters_gradient`
    fn compute_parameters_gradient(&self,
                                   backend: &B,
                                   output_data: &[&SharedTensor<f32>],
                                   output_gradients: &[&SharedTensor<f32>],
                                   input_data: &[&SharedTensor<f32>],
                                   parameters_gradients: &mut [&mut SharedTensor<f32>]) { }
}

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// Specifies configuration parameters for a Sequential Layer.
pub struct SequentialConfig {
    /// Defines the layers of the container via [LayerConfig][layer_config]s.
    ///
    /// [layer_config]: ../../../layer/struct.LayerConfig.html
    pub layers: Vec<LayerConfig>,

    /// Defines the names and shapes of the input tensors.
    ///
    /// The inputs are identified by name so they can be referenced as input tensors
    /// in a [LayerConfig][layer_config].
    ///
    /// [layer_config]: ../../../layer/struct.LayerConfig.html
    pub inputs: Vec<(String, Vec<usize>)>,

    /// Defines if the container will force every layer to do [backpropagation][1].
    /// [1]: https://en.wikipedia.org/wiki/Backpropagation
    ///
    /// If set to `false`, then the execution of backpropagation is determined automatically
    /// according to the network structure and learning rates.
    ///
    /// Default: `false`
    pub force_backward: bool,
}

impl SequentialConfig {
    /// Tries to find the output of a previous layer that is usable as in-place output for the n-th layer.
    pub fn find_in_place_output(&self, n: usize) -> Option<String> {
        if let Some(layer) = self.layers.get(n) {
            if layer.layer_type.supports_in_place() {
                // look through all previous layers until we find the first one that is not doing in-place.
                for prev_layer in self.layers.iter().take(n).collect::<Vec<_>>().iter().rev() {
                    if !prev_layer.layer_type.supports_in_place() {
                        if let Some(output_name) = prev_layer.outputs.get(0) {
                            return Some(output_name.to_owned())
                        }
                    }
                }
                // use input if there are no previous layers to use
                if let Some(input) = self.inputs.get(0) {
                    return Some(input.0.to_owned())
                }
            }
        }

        None
    }

    /// Add layer at the end of the sequential container.
    pub fn add_layer(&mut self, layer: LayerConfig) {
        self.layers.push(layer);
    }

    /// Add a input to the network.
    pub fn add_input(&mut self, input_name: &str, shape: &[usize]) {
        self.inputs.push((input_name.to_owned(), shape.to_owned()));
    }

    /// Write a input into a capnp message.
    fn write_capnp_shaped_input(&self, builder: &mut capnp_shaped_input::Builder, i: usize) {
        let input = self.inputs.get(i).unwrap();
        let ref name = input.0;
        let ref shape = input.1;
        builder.set_name(name);
        let mut dimensions = builder.borrow().init_shape(shape.len() as u32);
        for (i, dim) in shape.iter().enumerate() {
            dimensions.set(i as u32, *dim as u64);
        }
    }
}

impl<'a> CapnpWrite<'a> for SequentialConfig {
    type Builder = capnp_config::Builder<'a>;

    /// Write the SequentialConfig into a capnp message.
    fn write_capnp(&self, builder: &mut Self::Builder) {
        {
            let mut layers = builder.borrow().init_layers(self.layers.len() as u32);
            for (i, layer) in self.layers.iter().enumerate() {
                let mut layer_config = layers.borrow().get(i as u32);
                layer.write_capnp(&mut layer_config);
            }
        }
        {
            let mut inputs = builder.borrow().init_inputs(self.inputs.len() as u32);
            for (i, _) in self.inputs.iter().enumerate() {
                let mut shaped_input = inputs.borrow().get(i as u32);
                self.write_capnp_shaped_input(&mut shaped_input, i);
            }
        }
        builder.set_force_backward(self.force_backward);
    }
}

impl Into<LayerType> for SequentialConfig {
    fn into(self) -> LayerType {
        LayerType::Sequential(self)
    }
}

impl ::std::default::Default for SequentialConfig {
    fn default() -> SequentialConfig {
        SequentialConfig {
            layers: vec![],
            inputs: vec![],
            force_backward: false,
        }
    }
}
