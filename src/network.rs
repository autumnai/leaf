use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use layer::Layer;
use layer::LayerConfig;
use blob::Blob;

pub struct Network {
    pub name: String,
    layers: Vec<Layer>,

    blobs: Vec<Arc<RwLock<Box<Blob<f32>>>>>, // the blobs storing intermediate results between the layer.
    blob_names: Vec<String>,
    blob_need_backwards: Vec<bool>,

    top_vecs: Vec<Vec<Arc<RwLock<Box<Blob<f32>>>>>>, // stores the vectors containing the output for each layer
    top_id_vecs: Vec<Vec<usize>>, // stores the vectors containing the output for each layer

    input_blobs: Vec<Arc<RwLock<Box<Blob<f32>>>>>,
    input_blob_indices: Vec<usize>,
}

impl Network {
    // fn init(&self, bottom: &Vec<Box<Blob<f32>>>) -> f32 {
    fn init(&self, bottom: &Vec<Box<Blob<f32>>>) {
        // Caffe
        //   for (int input_id = 0; input_id < param.input_size(); ++input_id) {
        //     const int layer_id = -1;  // inputs have fake layer ID -1
        //     AppendTop(param, layer_id, input_id, &available_blobs, &blob_name_to_idx);
        //   }
    }


    fn append_top(&mut self, config: NetworkConfig, layer_id: Option<usize>, top_id: usize, available_blobs: Option<HashSet<&str>>, blob_name_to_idx: Option<&mut HashMap<String, usize>>) {
        let mut layer_config: Option<&LayerConfig> = None;
        if layer_id.is_some() {
            layer_config = config.layer(layer_id.unwrap());
        }

        let blob_name: String;
        match layer_config {
            Some(layer_config) => {
                if layer_config.top(top_id).is_some() { blob_name = String::from(layer_config.top(top_id).unwrap().clone()); }
                else { blob_name = "(automatic)".to_string(); }
            },
            None => {
                blob_name = String::from(config.input(top_id).unwrap().clone());
            }
        }

        if blob_name_to_idx.is_some() && layer_config.is_some()
           && layer_config.unwrap().bottom(top_id).is_some()
           && *layer_config.unwrap().bottom(top_id).unwrap() == blob_name {
            info!("{} -> {} (in-place)", layer_config.unwrap().name, blob_name);
            let idx = blob_name_to_idx.unwrap()[&blob_name];
            let blob = self.blobs[idx].clone();
            self.top_vecs[layer_id.unwrap()].push(blob);
            self.top_id_vecs[layer_id.unwrap()].push(idx);
        }
        else if blob_name_to_idx.is_some() && blob_name_to_idx.as_ref().unwrap().get(&blob_name).is_some() {
            // If we are not doing in-place computation but have duplicated blobs, raise an error.
            error!("Top blob {} produced by multiple sources.", blob_name);
        }
        else {
            // if (Caffe::root_solver()) {
            if true {
                if layer_config.is_some() {
                    info!("{} -> {}", layer_config.unwrap().name, blob_name);
                }
                info!("Input {} -> {}", top_id, blob_name);
            }

            let blob_pointer: Arc<RwLock<Box<Blob<f32>>>> = Arc::new(RwLock::new(Box::new(Blob::new())));
            let blob_id = self.blobs.len();
            self.blobs.push(blob_pointer.clone());
            self.blob_names.push(blob_name.to_string());
            self.blob_need_backwards.push(false);
            if blob_name_to_idx.is_some() { blob_name_to_idx.unwrap().insert(blob_name.to_string(), blob_id); }

            match layer_id {
                None => {
                    // Set the (explicitly specified) dimensions of the input blob.
                    blob_pointer.write().unwrap().reshape(config.input_shape(top_id).unwrap().clone());

                    self.input_blob_indices.push(blob_id);
                    self.input_blobs.push(blob_pointer);
                },
                Some(layer_id) => {
                    self.top_id_vecs[layer_id].push(blob_id);
                    self.top_vecs[layer_id].push(blob_pointer);
                },
            }
        }
        if available_blobs.is_some() { available_blobs.unwrap().insert(&blob_name); }
    }


    pub fn forward_backward(&self, bottom: &Vec<Box<Blob<f32>>>) -> f32 {
        let loss = 0f32; // TODO

        // self.forward(bottom, &loss);
        // self.backward();

        return loss;
    }

    // template <typename Dtype>
    // Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
    //   CHECK_GE(start, 0);
    //   CHECK_LT(end, layers_.size());
    //   Dtype loss = 0;
    //   if (debug_info_) {
    //     for (int i = 0; i < net_input_blobs_.size(); ++i) {
    //       InputDebugInfo(i);
    //     }
    //   }
    //   for (int i = start; i <= end; ++i) {
    //     // LOG(ERROR) << "Forwarding " << layer_names_[i];
    //     Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    //     loss += layer_loss;
    //     if (debug_info_) { ForwardDebugInfo(i); }
    //   }
    //   return loss;
    // }
    pub fn forward_from_to(&self, start: usize, end: usize) -> f32 {
        let mut loss = 0f32;

        // Caffe C++
        //   if (debug_info_) {
        //     for (int i = 0; i < net_input_blobs_.size(); ++i) {
        //       InputDebugInfo(i);
        //     }
        //   }

        for i in start..end {
            // loss += self.layers[i].forward(self.bottom_vecs[i], self.top_vecs[i]);
        }

        return loss;
    }


    //
    // template <typename Dtype>
    // Dtype Net<Dtype>::ForwardFrom(int start) {
    //   return ForwardFromTo(start, layers_.size() - 1);
    // }
    //
    // template <typename Dtype>
    // Dtype Net<Dtype>::ForwardTo(int end) {
    //   return ForwardFromTo(0, end);
    // }

    // pub fn forward_prefilled(&self, loss: Option<f32>) -> &Vec<Box<Blob<f32>>> {
    pub fn forward_prefilled(&self, loss: Option<f32>) {
        match loss {
            Some(loss_result) => {
                // not sure if loss_result will really be changed
                // loss_result = self.forward_from_to(0, self.layers.len() - 1);
            }
            None => {
                self.forward_from_to(0, self.layers.len() - 1);
            }
        }

        // return net_output_blobs_;
        // return self.layers; // WRONG
    }


// const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
//     const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
//   // Copy bottom to internal bottom
//   for (int i = 0; i < bottom.size(); ++i) {
//     net_input_blobs_[i]->CopyFrom(*bottom[i]);
//   }
//   return ForwardPrefilled(loss);
// }

// pub fn forward(&self, bottom: &Vec<Box<Blob<f32>>>, loss: &f32) ->
// &Vec<Box<Blob<f32>>> {
//     // let blob: Blob<f32> = Blob::new();
//     let blob = vec![Box::new(Blob::new())];
//
//
//     return &blob;
// }

}

pub struct NetworkConfig {
    pub name: String,
    inputs: Vec<String>, // The input blobs to the network.
    input_shapes: Vec<Vec<isize>>, // The shape of the input blobs.

    // Whether the network will force every layer to carry out backward operation.
    // If set False, then whether to carry out backward is determined
    // automatically according to the net structure and learning rates.
    force_backward: bool,

    // // The current "state" of the network, including the phase, level, and stage.
    // // Some layers may be included/excluded depending on this state and the states
    // // specified in the layers' include and exclude fields.
    // optional NetState state = 6;
    debug_info: bool, // Print debugging information about results

    layers: Vec<LayerConfig>, // The layers that make up the net.
}

impl NetworkConfig {
    pub fn layer(&self, layer_id: usize) -> Option<&LayerConfig> { return self.layers.get(layer_id); }

    pub fn input(&self, input_id: usize) -> Option<&String> { return self.inputs.get(input_id); }
    pub fn input_shape(&self, input_id: usize) -> Option<&Vec<isize>> { return self.input_shapes.get(input_id); }
}
