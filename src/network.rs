use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::cmp;
use layer::Layer;
use layer::LayerConfig;
use blob::Blob;

pub struct Network {
    pub name: String,
    layers: Vec<Layer>,
    layer_names: Vec<String>,

    blobs: Vec<Arc<RwLock<Box<Blob<f32>>>>>, // the blobs storing intermediate results between the layer.
    blob_names: Vec<String>,
    blob_need_backwards: Vec<bool>,

    // stores the vectors containing the output for each layer (only references to the blobs)
    top_vecs: Vec<Vec<Arc<RwLock<Box<Blob<f32>>>>>>,
    top_id_vecs: Vec<Vec<usize>>,

    bottom_vecs: Vec<Vec<Arc<RwLock<Box<Blob<f32>>>>>>, // stores the vectors containing the input for each layer
    bottom_id_vecs: Vec<Vec<usize>>,
    bottom_need_backwards: Vec<Vec<bool>>,

    input_blobs: Vec<Arc<RwLock<Box<Blob<f32>>>>>,
    input_blob_indices: Vec<usize>,

    // Vector of weight in the loss (or objective) function of each net blob, indexed by blob_id.
    blob_loss_weights: Vec<f32>,

    param_id_vecs: Vec<Vec<usize>>,
}

impl Network {
    // fn init(&self, bottom: &Vec<Box<Blob<f32>>>) -> f32 {

    fn init(&mut self, in_param: &NetworkConfig) {

        let param = in_param.clone();
        let available_blobs = &mut HashSet::new();
        let blob_name_to_idx = &mut HashMap::<String, usize>::new();
        for (input_id, _) in param.inputs.iter().enumerate() {
            self.append_top(param,
                            None,
                            input_id,
                            Some(available_blobs),
                            Some(blob_name_to_idx));
        }

        self.resize_vecs(param.layers.len());

        for (layer_id, _) in param.inputs.iter().enumerate() {
            // Caffe
            // bool share_from_root = !Caffe::root_solver()
            //     && root_net_->layers_[layer_id]->ShareInParallel();
            // // Inherit phase from net if unset.
            // if (!param.layer(layer_id).has_phase()) {
            //   param.mutable_layer(layer_id)->set_phase(phase_);
            // }

            // Setup layer.
            let layer_config = param.layer(layer_id).unwrap(); // TODO: should be safer
            if !layer_config.check_propagate_down_len() {
                // TODO: move layer validation to layer
                error!("propagate_down param must be specified either 0 or bottom_size times")
            }

            // Caffe
            // if (share_from_root) {
            //   LOG(INFO) << "Sharing layer " << layer_param.name() << " from root net";
            //   layers_.push_back(root_net_->layers_[layer_id]);
            //   layers_[layer_id]->SetShared(true);
            // else {
            {
                // TODO
                // layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
            }
            self.layer_names.push(layer_config.name.clone());
            info!("Creating Layer {}", layer_config.name.clone());
            let need_backward = false;

            // Figure out this layer's input and output

            for bottom_id in 0..(layer_config.bottoms_len() - 1) {
                // TODO
                // const int blob_id = AppendBottom(param, layer_id, bottom_id,
                //                                &available_blobs, &blob_name_to_idx);
                // // If a blob needs backward, this layer should provide it.
                // need_backward |= blob_need_backward_[blob_id];
            }
            let num_top = layer_config.tops_len();
            for top_id in 0..(num_top - 1) {
                self.append_top(param,
                                Some(layer_id),
                                top_id,
                                Some(available_blobs),
                                Some(blob_name_to_idx))
            }

            // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
            // specified fewer than the required number (as specified by
            // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
            let auto_top_blobs = self.layers.get(layer_id).unwrap().auto_top_blobs();
            let min_top_blobs = self.layers.get(layer_id).unwrap().min_top_blobs();
            let exact_num_top_blobs = self.layers.get(layer_id).unwrap().exact_num_top_blobs();
            if auto_top_blobs {
                let needed_num_top = cmp::max(min_top_blobs, exact_num_top_blobs);
                for _ in 0..(needed_num_top - num_top) {
                    // Add "anonymous" top blobs -- do not modify available_blobs or
                    // blob_name_to_idx as we don't want these blobs to be usable as input
                    // to other layers.
                    info!("Adding anonymous top blob");
                    self.append_top(param, Some(layer_id), num_top, None, None);
                }
            }


            // After this layer is connected, set it up.
            // Caffe
            // if (share_from_root) {
            //   // Set up size of top blobs using root_net_
            //   const vector<Blob<Dtype>*>& base_top = root_net_->top_vecs_[layer_id];
            //   const vector<Blob<Dtype>*>& this_top = this->top_vecs_[layer_id];
            //   for (int top_id = 0; top_id < base_top.size(); ++top_id) {
            //     this_top[top_id]->ReshapeLike(*base_top[top_id]);
            //     LOG(INFO) << "Created top blob " << top_id << " (shape: "
            //         << this_top[top_id]->shape_string() <<  ") for shared layer "
            //         << layer_param.name();
            //   }
            // } else {
            {
                // layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
                // TODO
                // self.layers[layer_id].set_up(self.bottom_vecs[layer_id],
                // self.top_vecs[layer_id]);
            }

            info!("Setting up {}", self.layer_names[layer_id]);
            let layer = self.layers.get(layer_id).unwrap(); // TODO: should be safer?
            // let layer = (&self.layers[layer_id]).clone(); // TODO: should be safer?
            for top_id in 0..(self.top_vecs[layer_id].len() - 1) {
                if self.blob_loss_weights.len() <= self.top_id_vecs[layer_id][top_id] {
                    self.blob_loss_weights.resize(self.top_id_vecs[layer_id][top_id] + 1, 0f32);
                }
                // self.blob_loss_weights[self.top_id_vecs[layer_id][top_id]] =
                // layer.loss(top_id);
            }

            // for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
            //   if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
            //     blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
            //   }
            //   blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
            //   LOG_IF(INFO, Caffe::root_solver())
            //       << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
            //   if (layer->loss(top_id)) {
            //     LOG_IF(INFO, Caffe::root_solver())
            //         << "    with loss weight " << layer->loss(top_id);
            //   }
            //   memory_used_ += top_vecs_[layer_id][top_id]->count();
            // }
        }
    }

    fn resize_vecs(&mut self, new_len: usize) {
        self.bottom_vecs.resize(new_len, vec![Arc::new(RwLock::new(Box::new(Blob::new())))]);
        self.top_vecs.resize(new_len, vec![Arc::new(RwLock::new(Box::new(Blob::new())))]);
        self.bottom_id_vecs.resize(new_len, vec![0]);
        self.top_id_vecs.resize(new_len, vec![0]);
        self.param_id_vecs.resize(new_len, vec![0]);
        self.bottom_need_backwards.resize(new_len, vec![false]);
    }


    fn append_top(&mut self,
                  config: &NetworkConfig,
                  layer_id: Option<usize>,
                  top_id: usize,
                  available_blobs: Option<&mut HashSet<String>>,
                  blob_name_to_idx: Option<&mut HashMap<String, usize>>) {
        let mut layer_config: Option<&LayerConfig> = None;
        if layer_id.is_some() {
            layer_config = config.layer(layer_id.unwrap());
        }

        let blob_name: String;
        match layer_config {
            Some(layer_config) => {
                if layer_config.top(top_id).is_some() {
                    blob_name = String::from(layer_config.top(top_id).unwrap().clone());
                } else {
                    blob_name = "(automatic)".to_string();
                }
            }
            None => {
                blob_name = String::from(config.input(top_id).unwrap().clone());
            }
        }

        if blob_name_to_idx.is_some() && layer_config.is_some() && layer_config.unwrap().bottom(top_id).is_some() &&
           *layer_config.unwrap().bottom(top_id).unwrap() == blob_name {
            info!("{} -> {} (in-place)", layer_config.unwrap().name, blob_name);
            let idx = blob_name_to_idx.unwrap()[&blob_name];
            let blob = self.blobs[idx].clone();
            self.top_vecs[layer_id.unwrap()].push(blob);
            self.top_id_vecs[layer_id.unwrap()].push(idx);
        } else if blob_name_to_idx.is_some() && blob_name_to_idx.as_ref().unwrap().get(&blob_name).is_some() {
            // If we are not doing in-place computation but have duplicated blobs, raise an
            // error.
            error!("Top blob {} produced by multiple sources.", blob_name);
        } else {
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
            if blob_name_to_idx.is_some() {
                blob_name_to_idx.unwrap().insert(blob_name.to_string(), blob_id);
            }

            match layer_id {
                None => {
                    // Set the (explicitly specified) dimensions of the input blob.
                    blob_pointer.write().unwrap().reshape(config.input_shape(top_id).unwrap().clone());

                    self.input_blob_indices.push(blob_id);
                    self.input_blobs.push(blob_pointer);
                }
                Some(layer_id) => {
                    self.top_id_vecs[layer_id].push(blob_id);
                    self.top_vecs[layer_id].push(blob_pointer);
                }
            }
        }
        if available_blobs.is_some() {
            available_blobs.unwrap().insert(blob_name.to_string());
        }
    }


    pub fn forward_backward(&self, bottom: &Vec<Box<Blob<f32>>>) -> f32 {
        let loss = 0f32; // TODO

        // self.forward(bottom, &loss);
        // self.backward();

        loss
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

        loss
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
    pub fn layer(&self, layer_id: usize) -> Option<&LayerConfig> {
        self.layers.get(layer_id)
    }

    pub fn input(&self, input_id: usize) -> Option<&String> {
        self.inputs.get(input_id)
    }
    pub fn input_shape(&self, input_id: usize) -> Option<&Vec<isize>> {
        self.input_shapes.get(input_id)
    }
}
