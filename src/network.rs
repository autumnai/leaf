use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::cmp;
use shared_memory::*;
use layer::{Layer, ILayer};
use layer::{LayerConfig, ParamConfig};
use phloem::Blob;

#[derive(Debug)]
/// The Network
pub struct Network<'a> {
    /// The name of the `Network`
    pub name: String,
    layers: Vec<Layer<'a>>,
    layer_names: Vec<String>,
    layer_names_index: HashMap<String, usize>,
    layer_need_backwards: Vec<bool>,

    blobs: Vec<ArcLock<HeapBlob>>, // the blobs storing intermediate results between the layer.
    blob_names: Vec<String>,
    blob_names_index: HashMap<String, usize>,
    blob_need_backwards: Vec<bool>,

    output_blobs: Vec<ArcLock<HeapBlob>>,
    output_blob_indices: Vec<usize>,

    // stores the vectors containing the output for each layer (only references to the blobs)
    top_vecs: Vec<Vec<ArcLock<HeapBlob>>>,
    top_id_vecs: Vec<Vec<usize>>,

    bottom_vecs: Vec<Vec<ArcLock<HeapBlob>>>, // stores the vectors containing the input for each layer
    bottom_id_vecs: Vec<Vec<usize>>,
    bottom_need_backwards: Vec<Vec<bool>>,

    input_blobs: Vec<ArcLock<HeapBlob>>,
    input_blob_indices: Vec<usize>,

    // Vector of weight in the loss (or objective) function of each net blob, indexed by blob_id.
    blob_loss_weights: Vec<f32>,

    param_id_vecs: Vec<Vec<usize>>,
    param_owners: Vec<Option<usize>>,
    param_display_names: Vec<String>,
    param_layer_indices: Vec<(usize, usize)>,
    param_names_index: HashMap<String, usize>,

    /// The parameters in the network.
    params: Vec<ArcLock<HeapBlob>>,
    learnable_params: Vec<ArcLock<HeapBlob>>,
    learnable_param_ids: Vec<usize>,

    params_lr: Vec<Option<f32>>,
    params_weight_decay: Vec<Option<f32>>,
}

impl<'a> Network<'a> {
    fn init(&mut self, in_param: &'a NetworkConfig) {
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
            self.init_layer(layer_id, param, available_blobs, blob_name_to_idx);
        }

        // Go through the net backwards to determine which blobs contribute to the
        // loss.  We can skip backward computation for blobs that don't contribute
        // to the loss.
        // Also checks if all bottom blobs don't need backward computation (possible
        // because the skip_propagate_down param) and so we can skip bacward
        // computation for the entire layer
        let blobs_under_loss = &mut HashSet::<String>::new();
        let blobs_skip_backp = &mut HashSet::<String>::new();
        // get mutable references to struct fields because Rust doesn't support
        // partially borrowed structs
        let layer_need_backwards = &mut self.layer_need_backwards.clone();
        let bottom_need_backwards = &mut self.bottom_need_backwards.clone();
        for (layer_id, _) in self.layers.iter().rev().enumerate() {
            self.init_backprop(layer_id,
                               layer_need_backwards,
                               bottom_need_backwards,
                               blobs_under_loss,
                               blobs_skip_backp);
        }

        if param.force_backward {
            self.init_force_backward();
        }

        // In the end, all remaining blobs are considered output blobs.
        for available_blob in available_blobs.iter() {
            info!("This network produces output {}", available_blob);
            let id = blob_name_to_idx[available_blob];
            self.output_blobs.push(self.blobs[id].clone());
            self.output_blob_indices.push(id);
        }

        // setup names->idx
        for (blob_id, blob_name) in self.blob_names.iter().enumerate() {
            self.blob_names_index.insert(blob_name.clone(), blob_id);
        }
        for (layer_id, layer_name) in self.layer_names.iter().enumerate() {
            self.layer_names_index.insert(layer_name.clone(), layer_id);
        }

        self.share_weights();

        info!("Network initialization done.");
        unimplemented!();
    }

    fn init_layer(&mut self,
                  layer_id: usize,
                  param: &'a NetworkConfig,
                  available_blobs: &mut HashSet<String>,
                  blob_name_to_idx: &mut HashMap<String, usize>) {
        // Caffe
        // bool share_from_root = !Caffe::root_solver()
        //     && root_net_->layers_[layer_id]->ShareInParallel();
        // // Inherit phase from net if unset.
        // if (!param.layer(layer_id).has_phase()) {
        //   param.mutable_layer(layer_id)->set_phase(phase_);
        // }

        // Setup layer.
        // let layer_config = Box::new(param.layer(layer_id).take()); // TODO: should
        // be safer
        // let layer_config = param.layer(layer_id).unwrap(); // TODO: should be safer
        // let layer_config = param.layer(layer_id).unwrap(); // TODO: should be safer
        let layer_config = (&param.layers[layer_id]).clone(); // TODO: should be safer
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
            self.layers.push(Layer::from_config(&layer_config));
        }
        self.layer_names.push(layer_config.name.clone());
        info!("Creating Layer {}", layer_config.name.clone());
        let mut need_backward = false;

        // Figure out this layer's input and output

        for bottom_id in 0..(layer_config.bottoms_len() - 1) {
            let blob_id = self.append_bottom(param,
                                             layer_id,
                                             bottom_id,
                                             available_blobs,
                                             blob_name_to_idx);

            // If a blob needs backward, this layer should provide it.
            need_backward |= self.blob_need_backwards[blob_id];
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
        let auto_top_blobs = self.layers.get(layer_id).unwrap().worker.auto_top_blobs();
        let min_top_blobs = self.layers.get(layer_id).unwrap().worker.min_top_blobs();
        let exact_num_top_blobs = self.layers.get(layer_id).unwrap().worker.exact_num_top_blobs();
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
        for top_id in 0..(self.top_vecs[layer_id].len() - 1) {
            if self.blob_loss_weights.len() <= self.top_id_vecs[layer_id][top_id] {
                self.blob_loss_weights.resize(self.top_id_vecs[layer_id][top_id] + 1, 0f32);
            }
            self.blob_loss_weights[self.top_id_vecs[layer_id][top_id]] = *layer.loss(top_id).unwrap();
            info!("Top shape: {}",
                  self.top_vecs[layer_id][top_id].read().unwrap().shape_string());
            info!("   with loss weight {}", *layer.loss(top_id).unwrap());
        }

        // TODO: only needed if we allow blobs to be passed along in the layer_config
        // const int param_size = layer_param.param_size();
        // const int num_param_blobs = layers_[layer_id]->blobs().size();
        // CHECK_LE(param_size, num_param_blobs)
        //     << "Too many params specified for layer " << layer_param.name();
        // ParamSpec default_param_spec;
        // for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
        //   const ParamSpec* param_spec = (param_id < param_size) ?
        //       &layer_param.param(param_id) : &default_param_spec;
        //   const bool param_need_backward = param_spec->lr_mult() != 0;
        //   need_backward |= param_need_backward;
        //   layers_[layer_id]->set_param_propagate_down(param_id,
        //                                               param_need_backward);
        // }
        // for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
        //   AppendParam(param, layer_id, param_id);
        // }

        // Finally, set the backward flag
        self.layer_need_backwards.push(need_backward);
        if need_backward {
            for top_id in 0..(self.top_id_vecs[layer_id].len() - 1) {
                self.blob_need_backwards[self.top_id_vecs[layer_id][top_id]] = true;
            }
        }
    }

    // Go through the net backwards to determine which blobs contribute to the
    // loss.  We can skip backward computation for blobs that don't contribute
    // to the loss.
    // Also checks if all bottom blobs don't need backward computation (possible
    // because the skip_propagate_down param) and so we can skip bacward
    // computation for the entire layer
    fn init_backprop(&self,
                     layer_id: usize,
                     layer_need_backwards: &mut Vec<bool>,
                     bottom_need_backwards: &mut Vec<Vec<bool>>,
                     blobs_under_loss: &mut HashSet<String>,
                     blobs_skip_backp: &mut HashSet<String>) {
        let mut layer_contributes_loss = false;
        let mut layer_skip_propagate_down = true;
        for (top_id, _) in self.top_vecs[layer_id].iter().enumerate() {
            let blob_name = self.blob_names[self.top_id_vecs[layer_id][top_id]].clone();

            // layer is a loss layer or under a loss layer
            if self.layers[layer_id].loss(top_id).is_some() || blobs_under_loss.contains(&blob_name) {
                layer_contributes_loss = true;
            }
            // layer is not marked to skip backprop TODO: confirm doc
            if !blobs_skip_backp.contains(&blob_name) {
                layer_skip_propagate_down = false;
            }
            // layer contributes loss to some
            if layer_contributes_loss && !layer_skip_propagate_down {
                break;
            }
        }

        // If this layer can skip backward computation, also all his bottom blobs
        // don't need backpropagation
        if layer_need_backwards[layer_id] && layer_skip_propagate_down {
            layer_need_backwards[layer_id] = false;
            for (bottom_id, _) in self.bottom_vecs[layer_id].iter().enumerate() {
                bottom_need_backwards[layer_id][bottom_id] = false;
            }
        }
        // layer doesn't contribute loss so it does not need to be backpropagated
        if !layer_contributes_loss {
            layer_need_backwards[layer_id] = false;
        }
        // if (Caffe::root_solver()) { // Caffe
        {
            info!("{} needs backward computation: {}",
                  self.layer_names[layer_id],
                  self.layer_need_backwards[layer_id]);
        }

        for (bottom_id, _) in self.bottom_vecs[layer_id].iter().enumerate() {
            let blob_name = &self.blob_names[self.bottom_id_vecs[layer_id][bottom_id]];
            if layer_contributes_loss {
                blobs_under_loss.insert(blob_name.clone());
            } else {
                bottom_need_backwards[layer_id][bottom_id] = false;
            }
            if !self.bottom_need_backwards[layer_id][bottom_id] {
                blobs_skip_backp.insert(blob_name.clone());
            }
        }
    }

    fn init_force_backward(&mut self) {
        for (layer_id, layer) in self.layers.iter_mut().enumerate() {
            self.layer_need_backwards[layer_id] = true;
            for (bottom_id, _) in self.bottom_need_backwards[layer_id].clone().iter().enumerate() {
                self.bottom_need_backwards[layer_id][bottom_id] =
                    *self.bottom_need_backwards[layer_id]
                         .get(bottom_id)
                         .unwrap_or(&layer.worker.allow_force_backward(bottom_id));
                self.blob_need_backwards[self.bottom_id_vecs[layer_id][bottom_id]] =
                    *self.blob_need_backwards
                         .get(self.bottom_id_vecs[layer_id][bottom_id])
                         .unwrap_or(&self.bottom_need_backwards[layer_id][bottom_id])
            }
            for (param_id, _) in layer.blobs.clone().iter().enumerate() {
                layer.set_param_propagate_down(param_id, true);
            }
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

    fn share_weights(&mut self) {
        // for (int i = 0; i < params_.size(); ++i) {
        //     if (param_owners_[i] < 0) { continue; }
        //     params_[i]->ShareData(*params_[param_owners_[i]]);
        //     params_[i]->ShareDiff(*params_[param_owners_[i]]);
        // }
        unimplemented!();
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
                    blob_name = "(automatic)".to_owned();
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

            let blob_pointer: ArcLock<HeapBlob> = Arc::new(RwLock::new(Box::new(Blob::new())));
            let blob_id = self.blobs.len();
            self.blobs.push(blob_pointer.clone());
            self.blob_names.push(blob_name.to_owned());
            self.blob_need_backwards.push(false);
            if blob_name_to_idx.is_some() {
                blob_name_to_idx.unwrap().insert(blob_name.to_owned(), blob_id);
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
            available_blobs.unwrap().insert(blob_name.to_owned());
        }
    }

    fn append_bottom(&mut self,
                     param: &NetworkConfig,
                     layer_id: usize,
                     bottom_id: usize,
                     available_blobs: &mut HashSet<String>,
                     blob_name_to_idx: &mut HashMap<String, usize>)
                     -> usize {
        let layer_config = param.layer(layer_id).unwrap();
        let blob_name = layer_config.bottom(bottom_id).unwrap();

        if !available_blobs.contains(blob_name) {
            error!("Unknown bottom blob {} (layer '{}', bottom index {})",
                   blob_name,
                   layer_config.name,
                   bottom_id);
        }

        let blob_id = blob_name_to_idx[blob_name];
        info!("{} <- {}", self.layer_names[layer_id], blob_name);

        self.bottom_vecs[layer_id].push(self.blobs[blob_id].clone());
        self.bottom_id_vecs[layer_id].push(blob_id);
        available_blobs.remove(blob_name);

        let mut propagate_down = true;
        // Check if the backpropagation on bottom_id should be skipped
        if !layer_config.propagate_down.is_empty() {
            propagate_down = layer_config.propagate_down[bottom_id];
        }
        let need_backward = self.blob_need_backwards[blob_id] && propagate_down;
        self.bottom_need_backwards[layer_id].push(need_backward);

        blob_id
    }

    fn append_param(&mut self, param: &NetworkConfig, layer_id: usize, param_id: usize) {
        let layer_config = self.layers[layer_id].config.clone();
        let param_size = self.params.len();
        let param_name = if param_size > param_id {
            layer_config.param(param_id).unwrap().name.clone()
        } else {
            "".to_owned()
        };

        // use param_name (or param_id as a fallback) as display_name
        if !param_name.is_empty() {
            self.param_display_names.push(param_name.clone());
        } else {
            self.param_display_names.push(format!("{}", param_id));
        }

        // add to tracking vectors
        let net_param_id = param_size;
        self.params.push(self.layers[layer_id].blobs[param_id].clone());
        self.param_id_vecs[layer_id].push(net_param_id);
        self.param_layer_indices.push((layer_id, param_id));

        let mut param_spec = &ParamConfig::default();
        if layer_config.params_len() > param_id {
            param_spec = layer_config.param(param_id).unwrap();
        }
        // This layer "owns" this parameter blob -- it is either anonymous
        // (i.e., not given a param_name) or explicitly given a name that we
        // haven't already seen.
        if param_name.is_empty() || !self.param_names_index.contains_key(&param_name) {
            self.param_owners.push(None);
            if !param_name.is_empty() {
                self.param_names_index.insert(param_name.clone(), net_param_id);
            }
            let learnable_param_id = self.learnable_params.len();
            self.learnable_params.push(self.params[net_param_id].clone());
            self.learnable_param_ids.push(learnable_param_id);
            //     has_params_lr_.push_back(param_spec->has_lr_mult());
            //     has_params_decay_.push_back(param_spec->has_decay_mult());
            self.params_lr.push(param_spec.lr_mult.clone());
            self.params_weight_decay.push(param_spec.decay_mult.clone());
        } else {
            // Named param blob with name we've seen before: share params

            let owner_net_param_id = *self.param_names_index.get(&param_name).unwrap();
            self.param_owners.push(Some(owner_net_param_id));
            let (owner_layer_id, owner_param_id) = self.param_layer_indices[owner_net_param_id];
            info!("Sharing parameters '{}' owned by layer '{}', param index {}",
                  param_name.clone(),
                  self.layer_names[owner_layer_id],
                  owner_param_id);
            let this_blob = self.layers[layer_id].blobs[param_id].clone();
            let owner_blob = self.layers[owner_layer_id].blobs[owner_param_id].clone();
            // can only share parameters if blobs match by shape or capacity
            if param_size > param_id {
                if let Err(e) = layer_config.param(param_id)
                                            .unwrap()
                                            .check_dimensions(&this_blob.read().unwrap(),
                                                              &owner_blob.read().unwrap(),
                                                              param_name.clone(),
                                                              self.layer_names[owner_layer_id].clone(),
                                                              self.layer_names[layer_id].clone()) {
                    error!("{}", e)
                }
            }

            let learnable_param_id = self.learnable_param_ids[owner_net_param_id];
            self.learnable_param_ids.push(learnable_param_id);
            // can only share parameters if both have same lr_mult
            if let Some(lr_mult) = param_spec.lr_mult {
                if let Some(owner_lr_mult) = self.params_lr[learnable_param_id] {
                    if !lr_mult.eq(&owner_lr_mult) {
                        error!("Shared param '{}' has mismatched lr_mult.",
                               param_name.clone());
                    }
                } else {
                    self.params_lr[learnable_param_id] = param_spec.lr_mult;
                }
            }
            // can only share parameters if both have same decay_mult
            if let Some(decay_mult) = param_spec.decay_mult {
                if let Some(owner_decay_mult) = self.params_weight_decay[learnable_param_id] {
                    if !decay_mult.eq(&owner_decay_mult) {
                        error!("Shared param '{}' has mismatched decay_mult.",
                               param_name.clone());
                    }
                } else {
                    self.params_weight_decay[learnable_param_id] = param_spec.decay_mult;
                }
            }
        }
    }


    /// Comute one forward and backward step for the network.
    pub fn forward_backward(&mut self, bottom: &[ArcLock<HeapBlob>]) -> f32 {
        let loss = &mut 0f32;

        self.forward(bottom, loss);
        // self.backward();

        *loss
    }

    /// Copy supplied bottom to input blobs and compute one forward step for the network.
    pub fn forward(&mut self, bottom: &[ArcLock<HeapBlob>], loss: &mut f32) -> &Vec<ArcLock<HeapBlob>> {
        for (i, btm) in bottom.iter().enumerate() {
            self.input_blobs[i] = btm.clone();
        }

        self.forward_prefilled(Some(loss))
    }

    /// Compute one forward step for the network after the input blobs have been put into the network.
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

    /// Forward the network from the supplied start to the supplied end.
    pub fn forward_from_to(&mut self, start: usize, end: usize) -> f32 {
        assert!(end < self.layers.len());

        let mut loss = 0f32;

        //  Caffe
        //   if (debug_info_) {
        //     for (int i = 0; i < net_input_blobs_.size(); ++i) {
        //       InputDebugInfo(i);
        //     }
        //   }

        for i in start..end {
            loss += self.layers[i].worker.forward(&self.bottom_vecs[i], &mut self.top_vecs[i]);
            // if (debug_info_) { ForwardDebugInfo(i); }  // Caffe
        }

        loss
    }

    /// Forward the network from the supplied start until the end.
    pub fn forward_from(&mut self, start: usize) -> f32 {
        let end = self.layers.len() - 1;
        self.forward_from_to(start, end)
    }

    /// Forward the network from the start until the supplied end.
    pub fn forward_to(&mut self, end: usize) -> f32 {
        self.forward_from_to(0, end)
    }
}

#[derive(Debug)]
/// The Network Configuration
pub struct NetworkConfig {
    /// The name of the `Network`
    pub name: String,

    /// The names of input `Blob`s to the `Network`
    inputs: Vec<String>,

    /// The shape of the input `Blob`s.
    input_shapes: Vec<Vec<usize>>,

    /// Whether the `Network` will force every layer to carry out backward operation.
    /// If set `false`, then whether to carry out backward is determined
    /// automatically according to the net structure and learning rates.
    force_backward: bool,

    // // The current "state" of the network, including the phase, level, and stage.
    // // Some layers may be included/excluded depending on this state and the states
    // // specified in the layers' include and exclude fields.
    // optional NetState state = 6;

    /// Wheter the `Network` will print debugging information about results
    debug_info: bool,

    /// The `Layers` that make up the `Network`
    pub layers: Vec<LayerConfig>,
}

impl NetworkConfig {

    /// Return a specifc `Layer`
    pub fn layer(&self, layer_id: usize) -> Option<&LayerConfig> {
        self.layers.get(layer_id)
    }

    /// Return a specific `Blob`s' name
    pub fn input(&self, input_id: usize) -> Option<&String> {
        self.inputs.get(input_id)
    }

    /// Return a specific `Blob`s' shape
    pub fn input_shape(&self, input_id: usize) -> Option<&Vec<usize>> {
        self.input_shapes.get(input_id)
    }
}
