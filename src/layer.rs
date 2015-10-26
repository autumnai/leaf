use math::*;
use shared_memory::*;
use blob::Blob;

pub struct Layer {
    top: Vec<Layer>,
}

fn sigmoid(z: f32) -> f32 {
    1f32 / (1f32 + (-z).exp())
}

fn sigmoid_prime(z: f32) -> f32 {
    sigmoid_prime_precalc(sigmoid(z))
}

fn sigmoid_prime_precalc(sigmoid_z: f32) -> f32 {
    sigmoid_z * (1f32 - sigmoid_z)
}

impl Layer {
    pub fn forward(&self, bottom: &[HeapBlob], top: &mut Vec<HeapBlob>) -> f32 {
        // Lock();
        // Reshape(bottom, top); // Reshape the layer to fit top & bottom blob
        let mut loss = 0f32;

        self.forward_cpu(bottom, top);

        for top_layer in top {
            // if (!this->loss(top_id)) { continue; } // C++
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

    // forward_cpu for sigmoid layer
    pub fn forward_cpu(&self, bottom: &[HeapBlob], top: &mut Vec<HeapBlob>) {
        let bottom_data = bottom[0].cpu_data();
        let top_data = top[0].mutable_cpu_data();

        for (i, _) in bottom_data.iter().enumerate() {
            top_data[i] = sigmoid(bottom_data[i])
        }
    }

    // backward_cpu for sigmoid layer
    pub fn backward_cpu(&self, top: &[HeapBlob], propagate_down: &[bool], bottom: &mut Vec<HeapBlob>) {
        if propagate_down[0] {
            let top_data = top[0].cpu_data();
            let top_diff = top[0].cpu_diff();
            let count = bottom[0].len();
            let bottom_diff = bottom[0].mutable_cpu_diff();

            for i in 0..count {
                let sigmoid_x = top_data[i];
                // bottom_diff[i] = top_diff[i] * sigmoid_x * (1f32 - sigmoid_x);
                bottom_diff[i] = top_diff[i] * sigmoid_prime_precalc(sigmoid_x)
            }
        }
    }

    pub fn auto_top_blobs(&self) -> bool {
        false
    }
    pub fn min_top_blobs(&self) -> usize {
        0
    }
    pub fn exact_num_top_blobs(&self) -> usize {
        0
    }
}

pub struct LayerConfig {
    pub name: String, // the layer name
    layer_type: String, // the layer type

    bottoms: Vec<String>, // the name of each bottom blob; called bottom in Caffe
    tops: Vec<String>, // the name of each top blob; called top in Caffe

    // Specifies on which bottoms the backpropagation should be skipped.
    // The size must be either 0 or equal to the number of bottoms.
    propagate_down: Vec<bool>, // minimal, a lot of Caffe not ported yet
}

impl LayerConfig {
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

    pub fn check_propagate_down_len(&self) -> bool {
        self.propagate_down.is_empty() || self.propagate_down.len() == self.bottoms.len()
    }
}
