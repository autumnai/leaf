use blob::Blob;
use math::*;

pub struct Layer {
    top: Vec<Layer>
}

fn sigmoid(z: f32) -> f32 {
    return 1f32 / (1f32 + (-z).exp())
}

fn sigmoid_prime(z: f32) -> f32 {
    return sigmoid_prime_precalc(sigmoid(z))
}

fn sigmoid_prime_precalc(sigmoid_z: f32) -> f32 {
    return sigmoid_z * (1f32 - sigmoid_z)
}

impl Layer {
    pub fn forward(&self, bottom: &Vec<Box<Blob<f32>>>, top: &mut Vec<Box<Blob<f32>>>) -> f32 {
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

        return loss
    }

    // forward_cpu for sigmoid layer
    pub fn forward_cpu(&self, bottom: &Vec<Box<Blob<f32>>>, top: &mut Vec<Box<Blob<f32>>>){
        let bottom_data = bottom[0].cpu_data();
        let top_data = top[0].mutable_cpu_data();

        for (i, _) in bottom_data.iter().enumerate() {
            top_data[i] = sigmoid(bottom_data[i])
        }
    }

    // backward_cpu for sigmoid layer
    pub fn backward_cpu(&self, top: &Vec<Box<Blob<f32>>>, propagate_down: &Vec<bool>, bottom: &mut Vec<Box<Blob<f32>>>) {
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
}

pub struct LayerConfig {
    pub name: String, // the layer name
    layer_type: String, // the layer type

    bottoms: Vec<String>, // the name of each bottom blob; called bottom in Caffe
    tops: Vec<String>, // the name of each top blob; called top in Caffe

    // minimal, a lot of Caffe not ported yet
}

impl LayerConfig {
    pub fn top(&self, top_id: usize) -> Option<&String> { return self.tops.get(top_id); }
    pub fn bottom(&self, bottom_id: usize) -> Option<&String> { return self.bottoms.get(bottom_id); }
}
