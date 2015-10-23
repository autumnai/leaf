// use std::f32;
// use synced_memory;
use blob::Blob;

pub struct Layer {
    top: Vec<Layer>
}

fn sigmoid(z: f32) -> f32 {
    return 1f32 / (1f32 + (-z).exp())
}

fn sigmoid_prime(z: f32) -> f32 {
    return sigmoid(z) * (1f32 - sigmoid(z))
}

impl Layer {
    pub fn forward(&self, bottom: &Vec<Box<Blob<f32>>>, top: &mut Vec<Box<Blob<f32>>>) {
        self.forward_cpu(bottom, top);

        // for (int top_id = 0; top_id < top.size(); ++top_id) {
        //   if (!this->loss(top_id)) { continue; }
        //   const int count = top[top_id]->count();
        //   const Dtype* data = top[top_id]->cpu_data();
        //   const Dtype* loss_weights = top[top_id]->cpu_diff();
        //   loss += caffe_cpu_dot(count, data, loss_weights);
        // }
    }

    // forward_cpu for sigmoid layer
    pub fn forward_cpu(&self, bottom: &Vec<Box<Blob<f32>>>, top: &mut Vec<Box<Blob<f32>>>){
        let bottom_data = bottom[0].cpu_data();
        let top_data = top[0].mutable_cpu_data();

        for (i, _) in bottom_data.iter().enumerate() {
            top_data[i] = sigmoid(bottom_data[i])
        }
    }
}
