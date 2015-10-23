use layer::Layer;
use blob::Blob;

pub struct Network {
    pub name: String,
    layers: Vec<Layer>
}

impl Network {
    pub fn forward_backward(&self, bottom: &Vec<Box<Blob<f32>>>) -> f32 {
        let loss = 0f32; // TODO

        self.forward(bottom, &loss);
        // self.backward();

        return loss;
    }


    // const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    //     const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
    //   // Copy bottom to internal bottom
    //   for (int i = 0; i < bottom.size(); ++i) {
    //     net_input_blobs_[i]->CopyFrom(*bottom[i]);
    //   }
    //   return ForwardPrefilled(loss);
    // }
    pub fn forward(&self, bottom: &Vec<Box<Blob<f32>>>, loss: &f32) -> &Vec<Box<Blob<f32>>> {
        // let blob: Blob<f32> = Blob::new();
        let blob = vec![Box::new(Blob::new())];
        

        return &blob;
    }
}
