use solver::*;
use network::Network;

#[derive(Debug, Copy, Clone)]
/// Stochastic Gradient Descent Solver
pub struct SGD;

impl SGD {
    fn clip_gradients(&self) {
        // const Dtype clip_gradients = this->param_.clip_gradients();
        // if (clip_gradients < 0) { return; }
        // const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
        // Dtype sumsq_diff = 0;
        // for (int i = 0; i < net_params.size(); ++i) {
        //   sumsq_diff += net_params[i]->sumsq_diff();
        // }
        // const Dtype l2norm_diff = std::sqrt(sumsq_diff);
        // if (l2norm_diff > clip_gradients) {
        //   Dtype scale_factor = clip_gradients / l2norm_diff;
        //   LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        //       << l2norm_diff << " > " << clip_gradients << ") "
        //       << "by scale factor " << scale_factor;
        //   for (int i = 0; i < net_params.size(); ++i) {
        //     net_params[i]->scale_diff(scale_factor);
        //   }
        // }
    }
}

impl ISolver for SGD {
    fn apply_update(&self, param: &SolverConfig, net: &mut Network, iter: usize) {
        // CHECK(Caffe::root_solver()); // Caffe
        let rate = param.get_learning_rate(iter);

        self.clip_gradients();
        for (param_id, param) in net.learnable_params().iter().enumerate() {
            //     Normalize(param_id);
            //     Regularize(param_id);
            //     ComputeUpdateValue(param_id, rate);
            unimplemented!();
        }
        net.update_params();

        unimplemented!();
    }
}
