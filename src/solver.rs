use shared_memory::*;
use network::*;
use solvers::*;

#[derive(Debug, Copy, Clone)]
/// Enum that holds all possible types of sovlers.
pub enum SolverKind {
    /// SGD = Stochastic Gradient Descent
    SGD,
}

#[derive(Debug)]
/// Solver that optimizes a `Network`
pub struct Solver<'a, S> {
    kind: SolverKind,
    net: Network<'a>,
    /// The implementation of the Solver
    pub worker: Box<S>,

    param: SolverConfig,
    /// The current iteration / number of times weights have been updated
    iter: usize,
    /// The current step (for the learning rate)
    current_step: usize,
}

impl<'a, S: ISolver> Solver<'a, S>{
    fn init(&'a mut self, param: SolverConfig) {
        // Caffe
        //   CHECK(Caffe::root_solver() || root_solver_)
        //       << "root_solver_ needs to be set for all non-root solvers";
        info!("Initializing solver from parameters: {:?}", param);
        self.param = param;
        assert!(self.param.average_loss > 1);
        // Caffe
        //   if (Caffe::root_solver() && param_.random_seed() >= 0) {
        //     Caffe::set_random_seed(param_.random_seed());
        //   }

        Solver::<S>::init_train_net(&mut self.param, &mut self.net);
        // if (Caffe::root_solver()) {
        {
            // self.init_test_nets();
            info!("Solver scaffolding done.");
        }
        self.iter = 0;
        self.current_step = 0;
    }

    /// Initialize the training net
    fn init_train_net<'_>(param: &'_ mut SolverConfig, net: &'_ mut Network<'_>) {
        // Caffe
        // Set the correct NetState.  We start with the solver defaults (lowest
        // precedence); then, merge in any NetState specified by the net_param itself;
        // finally, merge in any NetState specified by the train_state (highest
        // precedence).
        // NetState net_state;
        // net_state.set_phase(TRAIN);
        // net_state.MergeFrom(net_param.state());
        // net_state.MergeFrom(param_.train_state());
        // net_param.mutable_state()->CopyFrom(net_state);

        // TODO: there currently is no merging; we probably only need solver_default ||
        // net_param
        let solver_default = NetworkState { mode: NetworkMode::Train, ..NetworkState::default() };
        param.train_net.state = solver_default;

        // Caffe
        // if (Caffe::root_solver()) {
        //     net_.reset(new Net<Dtype>(net_param));
        // } else {
        //     net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
        // }
        *net = Network::from_config(&param.train_net);

        unimplemented!();
    }

    /// Initialize all the test nets
    fn init_test_nets(&mut self) {
        unimplemented!();
    }

    // might take a solver state as argument in the future to resume a stopped
    // solver
    fn solve(&mut self) {
        info!("Solving {}", self.net.name);

        let num_iter = 100;
        self.step(num_iter);
    }

    fn step(&mut self, iters: usize) {
        let start_iter = self.iter;
        let stop_iter = start_iter + iters;
        // int average_loss = this->param_.average_loss(); // Caffe
        let mut losses = Vec::<f32>::new();
        let mut smoothed_loss = 0f32;

        while self.iter < stop_iter {
            let mut loss = 0f32;

            self.net.clear_param_diffs();
            // if self.param.test_interval.is_some() && self.iter % self.param

            // run tests all `test_interval` iterations
            // unless it's the first iteration and we are not testing on initialization
            if let Some(test_interval) = self.param.test_interval {
                if self.iter % test_interval == 0 && (self.iter > 0 || self.param.test_initialization) {
                    // && Caffe::root_solver()) { // Caffe

                    // TODO
                    //   TestAll();
                    //   if (requested_early_exit_) {
                    //     // Break out of the while loop because stop was requested while testing.
                    //     break;
                    //   }
                }
            }
            // Caffe
            // for (int i = 0; i < callbacks_.size(); ++i) {
            //   callbacks_[i]->on_start();
            // }

            // Caffe : display info every .display() iterations
            // const bool display = param_.display() && iter_ % param_.display() == 0;
            // net_->set_debug_info(display && param_.debug_info());

            let noop_bottom = vec![new_shared_heapblob()];
            for _ in 0..self.param.minibatch_size - 1 {
                loss += self.net.forward_backward(&noop_bottom);
            }
            // average the loss across iterations of minibatch
            loss /= self.param.minibatch_size as f32;

            // average the loss across iterations for smoothed reporting
            if losses.len() < self.param.average_loss {
                losses.push(loss);
                let size = losses.len() as f32;
                smoothed_loss = (smoothed_loss * (size - 1f32) + loss) / size;
            } else {
                let idx = (self.iter - start_iter) % self.param.average_loss;
                smoothed_loss += (loss - losses[idx]) / self.param.average_loss as f32;
                losses[idx] = loss;
            }

            // Caffe
            // if (display) {
            //   LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
            //       << ", loss = " << smoothed_loss;
            //   const vector<Blob<Dtype>*>& result = net_->output_blobs();
            //   int score_index = 0;
            //   for (int j = 0; j < result.size(); ++j) {
            //     const Dtype* result_vec = result[j]->cpu_data();
            //     const string& output_name =
            //         net_->blob_names()[net_->output_blob_indices()[j]];
            //     const Dtype loss_weight =
            //         net_->blob_loss_weights()[net_->output_blob_indices()[j]];
            //     for (int k = 0; k < result[j]->count(); ++k) {
            //       ostringstream loss_msg_stream;
            //       if (loss_weight) {
            //         loss_msg_stream << " (* " << loss_weight
            //                         << " = " << loss_weight * result_vec[k] << " loss)";
            //       }
            //       LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
            //           << score_index++ << ": " << output_name << " = "
            //           << result_vec[k] << loss_msg_stream.str();
            //     }
            //   }
            // }
            // for (int i = 0; i < callbacks_.size(); ++i) {
            //   callbacks_[i]->on_gradients_ready();
            // }

            // Caffe / Display
            //   if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
            //     LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
            //   }
            self.worker.apply_update(&self.param, &mut self.net, self.iter);

            // Increment the internal iter counter -- its value should always indicate
            // the number of times the weights have been updated.
            self.iter += 1;

            // Caffe
            // SolverAction::Enum request = GetRequestedAction();
            //
            // // Save a snapshot if needed.
            // if ((param_.snapshot()
            //      && iter_ % param_.snapshot() == 0
            //      && Caffe::root_solver()) ||
            //      (request == SolverAction::SNAPSHOT)) {
            //   Snapshot();
            // }
            // if (SolverAction::STOP == request) {
            //   requested_early_exit_ = true;
            //   // Break out of training loop.
            //   break;
            // }

        }
    }
}

/// Implementation of a specific Solver.
pub trait ISolver {
    /// TODO: what does this do?
    fn apply_update(&self, param: &SolverConfig, network: &mut Network, iter: usize);
}

#[derive(Debug)]
/// Configuration for a Solver
pub struct SolverConfig {
    /// Name of the solver.
    pub name: String,
    /// The `NetworkConfig` that is used to initialize the training network.
    pub train_net: NetworkConfig,
    /// Display the loss averaged over the last average_loss iterations.
    ///
    /// Default: 1
    pub average_loss: usize,
    /// The number of iterations between two testing phases.
    ///
    /// Default: None
    pub test_interval: Option<usize>,
    /// If true, run an initial test pass before the first iteration,
    /// ensuring memory availability and printing the starting value of the loss.
    ///
    /// Default: true
    pub test_initialization: bool,
    /// Accumulate gradients over minibatch_size instances.
    ///
    /// Default: 1
    pub minibatch_size: usize,
    /// The learning rate policy to be used.
    ///
    /// Default: Fixed
    pub lr_policy: LRPolicy,
    /// The base learning rate.
    ///
    /// Default: 0.01
    pub base_lr: f32,
    /// gamma as used in the calculation of most learning rate policies.
    ///
    /// Default: 0.1
    pub gamma: f32,
    /// The stepsize used in Step and Sigmoid learning policies.
    ///
    /// Default: 10
    pub stepsize: usize,
}

impl Default for SolverConfig {
    fn default() -> SolverConfig {
        SolverConfig {
            name: "".to_owned(),
            train_net: NetworkConfig::default(),

            average_loss: 1,
            test_interval: None,
            test_initialization: true,
            minibatch_size: 1,

            lr_policy: LRPolicy::Fixed,
            base_lr: 0.01f32,
            gamma: 0.1f32,

            stepsize: 10,
        }
    }
}

impl SolverConfig {
    /// Return test interval (configured value or default of 0).
    pub fn test_interval(&self) -> usize {
        self.test_interval.unwrap_or(0)
    }

    /// Return the current learning rate. The currently implemented learning rate
    /// policies are as follows:
    ///    - fixed: always return base_lr.
    ///    - step: return base_lr * gamma ^ (floor(iter / step))
    ///    - exp: return base_lr * gamma ^ iter
    ///    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
    ///    - multistep: similar to step but it allows non uniform steps defined by
    ///      stepvalue
    ///    - poly: the effective learning rate follows a polynomial decay, to be
    ///      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
    ///    - sigmoid: the effective learning rate follows a sigmod decay
    ///      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
    ///
    /// where base_lr, max_iter, gamma, step, stepvalue and power are defined
    /// in the solver config, and iter is the current iteration.
    pub fn get_learning_rate(&self, iter: usize) -> f32 {
        match self.lr_policy() {
            LRPolicy::Fixed => {
                self.base_lr()
            }
            LRPolicy::Step => {
                let current_step = iter / self.stepsize();
                self.base_lr() * self.gamma().powf(current_step as f32)
            }
            LRPolicy::Multistep => {
                //   if (this->current_step_ < this->param_.stepvalue_size() &&
                //         this->iter_ >= this->param_.stepvalue(this->current_step_)) {
                //     this->current_step_++;
                //     LOG(INFO) << "MultiStep Status: Iteration " <<
                //     this->iter_ << ", step = " << this->current_step_;
                //   }
                //   rate = this->param_.base_lr() *
                //       pow(this->param_.gamma(), this->current_step_);
                unimplemented!();
            }
            LRPolicy::Exp => {
                self.base_lr() * self.gamma().powf(iter as f32)
            }
            LRPolicy::Inv => {
                //   rate = this->param_.base_lr() *
                //       pow(Dtype(1) + this->param_.gamma() * this->iter_,
                //           - this->param_.power());
                unimplemented!();
            }
            LRPolicy::Poly => {
                //   rate = this->param_.base_lr() * pow(Dtype(1.) -
                //       (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
                //       this->param_.power());
                unimplemented!();
            }
            LRPolicy::Sigmoid => {
                //   rate = this->param_.base_lr() * (Dtype(1.) /
                //       (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
                //         Dtype(this->param_.stepsize())))));
                unimplemented!();
            }
        }
    }

    /// Return learning rate policy.
    fn lr_policy(&self) -> LRPolicy {
        self.lr_policy
    }

    /// Return the base learning rate.
    fn base_lr(&self) -> f32 {
        self.base_lr
    }

    /// Return the gamma for learning rate calculations.
    fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Return the stepsize for learning rate calculations.
    fn stepsize(&self) -> usize {
        self.stepsize
    }
}


#[derive(Debug, Copy, Clone)]
/// Learning Rate Policy for a Solver
pub enum LRPolicy {
    /// always return base_lr
    Fixed,
    /// learning rate decays every `step` iterations.
    /// return base_lr * gamma ^ (floor(iter / step))
    Step,
    /// similar to step but it allows non uniform steps defined by
    /// stepvalue
    Multistep,
    /// return base_lr * gamma ^ iter
    Exp,
    /// return base_lr * (1 + gamma * iter) ^ (- power)
    Inv,
    /// the effective learning rate follows a polynomial decay, to be
    /// zero by the max_iter.
    /// return base_lr (1 - iter/max_iter) ^ (power)
    Poly,
    /// the effective learning rate follows a sigmod decay
    /// return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
    Sigmoid,
}
