use shared_memory::*;
use network::Network;

enum SolverKind {
    SGD,
}

struct Solver<'a> {
    kind: SolverKind,
    net: Network<'a>,
    iter: i32,
}

impl<'a> Solver<'a>{
    // might take a solver state as argument in the future to resume a stopped
    // solver
    fn solve(&mut self) {
        info!("Solving {}", self.net.name);

        let num_iter = 100;
        self.step(num_iter);
    }

    fn step(&mut self, iters: i32) {
        let start_iter = self.iter;
        let stop_iter = start_iter + iters;
        // int average_loss = this->param_.average_loss(); // Caffe
        let losses = Vec::<f32>::new();
        let smoothed_loss = 0f32;

        while self.iter < stop_iter {
            let mut loss = 0f32;

            let minibatch_size = 10;

            let noop_bottom = vec![new_shared_heapblob()];
            for _ in 0..minibatch_size {
                loss += self.net.forward_backward(&noop_bottom);
            }

            self.iter += 1;
        }
    }
}
