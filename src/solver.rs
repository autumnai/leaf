use network::Network;
use phloem::Blob;

enum SolverKind {
    SGD,
}

struct Solver {
    kind: SolverKind,
    net: Network,
    iter: i32,
}

impl Solver{
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

        while self.iter < stop_iter {
            let mut loss = 0f32;

            let minibatch_size = 10;

            let noop_bottom = vec![Box::new(Blob::new())];
            for _ in 0..minibatch_size {
                loss += self.net.forward_backward(&noop_bottom);
            }

            self.iter += 1;
        }
    }
}
