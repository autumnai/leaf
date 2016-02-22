extern crate leaf;
extern crate collenchyma as co;

#[cfg(all(test, whatever))]
// #[cfg(test)]
mod solver_specs {
    use leaf::solver::*;
    use co::backend::Backend;
    use co::frameworks::Native;

    #[test]
    // fixed: always return base_lr.
    fn lr_fixed() {
        let cfg = SolverConfig{ lr_policy: LRPolicy::Fixed, base_lr: 5f32, gamma: 0.5f32, ..SolverConfig::default()};
        assert!(cfg.get_learning_rate(0) == 5f32);
        assert!(cfg.get_learning_rate(100) == 5f32);
        assert!(cfg.get_learning_rate(1000) == 5f32);
    }

    #[test]
    // step: return base_lr * gamma ^ (floor(iter / step))
    fn lr_step() {
        let cfg = SolverConfig{ lr_policy: LRPolicy::Step, base_lr: 5f32, gamma: 0.5f32, stepsize: 10, ..SolverConfig::default()};
        assert!(cfg.get_learning_rate(0) == 5f32);
        assert!(cfg.get_learning_rate(10) == 2.5f32);
        assert!(cfg.get_learning_rate(20) == 1.25f32);
    }

    #[test]
    // exp: return base_lr * gamma ^ iter
    fn lr_exp() {
        let cfg = SolverConfig{ lr_policy: LRPolicy::Exp, base_lr: 5f32, gamma: 0.5f32, ..SolverConfig::default()};
        assert!(cfg.get_learning_rate(0) == 5f32);
        assert!(cfg.get_learning_rate(1) == 2.5f32);
        assert!(cfg.get_learning_rate(2) == 1.25f32);
        assert!(cfg.get_learning_rate(3) == 0.625f32);

        let cfg2 = SolverConfig{ lr_policy: LRPolicy::Exp, base_lr: 5f32, gamma: 0.25f32, ..SolverConfig::default()};
        assert!(cfg2.get_learning_rate(0) == 5f32);
        assert!(cfg2.get_learning_rate(1) == 1.25f32);
        assert!(cfg2.get_learning_rate(2) == 0.3125f32);
    }

    #[test]
    fn instantiate_solver_sgd_momentum() {
        let cfg = SolverConfig{ solver: SolverKind::SGD(SGDKind::Momentum), ..SolverConfig::default()};
        Solver::<Box<ISolver<Backend<Native>>>, Backend<Native>>::from_config(&cfg);
    }
}
