extern crate leaf;
extern crate phloem;
extern crate collenchyma as co;

#[cfg(test)]
mod network_spec {
    use std::rc::Rc;
    use co::backend::{Backend, BackendConfig};
    use co::framework::IFramework;
    use co::frameworks::Native;
    use leaf::network::*;

    fn backend() -> Rc<Backend<Native>> {
        let framework = Native::new();
        let hardwares = framework.hardwares();
        let backend_config = BackendConfig::new(framework, hardwares);
        Rc::new(Backend::new(backend_config).unwrap())
    }

    #[test]
    fn new_layer() {
        let cfg = NetworkConfig::default();
        Network::from_config(backend(), &cfg);
    }
}
