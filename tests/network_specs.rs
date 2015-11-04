extern crate leaf;
extern crate phloem;

#[cfg(test)]
mod network_spec {
    use leaf::network::*;

    #[test]
    fn new_layer() {
        let cfg = NetworkConfig::default();
        Network::from_config(&cfg);
    }
}
