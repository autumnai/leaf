extern crate leaf;
extern crate collenchyma as co;

#[cfg(test)]
mod layer_spec {
    use std::rc::Rc;
    use co::prelude::*;
    use leaf::layer::*;
    use leaf::layers::*;

    fn new_layer_config() -> LayerConfig {
        LayerConfig::new("foo", LayerType::Sigmoid)
    }


    fn native_backend() -> Rc<Backend<Native>> {
        Rc::new(Backend::<Native>::default().unwrap())
    }

    #[cfg(feature="cuda")]
    mod cuda {
        use std::rc::Rc;
        use co::prelude::*;
        use leaf::layer::*;
        use leaf::layers::*;

        fn cuda_backend() -> Rc<Backend<Cuda>> {
            Rc::new(Backend::<Cuda>::default().unwrap())
        }

        #[test]
        fn new_layer() {
            let cfg = super::new_layer_config();
            Layer::from_config(cuda_backend(), &cfg);
        }

        #[test]
        fn can_create_empty_sequential_layer() {
            let model = SequentialConfig::default();
            Layer::from_config(cuda_backend(), &LayerConfig::new("model", LayerType::Sequential(model)));
        }

        #[test]
        fn can_create_single_layer_sequential_layer() {
            let mut model = SequentialConfig::default();
            model.add_input("data", &vec![28, 28]);
            model.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));

            Layer::from_config(cuda_backend(), &LayerConfig::new("model", LayerType::Sequential(model)));
        }

        #[test]
        fn can_create_simple_network_sequential_layer() {
            let mut model = SequentialConfig::default();
            model.add_input("data", &vec![1, 784]);
            let linear1_cfg = LinearConfig { output_size: 1568 };
            model.add_layer(LayerConfig::new("linear1", LayerType::Linear(linear1_cfg)));
            model.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
            let linear2_cfg = LinearConfig { output_size: 10 };
            model.add_layer(LayerConfig::new("linear2", LayerType::Linear(linear2_cfg)));

            let network = Layer::from_config(cuda_backend(), &LayerConfig::new("model", LayerType::Sequential(model)));
        }
    }

    // #[test]
    // fn dim_check_strict() {
    //     let cfg = WeightConfig { share_mode: DimCheckMode::Strict, ..WeightConfig::default() };
    //     let blob_one = SharedTensor::<f32>::new(backend().device(), &vec![2, 3, 3]);
    //     let blob_two = SharedTensor::<f32>::new(backend().device(), &vec![3, 2, 3]);
    //     let param_name = "foo".to_owned();
    //     let owner_name = "owner".to_owned();
    //     let layer_name = "layer".to_owned();
    //
    //     assert!(cfg.check_dimensions(&blob_one,
    //                                  &blob_one,
    //                                  param_name.clone(),
    //                                  owner_name.clone(),
    //                                  layer_name.clone())
    //                .is_ok());
    //     assert!(cfg.check_dimensions(&blob_one,
    //                                  &blob_two,
    //                                  param_name.clone(),
    //                                  owner_name.clone(),
    //                                  layer_name.clone())
    //                .is_err());
    // }

    // #[test]
    // fn dim_check_permissive() {
    //     let cfg = WeightConfig { share_mode: DimCheckMode::Permissive, ..WeightConfig::default() };
    //     let blob_one = SharedTensor::<f32>::new(backend().device(), &vec![2, 3, 3]);
    //     let blob_two = SharedTensor::<f32>::new(backend().device(), &vec![3, 2, 3]);
    //     let blob_three = SharedTensor::<f32>::new(backend().device(), &vec![3, 10, 3]);
    //     let param_name = "foo".to_owned();
    //     let owner_name = "owner".to_owned();
    //     let layer_name = "layer".to_owned();
    //
    //     assert!(cfg.check_dimensions(&blob_one,
    //                                  &blob_one,
    //                                  param_name.clone(),
    //                                  owner_name.clone(),
    //                                  layer_name.clone())
    //                .is_ok());
    //     assert!(cfg.check_dimensions(&blob_one,
    //                                  &blob_two,
    //                                  param_name.clone(),
    //                                  owner_name.clone(),
    //                                  layer_name.clone())
    //                .is_ok());
    //     assert!(cfg.check_dimensions(&blob_one,
    //                                  &blob_three,
    //                                  param_name.clone(),
    //                                  owner_name.clone(),
    //                                  layer_name.clone())
    //                .is_err());
    //     assert!(cfg.check_dimensions(&blob_two,
    //                                  &blob_three,
    //                                  param_name.clone(),
    //                                  owner_name.clone(),
    //                                  layer_name.clone())
    //                .is_err());
    // }
}
