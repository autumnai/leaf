extern crate leaf;
extern crate collenchyma as co;

#[cfg(test)]
mod layer_spec {
    use std::rc::Rc;
    use co::prelude::*;
    use leaf::layer::*;

    fn new_layer_config() -> LayerConfig {
        LayerConfig::new("foo", LayerType::Sigmoid)
    }

    fn native_backend() -> Rc<Backend<Native>> {
        Rc::new(Backend::<Native>::default().unwrap())
    }

    #[cfg(feature="cuda")]
    fn cuda_backend() -> Rc<Backend<Cuda>> {
        Rc::new(Backend::<Cuda>::default().unwrap())
    }

    #[cfg(all(feature="native", feature="cuda"))]
    mod native_cuda {
        use leaf::layer::*;
        use leaf::layers::*;
        use super::{native_backend, cuda_backend};

        #[test]
        fn create_layer_with_either() {
            let cfg = super::new_layer_config();
            Layer::from_config(native_backend(), &cfg);

            let cfg = super::new_layer_config();
            Layer::from_config(cuda_backend(), &cfg);
        }
    }

    #[cfg(feature="cuda")]
    mod cuda {
        use std::sync::{Arc, RwLock};
        use co::prelude::*;
        use leaf::layer::*;
        use leaf::layers::*;
        use leaf::util::write_to_memory;
        use super::{native_backend, cuda_backend};

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
            model.add_layer(LayerConfig::new("linear1", LinearConfig { output_size: 1568 }));
            model.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
            model.add_layer(LayerConfig::new("linear2", LinearConfig { output_size: 10 }));

            let _ = Layer::from_config(cuda_backend(), &LayerConfig::new("model", LayerType::Sequential(model)));
        }

        #[test]
        fn reshape_does_not_affect_output() {
            let native_backend = native_backend();
            let cuda_backend = cuda_backend();

            let mut normal_model = SequentialConfig::default();
            normal_model.add_input("data", &vec![3]);
            normal_model.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
            let mut normal_network = Layer::from_config(cuda_backend.clone(), &LayerConfig::new("normal_model", LayerType::Sequential(normal_model)));

            let mut reshape_model = SequentialConfig::default();
            reshape_model.add_input("data", &vec![3]);
            reshape_model.add_layer(LayerConfig::new("reshape", ReshapeConfig { shape: vec![1, 1, 3] }));
            reshape_model.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
            let mut reshape_network = Layer::from_config(cuda_backend.clone(), &LayerConfig::new("reshape_model", LayerType::Sequential(reshape_model)));

            let input = vec![1f32, 1f32, 2f32];
            let mut normal_tensor = SharedTensor::<f32>::new(native_backend.device(), &(3)).unwrap();
            // let mut normal_tensor_output = SharedTensor::<f32>::new(native_backend.device(), &(3)).unwrap();
            let mut reshape_tensor = SharedTensor::<f32>::new(native_backend.device(), &(3)).unwrap();
            // let mut reshape_tensor_output = SharedTensor::<f32>::new(native_backend.device(), &(3)).unwrap();
            write_to_memory(normal_tensor.get_mut(native_backend.device()).unwrap(), &input);
            write_to_memory(reshape_tensor.get_mut(native_backend.device()).unwrap(), &input);

            let normal_tensor_output = normal_network.forward(&[Arc::new(RwLock::new(normal_tensor))])[0].clone();
            let _ = normal_tensor_output.write().unwrap().add_device(native_backend.device());
            normal_tensor_output.write().unwrap().sync(native_backend.device()).unwrap();
            let normal_tensor_output_native_ = normal_tensor_output.read().unwrap();
            let normal_tensor_output_native = normal_tensor_output_native_.get(native_backend.device()).unwrap().as_native().unwrap();
            assert_eq!(&[0.7310585786f32, 0.7310586f32, 0.880797f32], normal_tensor_output_native.as_slice::<f32>());

            let reshape_tensor_output = reshape_network.forward(&[Arc::new(RwLock::new(reshape_tensor))])[0].clone();
            let _ = reshape_tensor_output.write().unwrap().add_device(native_backend.device());
            reshape_tensor_output.write().unwrap().sync(native_backend.device()).unwrap();
            let reshape_tensor_output_native_ = reshape_tensor_output.read().unwrap();
            let reshape_tensor_output_native = reshape_tensor_output_native_.get(native_backend.device()).unwrap().as_native().unwrap();
            assert_eq!(&[0.7310585786f32, 0.7310586f32, 0.880797f32], reshape_tensor_output_native.as_slice::<f32>());
            assert_eq!(normal_tensor_output_native.as_slice::<f32>(), reshape_tensor_output_native.as_slice::<f32>());
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
