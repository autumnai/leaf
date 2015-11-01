extern crate leaf;
extern crate phloem;

#[cfg(test)]
mod layer_spec {

    use leaf::layer::*;
    use phloem::Blob;

    fn new_layer_config() -> LayerConfig {
        LayerConfig::new("foo".to_owned(), LayerType::Sigmoid)
    }

    #[test]
    fn new_layer() {
        let cfg = new_layer_config();
        Layer::from_config(&cfg);
    }

    #[test]
    fn dim_check_strict() {
        let cfg = ParamConfig { share_mode: DimCheckMode::Strict, ..ParamConfig::default() };
        let blob_one = Blob::<f32>::of_shape(vec![2, 3, 3]);
        let blob_two = Blob::<f32>::of_shape(vec![3, 2, 3]);
        let param_name = "foo".to_owned();
        let owner_name = "owner".to_owned();
        let layer_name = "layer".to_owned();

        assert!(cfg.check_dimensions(&blob_one,
                                     &blob_one,
                                     param_name.clone(),
                                     owner_name.clone(),
                                     layer_name.clone())
                   .is_ok());
        assert!(cfg.check_dimensions(&blob_one,
                                     &blob_two,
                                     param_name.clone(),
                                     owner_name.clone(),
                                     layer_name.clone())
                   .is_err());
    }

    #[test]
    fn dim_check_permissive() {
        let cfg = ParamConfig { share_mode: DimCheckMode::Permissive, ..ParamConfig::default() };
        let blob_one = Blob::<f32>::of_shape(vec![2, 3, 3]);
        let blob_two = Blob::<f32>::of_shape(vec![3, 2, 3]);
        let blob_three = Blob::<f32>::of_shape(vec![3, 10, 3]);
        let param_name = "foo".to_owned();
        let owner_name = "owner".to_owned();
        let layer_name = "layer".to_owned();

        assert!(cfg.check_dimensions(&blob_one,
                                     &blob_one,
                                     param_name.clone(),
                                     owner_name.clone(),
                                     layer_name.clone())
                   .is_ok());
        assert!(cfg.check_dimensions(&blob_one,
                                     &blob_two,
                                     param_name.clone(),
                                     owner_name.clone(),
                                     layer_name.clone())
                   .is_ok());
        assert!(cfg.check_dimensions(&blob_one,
                                     &blob_three,
                                     param_name.clone(),
                                     owner_name.clone(),
                                     layer_name.clone())
                   .is_err());
        assert!(cfg.check_dimensions(&blob_two,
                                     &blob_three,
                                     param_name.clone(),
                                     owner_name.clone(),
                                     layer_name.clone())
                   .is_err());
    }
}
