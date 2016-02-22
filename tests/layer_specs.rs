extern crate leaf;
extern crate collenchyma as co;

#[cfg(all(test, whatever))]
// #[cfg(test)]
mod layer_spec {

    use leaf::layer::*;
    use std::rc::Rc;
    use co::backend::{Backend, BackendConfig};
    use co::frameworks::Native;
    use co::framework::IFramework;

    fn new_layer_config() -> LayerConfig {
        LayerConfig::new("foo".to_owned(), LayerType::Sigmoid)
    }

    fn backend() -> Rc<Backend<Native>> {
        let framework = Native::new();
        let hardwares = framework.hardwares();
        let backend_config = BackendConfig::new(framework, hardwares);
        Rc::new(Backend::new(backend_config).unwrap())
    }

    #[test]
    fn new_layer() {
        let cfg = new_layer_config();
        Layer::from_config(backend(), &cfg);
    }

    #[test]
    fn dim_check_strict() {
        let cfg = WeightConfig { share_mode: DimCheckMode::Strict, ..WeightConfig::default() };
        let blob_one = Blob::<f32>::of_shape(Some(backend().device()), &[2, 3, 3]);
        let blob_two = Blob::<f32>::of_shape(Some(backend().device()), &[3, 2, 3]);
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
        let cfg = WeightConfig { share_mode: DimCheckMode::Permissive, ..WeightConfig::default() };
        let blob_one = Blob::<f32>::of_shape(Some(backend().device()), &[2, 3, 3]);
        let blob_two = Blob::<f32>::of_shape(Some(backend().device()), &[3, 2, 3]);
        let blob_three = Blob::<f32>::of_shape(Some(backend().device()), &[3, 10, 3]);
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
