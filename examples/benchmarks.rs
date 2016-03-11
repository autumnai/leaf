#[macro_use]
extern crate timeit;
extern crate env_logger;
extern crate collenchyma as co;
extern crate leaf;

use co::prelude::*;

use std::sync::{Arc, RwLock};
use leaf::layers::*;
use leaf::layer::*;
use std::rc::Rc;
use std::env;

fn main() {
    env_logger::init().unwrap();

    let nets: Vec<String> = vec!("alexnet".to_string(), "overfeat".to_string(), "vgg".to_string());
    if let Some(net) = env::args().nth(1) {
        if nets.contains(&net) {
            println!("Executing Model: {:?}", net);
            if net == "alexnet".to_string() {
                bench_alexnet();
            } else if net == "overfeat".to_string() {
                bench_overfeat();
            } else if net == "vgg".to_string() {
                bench_vgg_a();
            }
        } else {
            println!("Sorry, no model found with name '{:?}'. Valid options: {:?}", net, nets);
        }
    } else {
        println!("No `net` argument specified. Default: `alexnet`. Valid options: {:?}", nets);
        bench_alexnet();
    }
}

#[cfg(feature = "native")]
#[allow(dead_code)]
fn native_backend() -> Rc<Backend<Native>> {
    let framework = Native::new();
    let hardwares = &framework.hardwares().to_vec();
    let backend_config = BackendConfig::new(framework, hardwares);
    Rc::new(Backend::new(backend_config).unwrap())
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn cuda_backend() -> Rc<Backend<Cuda>> {
    let framework = Cuda::new();
    let hardwares = &framework.hardwares()[0..1].to_vec();
    println!("Device: {:?}/{}", hardwares[0].hardware_type().unwrap(), hardwares[0].name().unwrap());
    let backend_config = BackendConfig::new(framework, hardwares);
    Rc::new(Backend::new(backend_config).unwrap())
}

#[cfg(feature = "opencl")]
#[allow(dead_code)]
fn opencl_backend() -> Rc<Backend<OpenCL>> {
    let framework = OpenCL::new();
    let hardwares = &framework.hardwares()[1..2].to_vec();
    let backend_config = BackendConfig::new(framework, hardwares);
    Rc::new(Backend::new(backend_config).unwrap())
}

#[inline(never)]
fn bench_profile<F: FnMut() -> ()>(
    name: &str,
    mut bench_func: F,
    times: usize)
{
    println!("Running benchmark {}", name);
    println!("----------");
    for _ in 0..10 {
        bench_func();
    }
    let average_time = timeit_loops!(times, {
        bench_func();
    });
    println!("----------");
    println!("Average time {}", autoscale_time(average_time));
    println!("");
}

fn autoscale_time(sec: f64) -> String {
    let (div, unit_str) = get_time_scale(sec);
    format!("{:.5} {}", sec / div, unit_str)
}

fn scale_time(sec: f64, unit: &str) -> String {
    // let (div, unit_str) = get_time_scale(sec);
    let div = match unit {
        "s"  => 1.0,
        "ms" => 0.001,
        "µs" => 0.000_001,
        "ns" => 0.000_000_001,
        _ => panic!()
    };
    format!("{:.5} {}", sec / div, unit)
}

// get fitting order of magnitude for a time measurement
fn get_time_scale<'a>(sec: f64) -> (f64, &'a str) {
    if sec > 1.0 {
        (1.0, "s")
    } else if sec > 0.001 {
        (0.001, "ms")
    } else if sec > 0.000_001 {
        (0.000_001, "µs")
    } else {
        (0.000_000_001, "ns")
    }
}

#[cfg(feature="native")]
fn bench_alexnet() {
    println!("Examples run only with CUDA support at the moment, because of missing native convolution implementation for the Collenchyma NN Plugin.");
    println!("Try running with `cargo run --no-default-features --features cuda --example benchmarks alexnet`.");
}
#[cfg(all(feature="cuda", not(feature="native")))]
fn bench_alexnet() {
    let mut cfg = SequentialConfig::default();
    cfg.add_input("data", &vec![128, 3, 224, 224]);

    let conv1_layer_cfg = ConvolutionConfig { num_output: 64, filter_shape: vec![11], padding: vec![2], stride: vec![4] };
    cfg.add_layer(LayerConfig::new("conv1", conv1_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv1/relu", LayerType::ReLU));
    let pool1_layer_cfg = PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![3], stride: vec![2], padding: vec![0] };
    cfg.add_layer(LayerConfig::new("pool1", pool1_layer_cfg));

    let conv2_layer_cfg = ConvolutionConfig { num_output: 192, filter_shape: vec![5], padding: vec![2], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv2", conv2_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv2/relu", LayerType::ReLU));
    let pool2_layer_cfg = PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![3], stride: vec![2], padding: vec![0] };
    cfg.add_layer(LayerConfig::new("pool2", pool2_layer_cfg));

    let conv3_layer_cfg = ConvolutionConfig { num_output: 384, filter_shape: vec![3], padding: vec![1], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv3", conv3_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv3/relu", LayerType::ReLU));

    let conv4_layer_cfg = ConvolutionConfig { num_output: 256, filter_shape: vec![3], padding: vec![1], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv4", conv4_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv4/relu", LayerType::ReLU));

    let conv5_layer_cfg = ConvolutionConfig { num_output: 256, filter_shape: vec![3], padding: vec![1], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv5", conv5_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv5/relu", LayerType::ReLU));
    let pool3_layer_cfg = PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![3], stride: vec![2], padding: vec![0] };
    cfg.add_layer(LayerConfig::new("pool3", pool3_layer_cfg));

    cfg.add_layer(LayerConfig::new("fc1", LinearConfig { output_size: 4096 }));
    cfg.add_layer(LayerConfig::new("fc2", LinearConfig { output_size: 4096 }));
    cfg.add_layer(LayerConfig::new("fc3", LinearConfig { output_size: 1000 }));

    let backend = cuda_backend();
    // let native_backend = native_backend();
    let mut network = Layer::from_config(backend.clone(), &LayerConfig::new("alexnet", LayerType::Sequential(cfg)));

    {
        let func = || {
            let forward_time = timeit_loops!(1, {
                {
                    let inp = SharedTensor::<f32>::new(backend.device(), &vec![128, 3, 224, 224]).unwrap();

                    let inp_lock = Arc::new(RwLock::new(inp));
                    network.forward(&[inp_lock.clone()]);
                }
            });
            println!("Forward step: {}", scale_time(forward_time, "ms"));
        };
        { bench_profile("alexnet_forward", func, 10); }
    }
    {
        let func = || {
            let backward_time = timeit_loops!(1, {
                {
                    network.backward_input(&[]);
                }
            });
            println!("backward input step: {}", scale_time(backward_time, "ms"));
        };
        { bench_profile("alexnet_backward_input", func, 10); }
    }
    {
        let func = || {
            let backward_time = timeit_loops!(1, {
                {
                    network.backward_parameters();
                }
            });
            println!("backward parameters step: {}", scale_time(backward_time, "ms"));
        };
        { bench_profile("alexnet_backward_parameters", func, 10); }
    }
}

#[cfg(feature="native")]
fn bench_overfeat() {
    println!("Examples run only with CUDA support at the moment, because of missing native convolution implementation for the Collenchyma NN Plugin.");
    println!("Try running with `cargo run --no-default-features --features cuda --example benchmarks overfeat`.");
}
#[cfg(all(feature="cuda", not(feature="native")))]
fn bench_overfeat() {
    let mut cfg = SequentialConfig::default();
    cfg.add_input("data", &vec![128, 3, 231, 231]);

    let conv1_layer_cfg = ConvolutionConfig { num_output: 96, filter_shape: vec![11], padding: vec![0], stride: vec![4] };
    cfg.add_layer(LayerConfig::new("conv1", conv1_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv1/relu", LayerType::ReLU));
    let pool1_layer_cfg = PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![2], stride: vec![2], padding: vec![0] };
    cfg.add_layer(LayerConfig::new("pool1", pool1_layer_cfg));

    let conv2_layer_cfg = ConvolutionConfig { num_output: 256, filter_shape: vec![5], padding: vec![0], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv2", conv2_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv2/relu", LayerType::ReLU));
    let pool2_layer_cfg = PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![2], stride: vec![2], padding: vec![0] };
    cfg.add_layer(LayerConfig::new("pool2", pool2_layer_cfg));

    let conv3_layer_cfg = ConvolutionConfig { num_output: 512, filter_shape: vec![3], padding: vec![1], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv3", conv3_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv3/relu", LayerType::ReLU));

    let conv4_layer_cfg = ConvolutionConfig { num_output: 1024, filter_shape: vec![3], padding: vec![1], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv4", conv4_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv4/relu", LayerType::ReLU));

    let conv5_layer_cfg = ConvolutionConfig { num_output: 1024, filter_shape: vec![3], padding: vec![1], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv5", conv5_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv5/relu", LayerType::ReLU));
    let pool5_layer_cfg = PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![2], stride: vec![2], padding: vec![0] };
    cfg.add_layer(LayerConfig::new("pool5", pool5_layer_cfg));

    cfg.add_layer(LayerConfig::new("fc1", LinearConfig { output_size: 3072 }));
    cfg.add_layer(LayerConfig::new("fc2", LinearConfig { output_size: 4096 }));
    cfg.add_layer(LayerConfig::new("fc3", LinearConfig { output_size: 1000 }));

    let backend = cuda_backend();
    // let native_backend = native_backend();
    let mut network = Layer::from_config(backend.clone(), &LayerConfig::new("overfeat", LayerType::Sequential(cfg)));

    {
        let func = || {
            let forward_time = timeit_loops!(1, {
                {
                    let inp = SharedTensor::<f32>::new(backend.device(), &vec![128, 3, 231, 231]).unwrap();

                    let inp_lock = Arc::new(RwLock::new(inp));
                    network.forward(&[inp_lock.clone()]);
                }
            });
            println!("Forward step: {}", scale_time(forward_time, "ms"));
        };
        { bench_profile("overfeat_forward", func, 10); }
    }
    {
        let func = || {
            let backward_time = timeit_loops!(1, {
                {
                    network.backward_input(&[]);
                }
            });
            println!("backward input step: {}", scale_time(backward_time, "ms"));
        };
        { bench_profile("overfeat_backward_input", func, 10); }
    }
    {
        let func = || {
            let backward_time = timeit_loops!(1, {
                {
                    network.backward_parameters();
                }
            });
            println!("backward parameters step: {}", scale_time(backward_time, "ms"));
        };
        { bench_profile("overfeat_backward_parameters", func, 10); }
    }
}

#[cfg(feature="native")]
fn bench_vgg_a() {
    println!("Examples run only with CUDA support at the moment, because of missing native convolution implementation for the Collenchyma NN Plugin.");
    println!("Try running with `cargo run --no-default-features --features cuda --example benchmarks vgg`.");
}
#[cfg(all(feature="cuda", not(feature="native")))]
fn bench_vgg_a() {
    let mut cfg = SequentialConfig::default();
    cfg.add_input("data", &vec![64, 3, 224, 224]);

    let conv1_layer_cfg = ConvolutionConfig { num_output: 64, filter_shape: vec![3], padding: vec![1], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv1", conv1_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv1/relu", LayerType::ReLU));
    let pool1_layer_cfg = PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![2], stride: vec![2], padding: vec![0] };
    cfg.add_layer(LayerConfig::new("pool1", pool1_layer_cfg));

    let conv2_layer_cfg = ConvolutionConfig { num_output: 128, filter_shape: vec![3], padding: vec![1], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv2", conv2_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv2/relu", LayerType::ReLU));
    let pool2_layer_cfg = PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![2], stride: vec![2], padding: vec![0] };
    cfg.add_layer(LayerConfig::new("pool2", pool2_layer_cfg));

    let conv3_layer_cfg = ConvolutionConfig { num_output: 256, filter_shape: vec![3], padding: vec![1], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv3", conv3_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv3/relu", LayerType::ReLU));

    let conv4_layer_cfg = ConvolutionConfig { num_output: 256, filter_shape: vec![3], padding: vec![1], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv4", conv4_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv4/relu", LayerType::ReLU));
    let pool3_layer_cfg = PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![2], stride: vec![2], padding: vec![0] };
    cfg.add_layer(LayerConfig::new("pool3", pool3_layer_cfg));

    let conv5_layer_cfg = ConvolutionConfig { num_output: 512, filter_shape: vec![3], padding: vec![1], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv5", conv5_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv5/relu", LayerType::ReLU));

    let conv6_layer_cfg = ConvolutionConfig { num_output: 512, filter_shape: vec![3], padding: vec![1], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv6", conv6_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv6/relu", LayerType::ReLU));
    let pool4_layer_cfg = PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![2], stride: vec![2], padding: vec![0] };
    cfg.add_layer(LayerConfig::new("pool4", pool4_layer_cfg));

    let conv7_layer_cfg = ConvolutionConfig { num_output: 512, filter_shape: vec![3], padding: vec![1], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv7", conv7_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv7/relu", LayerType::ReLU));

    let conv8_layer_cfg = ConvolutionConfig { num_output: 512, filter_shape: vec![3], padding: vec![1], stride: vec![1] };
    cfg.add_layer(LayerConfig::new("conv8", conv8_layer_cfg));
    cfg.add_layer(LayerConfig::new("conv8/relu", LayerType::ReLU));
    let pool5_layer_cfg = PoolingConfig { mode: PoolingMode::Max, filter_shape: vec![2], stride: vec![2], padding: vec![0] };
    cfg.add_layer(LayerConfig::new("pool5", pool5_layer_cfg));
    cfg.add_layer(LayerConfig::new("fc1", LinearConfig { output_size: 4096 }));
    cfg.add_layer(LayerConfig::new("fc2", LinearConfig { output_size: 4096 }));
    cfg.add_layer(LayerConfig::new("fc3", LinearConfig { output_size: 1000 }));

    let backend = cuda_backend();
    // let native_backend = native_backend();
    let mut network = Layer::from_config(backend.clone(), &LayerConfig::new("vgg_a", LayerType::Sequential(cfg)));

    {
        let func = || {
            let forward_time = timeit_loops!(1, {
                {
                    let inp = SharedTensor::<f32>::new(backend.device(), &vec![64, 3, 224, 224]).unwrap();

                    let inp_lock = Arc::new(RwLock::new(inp));
                    network.forward(&[inp_lock.clone()]);
                }
            });
            println!("Forward step: {}", scale_time(forward_time, "ms"));
        };
        { bench_profile("overfeat_forward", func, 10); }
    }
    {
        let func = || {
            let backward_time = timeit_loops!(1, {
                {
                    network.backward_input(&[]);
                }
            });
            println!("backward input step: {}", scale_time(backward_time, "ms"));
        };
        { bench_profile("overfeat_backward_input", func, 10); }
    }
    {
        let func = || {
            let backward_time = timeit_loops!(1, {
                {
                    network.backward_parameters();
                }
            });
            println!("backward parameters step: {}", scale_time(backward_time, "ms"));
        };
        { bench_profile("overfeat_backward_parameters", func, 10); }
    }
}
