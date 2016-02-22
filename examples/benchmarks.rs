#[macro_use]
extern crate timeit;
extern crate collenchyma as co;
extern crate leaf;

use co::prelude::*;

use std::sync::{Arc, RwLock};
use leaf::layers::*;
use leaf::layer::*;
use leaf::network::*;
use std::rc::Rc;

fn main() {
    // bench_mnsit_forward();
    bench_alexnet();
    bench_overfeat();
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


// #[bench]
#[allow(dead_code)]
#[cfg(feature = "cuda")]
fn bench_mnsit_forward() {
    let mut cfg = NetworkConfig::default();
    // set up input
    cfg.add_input("in", &vec![1, 30, 30]);
    cfg.add_input("label", &vec![1, 1, 10]);
    // set up sigmoid
    let mut sig_cfg = LayerConfig::new("sig", LayerType::Sigmoid);
    sig_cfg.add_input("in");
    sig_cfg.add_output("sig_out");
    cfg.add_layer(sig_cfg);

    let fc_layer_cfg = LinearConfig {
        output_size: 10,
    };
    let mut fc_cfg = LayerConfig::new("fully_connected", LayerType::Linear(fc_layer_cfg));
    fc_cfg.add_input("sig_out");
    fc_cfg.add_output("fc_out");
    cfg.add_layer(fc_cfg);
    // // set up softmax_loss
    // let mut loss_cfg = LayerConfig::new("loss", LayerType::SoftmaxLoss);
    // loss_cfg.add_input("fc_out");
    // loss_cfg.add_input("label");
    // cfg.add_layer(loss_cfg);

    let backend = cuda_backend();
    let native_backend = native_backend();
    let mut network = Network::from_config(backend.clone(), &cfg);
    let loss = &mut 0f32;

    let func = || {
        let forward_time = timeit_loops!(1, {
            let inp = SharedTensor::<f32>::new(backend.device(), &vec![1, 30, 30]).unwrap();
            let label = SharedTensor::<f32>::new(native_backend.device(), &vec![1, 1, 10]).unwrap();

            let inp_lock = Arc::new(RwLock::new(inp));
            let label_lock = Arc::new(RwLock::new(label));

            network.forward(&[inp_lock, label_lock], loss);
        });
        println!("Forward step: {}", scale_time(forward_time, "ms"));
    };
    { bench_profile("mnist_forward", func, 10); }
}

#[cfg(not(feature = "cuda"))]
fn bench_alexnet() {}
#[cfg(feature = "cuda")]
fn bench_alexnet() {
    let mut cfg = NetworkConfig::default();
    // Layer: data
    cfg.add_input("data", &vec![128, 3, 224, 224]);
    // Layer: conv1
    let conv1_layer_cfg = ConvolutionConfig {
        num_output: 64,
        filter_shape: vec![11],
        padding: vec![2],
        stride: vec![4],
        axis: None
    };
    let mut conv1_cfg = LayerConfig::new("conv1", LayerType::Convolution(conv1_layer_cfg));
    conv1_cfg.add_input("data");
    conv1_cfg.add_output("conv1_preac");
    cfg.add_layer(conv1_cfg);
    // Layer: conv1/relu
    let mut conv1_relu_cfg = LayerConfig::new("conv1/relu", LayerType::ReLU);
    conv1_relu_cfg.add_input("conv1_preac");
    conv1_relu_cfg.add_output("conv1_out");
    cfg.add_layer(conv1_relu_cfg);
    // Layer: pool1
    let pool1_layer_cfg = PoolingConfig {
        mode: PoolingMode::Max,
        filter_shape: vec![3],
        stride: vec![2],
        padding: vec![0], // TODO: make optional
    };
    let mut pool1_cfg = LayerConfig::new("pool1", LayerType::Pooling(pool1_layer_cfg));
    pool1_cfg.add_input("conv1_out");
    pool1_cfg.add_output("pool1_out");
    cfg.add_layer(pool1_cfg);
    // Layer: conv2
    let conv2_layer_cfg = ConvolutionConfig {
        num_output: 192,
        filter_shape: vec![5],
        padding: vec![2],
        stride: vec![1],
        axis: None
    };
    let mut conv2_cfg = LayerConfig::new("conv2", LayerType::Convolution(conv2_layer_cfg));
    conv2_cfg.add_input("pool1_out");
    conv2_cfg.add_output("conv2_preac");
    cfg.add_layer(conv2_cfg);
    // Layer: conv2/relu
    let mut conv2_relu_cfg = LayerConfig::new("conv2/relu", LayerType::ReLU);
    conv2_relu_cfg.add_input("conv2_preac");
    conv2_relu_cfg.add_output("conv2_out");
    cfg.add_layer(conv2_relu_cfg);
    // Layer: pool2
    let pool2_layer_cfg = PoolingConfig {
        mode: PoolingMode::Max,
        filter_shape: vec![3],
        stride: vec![2],
        padding: vec![0], // TODO: make optional
    };
    let mut pool2_cfg = LayerConfig::new("pool2", LayerType::Pooling(pool2_layer_cfg));
    pool2_cfg.add_input("conv2_out");
    pool2_cfg.add_output("pool2_out");
    cfg.add_layer(pool2_cfg);
    // Layer: conv3
    let conv3_layer_cfg = ConvolutionConfig {
        num_output: 384,
        filter_shape: vec![3],
        padding: vec![1],
        stride: vec![1],
        axis: None
    };
    let mut conv3_cfg = LayerConfig::new("conv3", LayerType::Convolution(conv3_layer_cfg));
    conv3_cfg.add_input("pool2_out");
    conv3_cfg.add_output("conv3_preac");
    cfg.add_layer(conv3_cfg);
    // Layer: conv3/relu
    let mut conv3_relu_cfg = LayerConfig::new("conv3/relu", LayerType::ReLU);
    conv3_relu_cfg.add_input("conv3_preac");
    conv3_relu_cfg.add_output("conv3_out");
    cfg.add_layer(conv3_relu_cfg);
    // Layer: conv4
    let conv4_layer_cfg = ConvolutionConfig {
        num_output: 256,
        filter_shape: vec![3],
        padding: vec![1],
        stride: vec![1],
        axis: None
    };
    let mut conv4_cfg = LayerConfig::new("conv4", LayerType::Convolution(conv4_layer_cfg));
    conv4_cfg.add_input("conv3_out");
    conv4_cfg.add_output("conv4_preac");
    cfg.add_layer(conv4_cfg);
    // Layer: conv4/relu
    let mut conv4_relu_cfg = LayerConfig::new("conv4/relu", LayerType::ReLU);
    conv4_relu_cfg.add_input("conv4_preac");
    conv4_relu_cfg.add_output("conv4_out");
    cfg.add_layer(conv4_relu_cfg);
    // Layer: conv5
    let conv5_layer_cfg = ConvolutionConfig {
        num_output: 256,
        filter_shape: vec![3],
        padding: vec![1],
        stride: vec![1],
        axis: None
    };
    let mut conv5_cfg = LayerConfig::new("conv5", LayerType::Convolution(conv5_layer_cfg));
    conv5_cfg.add_input("conv4_out");
    conv5_cfg.add_output("conv5_preac");
    cfg.add_layer(conv5_cfg);
    // Layer: conv5/relu
    let mut conv5_relu_cfg = LayerConfig::new("conv5/relu", LayerType::ReLU);
    conv5_relu_cfg.add_input("conv5_preac");
    conv5_relu_cfg.add_output("conv5_out");
    cfg.add_layer(conv5_relu_cfg);
    // Layer: pool3
    let pool3_layer_cfg = PoolingConfig {
        mode: PoolingMode::Max,
        filter_shape: vec![3],
        stride: vec![2],
        padding: vec![0], // TODO: make optional
    };
    let mut pool3_cfg = LayerConfig::new("pool3", LayerType::Pooling(pool3_layer_cfg));
    pool3_cfg.add_input("conv5_out");
    pool3_cfg.add_output("pool3_out");
    cfg.add_layer(pool3_cfg);
    // Layer: fc1
    let fc1_layer_cfg = LinearConfig {
        output_size: 4096,
    };
    let mut fc1_cfg = LayerConfig::new("fc1", LayerType::Linear(fc1_layer_cfg));
    fc1_cfg.add_input("pool3_out");
    fc1_cfg.add_output("fc1_out");
    cfg.add_layer(fc1_cfg);
    // Layer: fc2
    let fc2_layer_cfg = LinearConfig {
        output_size: 4096,
    };
    let mut fc2_cfg = LayerConfig::new("fc2", LayerType::Linear(fc2_layer_cfg));
    fc2_cfg.add_input("fc1_out");
    fc2_cfg.add_output("fc2_out");
    cfg.add_layer(fc2_cfg);
    // Layer: fc3
    let fc3_layer_cfg = LinearConfig {
        output_size: 1000,
    };
    let mut fc3_cfg = LayerConfig::new("fc3", LayerType::Linear(fc3_layer_cfg));
    fc3_cfg.add_input("fc2_out");
    fc3_cfg.add_output("fc3_out");
    cfg.add_layer(fc3_cfg);

    let backend = cuda_backend();
    // let native_backend = native_backend();
    let mut network = Network::from_config(backend.clone(), &cfg);

    {
        let func = || {
            let forward_time = timeit_loops!(1, {
                {
                    let loss = &mut 0f32;
                    let inp = SharedTensor::<f32>::new(backend.device(), &vec![128, 3, 224, 224]).unwrap();

                    let inp_lock = Arc::new(RwLock::new(inp));
                    network.forward(&[inp_lock.clone()], loss);
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
                    network.backward_input();
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

#[cfg(not(feature = "cuda"))]
fn bench_overfeat() {}
#[cfg(feature = "cuda")]
fn bench_overfeat() {
    let mut cfg = NetworkConfig::default();
    // Layer: data
    cfg.add_input("data", &vec![128, 3, 231, 231]);
    // Layer: conv1
    let conv1_layer_cfg = ConvolutionConfig {
        num_output: 96,
        filter_shape: vec![11],
        padding: vec![0],
        stride: vec![4],
        axis: None
    };
    let mut conv1_cfg = LayerConfig::new("conv1", LayerType::Convolution(conv1_layer_cfg));
    conv1_cfg.add_input("data");
    conv1_cfg.add_output("conv1_preac");
    cfg.add_layer(conv1_cfg);
    // Layer: conv1/relu
    let mut conv1_relu_cfg = LayerConfig::new("conv1/relu", LayerType::ReLU);
    conv1_relu_cfg.add_input("conv1_preac");
    conv1_relu_cfg.add_output("conv1_out");
    cfg.add_layer(conv1_relu_cfg);
    // Layer: pool1
    let pool1_layer_cfg = PoolingConfig {
        mode: PoolingMode::Max,
        filter_shape: vec![2],
        stride: vec![2],
        padding: vec![0], // TODO: make optional
    };
    let mut pool1_cfg = LayerConfig::new("pool1", LayerType::Pooling(pool1_layer_cfg));
    pool1_cfg.add_input("conv1_out");
    pool1_cfg.add_output("pool1_out");
    cfg.add_layer(pool1_cfg);
    // Layer: conv2
    let conv2_layer_cfg = ConvolutionConfig {
        num_output: 256,
        filter_shape: vec![5],
        padding: vec![0],
        stride: vec![1],
        axis: None
    };
    let mut conv2_cfg = LayerConfig::new("conv2", LayerType::Convolution(conv2_layer_cfg));
    conv2_cfg.add_input("pool1_out");
    conv2_cfg.add_output("conv2_preac");
    cfg.add_layer(conv2_cfg);
    // Layer: conv2/relu
    let mut conv2_relu_cfg = LayerConfig::new("conv2/relu", LayerType::ReLU);
    conv2_relu_cfg.add_input("conv2_preac");
    conv2_relu_cfg.add_output("conv2_out");
    cfg.add_layer(conv2_relu_cfg);
    // Layer: pool2
    let pool2_layer_cfg = PoolingConfig {
        mode: PoolingMode::Max,
        filter_shape: vec![2],
        stride: vec![2],
        padding: vec![0], // TODO: make optional
    };
    let mut pool2_cfg = LayerConfig::new("pool2", LayerType::Pooling(pool2_layer_cfg));
    pool2_cfg.add_input("conv2_out");
    pool2_cfg.add_output("pool2_out");
    cfg.add_layer(pool2_cfg);
    // Layer: conv3
    let conv3_layer_cfg = ConvolutionConfig {
        num_output: 512,
        filter_shape: vec![3],
        padding: vec![1],
        stride: vec![1],
        axis: None
    };
    let mut conv3_cfg = LayerConfig::new("conv3", LayerType::Convolution(conv3_layer_cfg));
    conv3_cfg.add_input("pool2_out");
    conv3_cfg.add_output("conv3_preac");
    cfg.add_layer(conv3_cfg);
    // Layer: conv3/relu
    let mut conv3_relu_cfg = LayerConfig::new("conv3/relu", LayerType::ReLU);
    conv3_relu_cfg.add_input("conv3_preac");
    conv3_relu_cfg.add_output("conv3_out");
    cfg.add_layer(conv3_relu_cfg);
    // Layer: conv4
    let conv4_layer_cfg = ConvolutionConfig {
        num_output: 1024,
        filter_shape: vec![3],
        padding: vec![1],
        stride: vec![1],
        axis: None
    };
    let mut conv4_cfg = LayerConfig::new("conv4", LayerType::Convolution(conv4_layer_cfg));
    conv4_cfg.add_input("conv3_out");
    conv4_cfg.add_output("conv4_preac");
    cfg.add_layer(conv4_cfg);
    // Layer: conv4/relu
    let mut conv4_relu_cfg = LayerConfig::new("conv4/relu", LayerType::ReLU);
    conv4_relu_cfg.add_input("conv4_preac");
    conv4_relu_cfg.add_output("conv4_out");
    cfg.add_layer(conv4_relu_cfg);
    // Layer: conv5
    let conv5_layer_cfg = ConvolutionConfig {
        num_output: 1024,
        filter_shape: vec![3],
        padding: vec![1],
        stride: vec![1],
        axis: None
    };
    let mut conv5_cfg = LayerConfig::new("conv5", LayerType::Convolution(conv5_layer_cfg));
    conv5_cfg.add_input("conv4_out");
    conv5_cfg.add_output("conv5_preac");
    cfg.add_layer(conv5_cfg);
    // Layer: conv5/relu
    let mut conv5_relu_cfg = LayerConfig::new("conv5/relu", LayerType::ReLU);
    conv5_relu_cfg.add_input("conv5_preac");
    conv5_relu_cfg.add_output("conv5_out");
    cfg.add_layer(conv5_relu_cfg);
    // Layer: pool5
    let pool5_layer_cfg = PoolingConfig {
        mode: PoolingMode::Max,
        filter_shape: vec![2],
        stride: vec![2],
        padding: vec![0], // TODO: make optional
    };
    let mut pool5_cfg = LayerConfig::new("pool5", LayerType::Pooling(pool5_layer_cfg));
    pool5_cfg.add_input("conv5_out");
    pool5_cfg.add_output("pool5_out");
    cfg.add_layer(pool5_cfg);
    // Layer: fc1
    let fc1_layer_cfg = LinearConfig {
        output_size: 3072,
    };
    let mut fc1_cfg = LayerConfig::new("fc1", LayerType::Linear(fc1_layer_cfg));
    fc1_cfg.add_input("pool5_out");
    fc1_cfg.add_output("fc1_out");
    cfg.add_layer(fc1_cfg);
    // Layer: fc2
    let fc2_layer_cfg = LinearConfig {
        output_size: 4096,
    };
    let mut fc2_cfg = LayerConfig::new("fc2", LayerType::Linear(fc2_layer_cfg));
    fc2_cfg.add_input("fc1_out");
    fc2_cfg.add_output("fc2_out");
    cfg.add_layer(fc2_cfg);
    // Layer: fc3
    let fc3_layer_cfg = LinearConfig {
        output_size: 1000,
    };
    let mut fc3_cfg = LayerConfig::new("fc3", LayerType::Linear(fc3_layer_cfg));
    fc3_cfg.add_input("fc2_out");
    fc3_cfg.add_output("fc3_out");
    cfg.add_layer(fc3_cfg);

    let backend = cuda_backend();
    // let native_backend = native_backend();
    let mut network = Network::from_config(backend.clone(), &cfg);

    {
        let func = || {
            let forward_time = timeit_loops!(1, {
                {
                    let loss = &mut 0f32;
                    let inp = SharedTensor::<f32>::new(backend.device(), &vec![128, 3, 231, 231]).unwrap();

                    let inp_lock = Arc::new(RwLock::new(inp));
                    network.forward(&[inp_lock.clone()], loss);
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
                    network.backward_input();
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
