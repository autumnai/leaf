//! Provides common utility functions
use std::sync::{Arc, RwLock};
use co::prelude::*;
use coblas::plugin::*;
use conn;
use num::traits::{NumCast, cast};

/// Shared Lock used for our tensors
pub type ArcLock<T> = Arc<RwLock<T>>;

/// Create a simple native backend.
///
/// This is handy when you need to sync data to host memory to read/write it.
pub fn native_backend() -> Backend<Native> {
    let framework = Native::new();
    let hardwares = &framework.hardwares().to_vec();
    let backend_config = BackendConfig::new(framework, hardwares);
    Backend::new(backend_config).unwrap()
}

/// Write into a native Collenchyma Memory.
pub fn write_to_memory<T: NumCast + ::std::marker::Copy>(mem: &mut MemoryType, data: &[T]) {
    write_to_memory_offset(mem, data, 0);
}

/// Write into a native Collenchyma Memory with a offset.
pub fn write_to_memory_offset<T: NumCast + ::std::marker::Copy>(mem: &mut MemoryType, data: &[T], offset: usize) {
    match mem {
        &mut MemoryType::Native(ref mut mem) => {
            let mut mem_buffer = mem.as_mut_slice::<f32>();
            for (index, datum) in data.iter().enumerate() {
                // mem_buffer[index + offset] = *datum;
                mem_buffer[index + offset] = cast(*datum).unwrap();
            }
        },
        #[cfg(any(feature = "opencl", feature = "cuda"))]
        _ => {}
    }
}

/// Write the `i`th sample of a batch into a SharedTensor.
///
/// The size of a single sample is infered through
/// the first dimension of the SharedTensor, which
/// is asumed to be the batchsize.
///
/// Allocates memory on a Native Backend if neccessary.
pub fn write_batch_sample<T: NumCast + ::std::marker::Copy>(tensor: &mut SharedTensor<f32>, data: &[T], i: usize) {
    let native_backend = native_backend();

    let batch_size = tensor.desc().size();
    let sample_size = batch_size / tensor.desc()[0];

    let _ = tensor.add_device(native_backend.device());
    tensor.sync(native_backend.device()).unwrap();
    write_to_memory_offset(tensor.get_mut(native_backend.device()).unwrap(), &data, i * sample_size);
}

/// Create a Collenchyma SharedTensor for a scalar value.
pub fn native_scalar<T: NumCast + ::std::marker::Copy>(scalar: T) -> SharedTensor<T> {
    let native = native_backend();
    let mut shared_scalar = SharedTensor::<T>::new(native.device(), &vec![1]).unwrap();
    write_to_memory(shared_scalar.get_mut(native.device()).unwrap(), &[scalar]);

    shared_scalar
}

/// Casts a Vec<usize> to as Vec<i32>
pub fn cast_vec_usize_to_i32(input: Vec<usize>) -> Vec<i32> {
    let mut out = Vec::new();
    for i in input.iter() {
        out.push(*i as i32);
    }
    out
}

/// Extends IBlas with Axpby
pub trait Axpby<F> : Axpy<F> + Scal<F> {
    /// Performs the operation y := a*x + b*y .
    ///
    /// Consists of a scal(b, y) followed by a axpby(a,x,y).
    fn axpby(&self, a: &mut SharedTensor<F>, x: &mut SharedTensor<F>, b: &mut SharedTensor<F>, y: &mut SharedTensor<F>) -> Result<(), ::co::error::Error> {
        try!(self.scal(b, y));
        try!(self.axpy(a, x, y));
        Ok(())
    }

    /// Performs the operation y := a*x + b*y .
    ///
    /// Consists of a scal(b, y) followed by a axpby(a,x,y).
    fn axpby_plain(&self, a: &SharedTensor<F>, x: &SharedTensor<F>, b: &SharedTensor<F>, y: &mut SharedTensor<F>) -> Result<(), ::co::error::Error> {
        try!(self.scal_plain(b, y));
        try!(self.axpy_plain(a, x, y));
        Ok(())
    }
}

impl<T: Axpy<f32> + Scal<f32>> Axpby<f32> for T {}

/// Encapsulates all traits required by Solvers.
// pub trait SolverOps<F> : Axpby<F> + Dot<F> + Copy<F> {}
//
// impl<T: Axpby<f32> + Dot<f32> + Copy<f32>> SolverOps<f32> for T {}
pub trait SolverOps<F> : LayerOps<F> + Axpby<F> + Dot<F> + Copy<F> {}

impl<T: LayerOps<f32> + Axpby<f32> + Dot<f32> + Copy<f32>> SolverOps<f32> for T {}

/// Encapsulates all traits used in Layers.
#[cfg(all(feature="cuda", not(feature="native")))]
pub trait LayerOps<F> : conn::Convolution<F>
                      + conn::Pooling<F>
                      + conn::Relu<F> + conn::ReluPointwise<F>
                      + conn::Sigmoid<F> + conn::SigmoidPointwise<F>
                      + conn::Tanh<F> + conn::TanhPointwise<F>
                      + conn::Softmax<F> + conn::LogSoftmax<F>
                      + Gemm<F> {}
#[cfg(feature="native")]
/// Encapsulates all traits used in Layers.
pub trait LayerOps<F> : conn::Relu<F>
                      + conn::Sigmoid<F>
                      + conn::Tanh<F>
                      + conn::Softmax<F> + conn::LogSoftmax<F>
                      + Gemm<F> {}

#[cfg(all(feature="cuda", not(feature="native")))]
impl<T: conn::Convolution<f32>
      + conn::Pooling<f32>
      + conn::Relu<f32> + conn::ReluPointwise<f32>
      + conn::Sigmoid<f32> + conn::SigmoidPointwise<f32>
      + conn::Tanh<f32> + conn::TanhPointwise<f32>
      + conn::Softmax<f32> + conn::LogSoftmax<f32>
      + Gemm<f32>> LayerOps<f32> for T {}
#[cfg(feature="native")]
impl<T: conn::Relu<f32>
      + conn::Sigmoid<f32>
      + conn::Tanh<f32>
      + conn::Softmax<f32> + conn::LogSoftmax<f32>
      + Gemm<f32>> LayerOps<f32> for T {}
