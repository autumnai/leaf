//! Provides common utility functions
use std::sync::{Arc, RwLock};
use co::backend::{Backend, BackendConfig};
use co::framework::IFramework;
use co::frameworks::Native;
use co::memory::MemoryType;
use co::tensor::SharedTensor;
use co::plugin::numeric_helpers::*;
use coblas::plugin::*;
use conn;

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
pub fn write_to_memory<T: ::std::marker::Copy>(mem: &mut MemoryType, data: &[T]) {
    match mem {
        &mut MemoryType::Native(ref mut mem) => {
            let mut mem_buffer = mem.as_mut_slice::<T>();
            for (index, datum) in data.iter().enumerate() {
                mem_buffer[index] = *datum;
            }
        },
        #[cfg(any(feature = "opencl", feature = "cuda"))]
        _ => {}
    }
}

/// Create a Collenchyma SharedTensor for a scalar value.
pub fn native_scalar<T: ::std::marker::Copy>(scalar: T) -> SharedTensor<T> {
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
pub trait Axpby<F: Float> : Axpy<F> + Scal<F> {
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
pub trait SolverOps<F: Float> : Axpby<F> + Dot<F> + Copy<F> {}

impl<T: Axpby<f32> + Dot<f32> + Copy<f32>> SolverOps<f32> for T {}

/// Encapsulates all traits used in Layers.
pub trait LayerOps<F: Float> : conn::Convolution<F> + conn::Pooling<F> + conn::Relu<F> + conn::Sigmoid<F> + conn::Softmax<F> + conn::LogSoftmax<F>
                             + Gemm<F> {}

impl<T: conn::Convolution<f32> + conn::Pooling<f32> + conn::Relu<f32> + conn::Sigmoid<f32> + conn::Softmax<f32> + conn::LogSoftmax<f32>
      + Gemm<f32>> LayerOps<f32> for T {}

// pub trait LayerOps<F: Float> : conn::Relu<F> + conn::Sigmoid<F> + conn::Softmax<F> + conn::LogSoftmax<F>
//                              + Gemm<F> {}
//
// impl<T: conn::Relu<f32> + conn::Sigmoid<f32> + conn::Softmax<f32> + conn::LogSoftmax<f32>
//       + Gemm<f32>> LayerOps<f32> for T {}
