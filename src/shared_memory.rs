//!
//!
//! This is quite unimportant and might be refactored soon.
//!
//! See [Issue #22][issue] for more informations.
//! [issue]: https://github.com/autumnai/leaf/issues/22
use std::sync::{Arc, RwLock};
use phloem::Blob;

/// shared Lock used for our memory blobs
pub type ArcLock<T> = Arc<RwLock<T>>;
/// Blob allocated on the heap via Box
pub type HeapBlob = Box<Blob<f32>>;

/// ...
pub fn new_shared_heapblob() -> ArcLock<HeapBlob> {
    Arc::new(RwLock::new(Box::new(Blob::new())))
}
