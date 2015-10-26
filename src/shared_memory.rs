use std::sync::{Arc, RwLock};
use blob::Blob;

/// shared Lock used for our memory blobs
pub type ArcLock<T> = Arc<RwLock<T>>;
/// Blob allocated on the heap via Box
pub type HeapBlob = Box<Blob<f32>>;
