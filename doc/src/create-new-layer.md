# Create a new Layer

A layer in Leaf can implement any behavior as long as it takes an input and
produces an output. There are a lot of layers, that would prove valuable but,
are not yet implemented. This chapter shows how straight forward it is to add
a new layer to Leaf.

A not exclusive list of steps to take in order to implement a new layer.
The Rust compiler is also very helpful with pointing out the necessary steps for
implementing a new layer struct. It might be helpful to copy a file of an
already existing layer and start from there.

1. Decide to which of the [five types](./layers.html#What&#32;can&#32;Layers&#32;do?)
the new layer belongs. This decides under which directory to put the layer
implementation in the Leaf project.

2. Create the `Layer` worker struct.

3. Expose the `Layer` worker struct in the `mod.rs` of the layer type directory.

4. Expose the `Layer` worker struct in the `mod.rs` of the `/layers` directory.

5. Implement `ILayer` and its trait boundaries for the new `Layer` worker struct.

6. Add the new layer to the `LayerType` in `layer.rs` and add the matching
for `.support_in_place` and `.worker_from_config`.

7. If the new layer relies on a collenchyma operation, also add the collenchyma
trait boundary.

8. Add documentation and serialization to the new layer.

9. (optional) Depending on how complex the layer is, you might also add tests and more
advanced implementations for its `.from_config`, `.reshape` or other helper
methods.
