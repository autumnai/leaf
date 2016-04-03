# Create a new Layer

A layer in Leaf can implement any behavior as long as it takes an input and
produces an output. As Leaf is new, there are still many valuable layers that
are not yet implemented. This is why this chapter shows how you can add new
layers to Leaf.

A not exclusive list of steps to take in order to implement a new layer:

> The Rust compiler is also very helpful with pointing out the necessary steps
> for implementing a new layer struct. It might be beneficial to start the
> implementation of a new layer from a copied file of an already existing layer.

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
