extern crate capnpc;

fn main() {
    ::capnpc::compile("capnp", &["capnp/leaf.capnp"]).unwrap();
}
