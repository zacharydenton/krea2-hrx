// Cargo propagates a dependency's link libraries but not its link arguments,
// so the rpath the hrx crate sets for itself has to be repeated here.
fn main() {
    krea2_build_support::emit_rpath();
}
