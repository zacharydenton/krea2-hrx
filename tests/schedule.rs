//! Complete Turbo/Raw scheduler grid, compared against an immutable external oracle.
use krea2::pipeline::schedule;
#[test]
fn all_sigma_grids_match_the_recorded_diffusers_bits() {
    let bytes = include_bytes!("fixtures/sigmas.f32le");
    let mut expected = bytes.chunks_exact(4);
    for tokens in [0usize, 16, 256, 589, 4096, 6400, 16384] {
        let mu = if tokens == 0 { 1.15 } else { schedule::dynamic_mu(tokens) };
        for steps in 1..=100 {
            for step in 0..=steps {
                let want = u32::from_le_bytes(expected.next().unwrap().try_into().unwrap());
                assert_eq!(
                    schedule::sigma(step, steps, mu).to_bits(),
                    want,
                    "tokens={tokens}, steps={steps}, step={step}"
                );
            }
        }
    }
    assert!(expected.next().is_none());
}
