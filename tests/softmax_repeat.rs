//! Constant logits have exactly known probabilities, so a repeated resident
//! dispatch exposes an LDS read/write race with no model, no random data and
//! no reference implementation to disagree with.
use hrx::BufferPool;
use krea2::numerics::from_f32;
use krea2::ops::{config, Ops, Scalars};

#[test]
#[ignore = "requires gfx1151 and the provisioned HRX runtime"]
fn the_softmax_is_exact_and_stays_exact_over_repeated_dispatches() {
    let mut stream = hrx::Stream::open().expect("a stream");
    let ops = Ops::new(BufferPool::new());
    for tokens in [256usize, 33, 257, 1024] {
        for causal in [false, true] {
            let rows = 8 * tokens;
            let count = rows * tokens;
            // A 256-token tile is the one the resident kernels reuse most, so
            // it gets the long repeat.
            let repeats = if tokens == 256 { 256 } else { 32 };

            let scores = ops.pool().acquire(&stream, count * 4).expect("scores");
            stream
                .upload(scores.binding(), bytemuck::cast_slice(&vec![20f32; count]))
                .expect("upload");
            let out = ops.tensor(&stream, rows, tokens).expect("probabilities");

            let expected: Vec<u16> = (0..count)
                .map(|index| {
                    let (row, column) = (index / tokens, index % tokens);
                    let valid = if causal { row % tokens + 1 } else { tokens };
                    from_f32(if column < valid { 1.0 / valid as f32 } else { 0.0 })
                })
                .collect();

            let scalars = Scalars::new().index(rows);
            let name = if causal { "softmax_causal" } else { "softmax" };
            for repeat in 0..repeats {
                let bindings = [scores.binding(), out.binding().expect("a binding")];
                unsafe {
                    ops.launch(
                        &stream,
                        name,
                        config(&[("xsize", count), ("tokens", tokens)]),
                        &scalars,
                        &bindings,
                        rows,
                        1,
                        256,
                    )
                }
                .expect("launch");
                let actual = out.download(&mut stream).expect("download");
                if let Some(index) = (0..count).find(|&index| actual[index] != expected[index])
                {
                    panic!(
                        "{name} tokens={tokens} repeat={repeat} row={} col={} \
                         expected={:#06x} actual={:#06x}",
                        index / tokens,
                        index % tokens,
                        expected[index],
                        actual[index]
                    );
                }
            }
        }
    }
}
