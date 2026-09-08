//! Constant logits have exactly known probabilities, so a repeated resident
//! dispatch exposes an LDS read/write race with no model, no random data and
//! no reference implementation to disagree with.
use krea2_numerics::from_f32;
use krea2_ops::{config, Args, Ops, Pool};

fn usable() -> bool {
    let compiler = loom::compiler(None);
    let found = std::path::Path::new(&compiler).exists()
        || std::env::var_os("PATH").is_some_and(|path| {
            std::env::split_paths(&path).any(|entry| entry.join(&compiler).exists())
        });
    found && hrx::try_device().is_ok()
}

#[test]
fn the_softmax_is_exact_and_stays_exact_over_repeated_dispatches() {
    if !usable() {
        return;
    }
    let ops = Ops::new(Pool::new());
    for tokens in [256usize, 33, 257, 1024] {
        for causal in [false, true] {
            let rows = 8 * tokens;
            let count = rows * tokens;
            // A 256-token tile is the one the resident kernels reuse most, so
            // it gets the long repeat.
            let repeats = if tokens == 256 { 256 } else { 32 };

            let scores = ops.pool().scratch(count * 4).expect("scores");
            hrx::device().write(scores.ptr(), &vec![20f32; count]).expect("upload");
            let out = ops.tensor(rows, tokens).expect("probabilities");

            let expected: Vec<u16> = (0..count)
                .map(|index| {
                    let (row, column) = (index / tokens, index % tokens);
                    let valid = if causal { row % tokens + 1 } else { tokens };
                    from_f32(if column < valid { 1.0 / valid as f32 } else { 0.0 })
                })
                .collect();

            let mut args = Args::new();
            args.i32(rows as i32).ptr(scores.ptr()).ptr(out.ptr());
            let name = if causal { "softmax_causal" } else { "softmax" };
            for repeat in 0..repeats {
                ops.launch(
                    name,
                    config(&[("xsize", count), ("tokens", tokens)]),
                    &args,
                    rows,
                    1,
                    256,
                )
                .expect("launch");
                let actual = out.download().expect("download");
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
