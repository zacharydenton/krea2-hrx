//! What a dispatch costs the host, which bounds what recording a fixed
//! sequence could ever save. Queues N launches without synchronizing, so the
//! wall time is submission cost rather than GPU time.
use hrx::Stream;
use kernels::{auxiliary_kernel, config, Scalars};

fn main() {
    let mut stream = Stream::open().expect("a stream");
    let count = 256usize;
    let grid = (1u32, 1u32);
    let kernel = auxiliary_kernel(
        &stream,
        "unary_one",
        &config([("count_b", count as u64)]),
        grid,
        None,
    )
    .expect("unary_one");
    let x = stream.allocate(count * 2).unwrap();
    let y = stream.allocate(count * 2).unwrap();
    let constants = Scalars::new().index(count).pack("unary_one", &kernel).unwrap();
    let bindings = [x.binding(), y.binding()];

    for round in 0..3 {
        let launches = 10_000;
        stream.synchronize().unwrap();
        let began = std::time::Instant::now();
        for _ in 0..launches {
            unsafe { stream.dispatch(&kernel, [1, 1, 1], [256, 1, 1], &constants, &bindings) }
                .unwrap();
        }
        let queued = began.elapsed();
        stream.synchronize().unwrap();
        let drained = began.elapsed();
        println!(
            "round {round}: {:.2} us/dispatch queued, {:.2} us/dispatch drained",
            queued.as_secs_f64() * 1e6 / launches as f64,
            drained.as_secs_f64() * 1e6 / launches as f64,
        );
    }
}
