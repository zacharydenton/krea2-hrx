//! Host recording and completed batch costs, with and without the operation cache.
use hrx::Stream;
use krea2::kernels::{cache::PreparedKernels, config, Scalars};

fn main() {
    let mut stream = Stream::open().expect("a stream");
    let prepared = PreparedKernels::default();
    let kernel =
        prepared.get(&stream, "unary_one", config([("count_b", 256)]), (1, 1)).unwrap();
    let x = stream.allocate(512).unwrap();
    let y = stream.allocate(512).unwrap();
    stream.fill(x.binding(), 0).unwrap();
    let constants = Scalars::new().index(256).pack("unary_one", &kernel).unwrap();
    let bindings = [x.binding(), y.binding()];
    for cached in [false, true] {
        let mut queued = Vec::new();
        let mut completed = Vec::new();
        for round in 0..12 {
            let launches = 2048;
            stream.synchronize().unwrap();
            let began = std::time::Instant::now();
            for _ in 0..launches {
                if cached {
                    let kernel = prepared
                        .get(&stream, "unary_one", config([("count_b", 256)]), (1, 1))
                        .unwrap();
                    let constants =
                        Scalars::new().index(256).pack("unary_one", &kernel).unwrap();
                    unsafe {
                        stream.dispatch(&kernel, [1; 3], [256, 1, 1], &constants, &bindings)
                    }
                    .unwrap();
                } else {
                    unsafe {
                        stream.dispatch(&kernel, [1; 3], [256, 1, 1], &constants, &bindings)
                    }
                    .unwrap();
                }
            }
            let host = began.elapsed();
            stream.synchronize().unwrap();
            if round >= 3 {
                queued.push(host.as_secs_f64() * 1e9 / launches as f64);
                completed.push(began.elapsed().as_secs_f64() * 1e9 / launches as f64);
            }
        }
        queued.sort_by(f64::total_cmp);
        completed.sort_by(f64::total_cmp);
        println!(
            "{}: {:.0} ns host/dispatch, {:.0} ns completed/dispatch",
            if cached { "prepared lookup + packing" } else { "direct" },
            queued[4],
            completed[4]
        );
    }
    let mut result = [0u8; 512];
    stream.read_blocking(y.binding(), &mut result).unwrap();
    assert!(result.chunks_exact(2).all(|b| u16::from_le_bytes([b[0], b[1]]) == 0x3f80));
}
