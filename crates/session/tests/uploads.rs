//! Weight upload plans exercise contiguous chunks, gathered rows and padded pitches.
use krea2_checkpoint::{Checkpoint, Plan, Segment, Span};
use krea2_session::weights::Weights;

#[test]
#[ignore = "requires gfx1151 and provisioned HRX"]
fn contiguous_and_padded_uploads_preserve_rows_across_chunks() {
    let path =
        std::env::temp_dir().join(format!("krea-upload-{}.safetensors", std::process::id()));
    let rows = (16 << 20) / 4 + 3;
    let payload: Vec<u8> = (0..rows * 4).map(|i| (i % 251) as u8).collect();
    let header = format!(
        r#"{{"w":{{"dtype":"I8","shape":[{rows},4],"data_offsets":[0,{}]}}}}"#,
        payload.len()
    );
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&payload);
    std::fs::write(&path, file).unwrap();
    let checkpoint = Checkpoint::open(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let flat_bytes = payload.len();
    let flat = Span {
        device_offset: 0,
        device_bytes: flat_bytes,
        rows,
        row_bytes: 4,
        device_row_bytes: 4,
        host: vec![],
        segments: vec![Segment { tensor: "w".into(), rows: 0..rows }],
    };
    let padded = Span {
        device_offset: flat_bytes,
        device_bytes: 32,
        rows: 4,
        row_bytes: 4,
        device_row_bytes: 8,
        host: vec![],
        segments: vec![
            Segment { tensor: "w".into(), rows: 3..5 },
            Segment { tensor: "w".into(), rows: 0..2 },
        ],
    };
    let gathered = Span {
        device_offset: flat_bytes + 32,
        device_bytes: 16,
        device_row_bytes: 4,
        ..padded.clone()
    };
    let plan = Plan {
        total_bytes: flat_bytes + 48,
        layers: 1,
        bits: 8,
        spans: [
            ("flat".into(), flat),
            ("padded".into(), padded),
            ("gathered".into(), gathered),
        ]
        .into(),
    };
    let mut stream = hrx::Stream::open().unwrap();
    let weights = Weights::upload(&mut stream, &checkpoint, plan).unwrap();
    let mut actual = vec![0; flat_bytes];
    stream
        .read(weights.view(weights.locate("flat", flat_bytes).unwrap()), &mut actual)
        .unwrap();
    assert_eq!(actual, payload);
    let mut expected = [0; 32];
    for (i, row) in [3, 4, 0, 1].into_iter().enumerate() {
        expected[i * 8..i * 8 + 4].copy_from_slice(&payload[row * 4..row * 4 + 4]);
    }
    let mut actual = [0xff; 32];
    stream.read(weights.view(weights.locate("padded", 16).unwrap()), &mut actual).unwrap();
    assert_eq!(actual, expected);
    let mut actual = [0xff; 16];
    stream.read(weights.view(weights.locate("gathered", 16).unwrap()), &mut actual).unwrap();
    assert_eq!(
        actual.as_slice(),
        [
            expected[0..4].to_vec(),
            expected[8..12].to_vec(),
            expected[16..20].to_vec(),
            expected[24..28].to_vec()
        ]
        .concat()
    );
}
