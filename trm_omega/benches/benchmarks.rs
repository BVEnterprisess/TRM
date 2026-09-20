use criterion::{black_box, criterion_group, criterion_main, Criterion};
use candle_core::{Device, Tensor};
use trm_omega::deq::{solve_fixed_point, DeqConfig};
use trm_omega::quantize::{quantize_rowwise_ternary, unpack_ternary};

fn bench_deq_anderson(c: &mut Criterion) {
    let device = Device::Cpu;
    let dim = 128;
    let initial_z = Tensor::zeros((1, dim), candle_core::DType::F32, &device).unwrap();

    let mut group = c.benchmark_group("deq_solver");

    // Naive fixed point iteration
    group.bench_function("naive_iteration_15", |b| {
        b.iter(|| {
            let mut z = initial_z.clone();
            for _ in 0..15 {
                let fz = z.affine(0.85, 0.05).unwrap();
                z = fz;
            }
            black_box(z)
        });
    });

    // Anderson accelerated fixed point
    let config = DeqConfig {
        max_iter: 15,
        tolerance: 1e-5,
        anderson_history: 4,
        damping_base: 0.7,
        ..Default::default()
    };

    group.bench_function("anderson_acceleration_15", |b| {
        b.iter(|| {
            let res = solve_fixed_point(
                &initial_z,
                |cur| cur.affine(0.85, 0.05),
                &config,
            ).unwrap();
            black_box(res)
        });
    });

    group.finish();
}

fn bench_ternary_quantization(c: &mut Criterion) {
    let out_dim = 256;
    let in_dim = 256;
    let weights: Vec<f32> = (0..(out_dim * in_dim))
        .map(|i| ((i % 17) as f32 / 8.0) - 1.0)
        .collect();

    let mut group = c.benchmark_group("quantization");

    group.bench_function("quantize_rowwise_ternary_256x256", |b| {
        b.iter(|| {
            let packed = quantize_rowwise_ternary(black_box(&weights), out_dim, in_dim);
            black_box(packed)
        });
    });

    let packed = quantize_rowwise_ternary(&weights, out_dim, in_dim);
    group.bench_function("unpack_ternary_256x256", |b| {
        b.iter(|| {
            let unpacked = unpack_ternary(black_box(&packed));
            black_box(unpacked)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_deq_anderson, bench_ternary_quantization);
criterion_main!(benches);
