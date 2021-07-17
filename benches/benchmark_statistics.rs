use compute::distributions::{Distribution, Normal};
use compute::statistics::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_mean(c: &mut Criterion) {
    let v = Normal::new(0., 100.).sample_n(1e6 as usize);
    c.bench_function("mean 1e6", |b| b.iter(|| mean(black_box(&v))));
    c.bench_function("welford mean 1e6", |b| {
        b.iter(|| welford_mean(black_box(&v)))
    });
}

criterion_group!(benches, criterion_mean);
criterion_main!(benches);
