use compute::distributions::{Distribution, Normal};
use compute::utils::*;
use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_dot(c: &mut Criterion) {
    let v1 = Normal::new(0., 100.).sample_vec(1e5 as usize);
    let v2 = Normal::new(0., 100.).sample_vec(1e5 as usize);
    c.bench_function("dot product 1e5", |b| b.iter(|| dot(&v1, &v2)));
}

criterion_group!(benches, criterion_dot);
criterion_main!(benches);
