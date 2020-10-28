use compute::distributions::{Distribution, Normal};
use compute::functions::*;
use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_mean(c: &mut Criterion) {
    let v = Normal::new(0., 100.).sample_iter(1e6 as usize);
    c.bench_function("logit 1e6", |b| b.iter(|| v.iter().map(|x| logistic(*x))));
    c.bench_function("logit 1e6", |b| b.iter(|| v.iter().map(|x| logit(*x))));
    c.bench_function("gamma 1e6", |b| b.iter(|| v.iter().map(|x| gamma(*x))));
    c.bench_function("digamma 1e6", |b| b.iter(|| v.iter().map(|x| digamma(*x))));
}

criterion_group!(benches, criterion_mean);
criterion_main!(benches);
