use compute::functions::*;
use compute::{
    distributions::{Distribution, Normal},
    prelude::DiscreteUniform,
};
use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_mean(c: &mut Criterion) {
    let v = Normal::new(0., 100.).sample_n(1e6 as usize);
    c.bench_function("logit 1e6", |b| b.iter(|| v.iter().map(|x| logistic(*x))));
    c.bench_function("logit 1e6", |b| b.iter(|| v.iter().map(|x| logit(*x))));
    c.bench_function("gamma 1e6", |b| b.iter(|| v.iter().map(|x| gamma(*x))));
    c.bench_function("digamma 1e6", |b| b.iter(|| v.iter().map(|x| digamma(*x))));
}

pub fn criterion_binomial(c: &mut Criterion) {
    let n: Vec<u64> = DiscreteUniform::new(5, 100)
        .sample_n(1000)
        .iter()
        .map(|x| *x as u64)
        .collect();
    let k: Vec<u64> = n.iter().map(|x| (x / 2)).collect();
    c.bench_function("binomial coeffs 1e3", |b| {
        b.iter(|| (0..1000).into_iter().map(|i| binom_coeff(n[i], k[i])))
    });
}

criterion_group!(benches, criterion_binomial);
criterion_main!(benches);
