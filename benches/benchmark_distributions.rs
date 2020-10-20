use compute::distributions::*;
use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_generate(c: &mut Criterion) {
    c.bench_function("generate 1e6 standard normals", |b| {
        b.iter(|| Normal::new(0., 1.).sample_iter(1e6 as usize))
    });
    c.bench_function("generate 1e6 uniforms", |b| {
        b.iter(|| Uniform::new(-1., 1.).sample_iter(1e6 as usize))
    });
    c.bench_function("generate 1e6 gammas", |b| {
        b.iter(|| Gamma::new(2., 4.).sample_iter(1e6 as usize))
    });
    c.bench_function("generate 1e6 exponentials", |b| {
        b.iter(|| Exponential::new(2.).sample_iter(1e6 as usize))
    });
    c.bench_function("generate 1e6 discrete uniform values by rounding", |b| {
        b.iter(|| {
            Uniform::new(2.1, 200.1)
                .sample_iter(1e6 as usize)
                .iter()
                .map(|x| f64::floor(*x) as i64)
                .collect::<Vec<_>>()
        })
    });
    c.bench_function("generate 1e6 discrete uniform values directly", |b| {
        b.iter(|| DiscreteUniform::new(2, 200).sample_iter(1e6 as usize))
    });
}

criterion_group!(benches, criterion_generate);
criterion_main!(benches);
