use compute::distributions::*;
use criterion::{criterion_group, criterion_main, Criterion};

pub fn generate(c: &mut Criterion) {
    c.bench_function("generate 1e6 standard normals", |b| {
        b.iter(|| Normal::new(0., 1.).sample_n(1e6 as usize))
    });
    c.bench_function("generate 1e6 gammas", |b| {
        b.iter(|| Gamma::new(2., 4.).sample_n(1e6 as usize))
    });
    c.bench_function("generate 1e6 exponentials", |b| {
        b.iter(|| Exponential::new(2.).sample_n(1e6 as usize))
    });
}

fn discrete_uniform(c: &mut Criterion) {
    c.bench_function("generate 1e6 discrete uniform values by rounding", |b| {
        b.iter(|| {
            Uniform::new(2.1, 200.1)
                .sample_n(1e6 as usize)
                .iter()
                .map(|x| f64::floor(*x) as i64)
                .collect::<Vec<_>>()
        })
    });
    c.bench_function("generate 1e6 discrete uniform values directly", |b| {
        b.iter(|| DiscreteUniform::new(2, 200).sample_n(1e6 as usize))
    });
}

fn poisson(c: &mut Criterion) {
    c.bench_function("generate 1e6 poissons with multiplication method", |b| {
        b.iter(|| Poisson::new(8.).sample_n(1e6 as usize))
    });
    c.bench_function("generate 1e6 poissons with PTRS algorithm", |b| {
        b.iter(|| Poisson::new(18.).sample_n(1e6 as usize))
    });
}

fn binomial(c: &mut Criterion) {
    c.bench_function("generate 1e3 poissons with inversion method", |b| {
        b.iter(|| Binomial::new(15, 0.4).sample_n(1e3 as usize))
    });
    c.bench_function("generate 1e3 poissons with BTPE algorithm", |b| {
        b.iter(|| Binomial::new(70, 0.7).sample_n(1e3 as usize))
    });
}

fn t(c: &mut Criterion) {
    c.bench_function("generate 1e6 t distributed variates", |b| {
        b.iter(|| T::new(2.).sample_n(1e6 as usize))
    });
}

criterion_group!(benches, t);
criterion_main!(benches);
