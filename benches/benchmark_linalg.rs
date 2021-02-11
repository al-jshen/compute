use compute::distributions::{Distribution, Normal, Uniform};
use compute::linalg::*;
use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_dot(c: &mut Criterion) {
    let v1 = Normal::new(0., 100.).sample_vec(1e5 as usize);
    let v2 = Normal::new(0., 100.).sample_vec(1e5 as usize);
    c.bench_function("dot product 1e5", |b| b.iter(|| dot(&v1, &v2)));
}

pub fn criterion_norm(c: &mut Criterion) {
    let v1 = Normal::new(0., 100.).sample_vec(1e3 as usize);
    c.bench_function("norm 1e3", |b| b.iter(|| norm(&v1)));
}

pub fn criterion_ludecomp(c: &mut Criterion) {
    let v = Uniform::new(2., 50.).sample_vec(100 as usize);
    c.bench_function("10x10 lu factorization", |b| b.iter(|| lu(&v)));
}

pub fn criterion_solve(c: &mut Criterion) {
    let a = Uniform::new(2., 50.).sample_vec(20 * 20);
    let bvec = Uniform::new(6., 30.).sample_vec(20);
    c.bench_function("20 variable linear solve", |b| b.iter(|| solve(&a, &bvec)));
}

criterion_group!(benches, criterion_solve);
criterion_main!(benches);
