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
    let v5 = Uniform::new(2., 50.).sample_vec(5 * 5);
    c.bench_function("5x5 lu factorization", |b| b.iter(|| lu(&v5)));
    let v25 = Uniform::new(2., 50.).sample_vec(25 * 25);
    c.bench_function("25x25 lu factorization", |b| b.iter(|| lu(&v25)));
}

pub fn criterion_solve(c: &mut Criterion) {
    let a5 = Uniform::new(2., 50.).sample_vec(5 * 5);
    let b5 = Uniform::new(6., 30.).sample_vec(5);
    c.bench_function("5 variable linear solve", |b| b.iter(|| solve(&a5, &b5)));
    let a10 = Uniform::new(2., 50.).sample_vec(10 * 10);
    let b10 = Uniform::new(6., 30.).sample_vec(10);
    c.bench_function("10 variable linear solve", |b| b.iter(|| solve(&a10, &b10)));
    let a30 = Uniform::new(2., 50.).sample_vec(30 * 30);
    let b30 = Uniform::new(6., 30.).sample_vec(30);
    c.bench_function("30 variable linear solve", |b| b.iter(|| solve(&a30, &b30)));
}

pub fn criterion_invert(c: &mut Criterion) {
    let a5 = Uniform::new(2., 50.).sample_vec(5 * 5);
    let a10 = Uniform::new(2., 50.).sample_vec(10 * 10);
    let a30 = Uniform::new(2., 50.).sample_vec(30 * 30);
    c.bench_function("5x5 inversion", |b| b.iter(|| invert_matrix(&a5)));
    c.bench_function("10x10 inversion", |b| b.iter(|| invert_matrix(&a10)));
    c.bench_function("30x30 inversion", |b| b.iter(|| invert_matrix(&a30)));
}

pub fn criterion_matmul(c: &mut Criterion) {
    let a1 = Normal::new(2., 50.).sample_vec(5 * 15);
    let a2 = Normal::new(2., 50.).sample_vec(15 * 10);
    c.bench_function("5x15x10 matmul", |b| {
        b.iter(|| matmul(&a1, &a2, 5, 15, false, false))
    });
}

pub fn criterion_xtx(c: &mut Criterion) {
    let a_20_6 = Uniform::new(2., 50.).sample_vec(20 * 6);
    c.bench_function("20x6 xtx", |b| b.iter(|| xtx(&a_20_6, 20)));
}

criterion_group!(benches, criterion_xtx);
criterion_main!(benches);
