use compute::distributions::{Distribution1D, Normal};
use compute::linalg::*;
use compute::statistics::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_mean(c: &mut Criterion) {
    let v = Normal::new(0., 100.).sample_n(1e6 as usize);
    c.bench_function("mean 1e6", |b| b.iter(|| mean(black_box(&v))));
    c.bench_function("welford mean 1e6", |b| {
        b.iter(|| welford_mean(black_box(&v)))
    });
}

pub fn criterion_hist_bins(c: &mut Criterion) {
    let bin10 = linspace(0., 10., 10);
    let bin100 = linspace(0., 10., 100);
    c.bench_function("hist bin edges 10", |b| b.iter(|| hist_bin_centers(&bin10)));
    c.bench_function("hist bin edges 100", |b| {
        b.iter(|| hist_bin_centers(&bin100))
    });
}

criterion_group!(benches, criterion_hist_bins);
criterion_main!(benches);
