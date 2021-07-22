use compute::distributions::{Distribution, Normal, Uniform};
use compute::linalg::lu::*;
use compute::linalg::*;
use compute::prelude::Distribution1D;
use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_dot(c: &mut Criterion) {
    let v1 = Normal::new(0., 100.).sample_n(1e6 as usize);
    let v2 = Normal::new(0., 100.).sample_n(1e6 as usize);
    let v3 = Normal::new(0., 100.).sample_n(1e4 as usize);
    let v4 = Normal::new(0., 100.).sample_n(1e4 as usize);
    c.bench_function("dot product 1e6", |b| b.iter(|| dot(&v1, &v2)));
    c.bench_function("dot product 1e4", |b| b.iter(|| dot(&v3, &v4)));
}

pub fn criterion_norm(c: &mut Criterion) {
    let v1 = Normal::new(0., 100.).sample_n(1e3 as usize);
    c.bench_function("norm 1e3", |b| b.iter(|| norm(&v1)));
}

pub fn criterion_logsumexp(c: &mut Criterion) {
    let v1 = Normal::new(0., 1.).sample_n(1e3 as usize);
    c.bench_function("logsumexp 1e3", |b| b.iter(|| logsumexp(&v1)));
}

pub fn criterion_vops_assign(c: &mut Criterion) {
    let v1 = Normal::default().sample_n(1000);
    let v2 = Normal::default().sample_n(1000);
    let s = Normal::default().sample();

    c.bench_function("vector add with assignment", |b| {
        b.iter(|| {
            let v = &v1 + &v2;
        })
    });
    c.bench_function("vector clone and mutating add", |b| {
        b.iter(|| {
            let mut v = v1.clone();
            v += &v2;
        })
    });

    c.bench_function("vector subtract with assignment", |b| {
        b.iter(|| {
            let v = &v1 - &v2;
        })
    });
    c.bench_function("vector clone and mutating subtract", |b| {
        b.iter(|| {
            let mut v = v1.clone();
            v -= &v2;
        })
    });

    c.bench_function("vector multiply with assignment", |b| {
        b.iter(|| {
            let v = &v1 * &v2;
        })
    });

    c.bench_function("vector clone and mutating multiply", |b| {
        b.iter(|| {
            let mut v = v1.clone();
            v *= &v2;
        })
    });

    c.bench_function("vector divide with assignment", |b| {
        b.iter(|| {
            let v = &v1 / &v2;
        })
    });
    c.bench_function("vector clone and mutating divide", |b| {
        b.iter(|| {
            let mut v = v1.clone();
            v /= &v2;
        })
    });
}

pub fn criterion_vops(c: &mut Criterion) {
    let mut v1 = Normal::default().sample_n(1000);
    let mut v2 = Normal::default().sample_n(1000);
    let s = Normal::default().sample();
    // c.bench_function("vector add", |b| b.iter(|| &v1 + &v2));
    // c.bench_function("normal add", |b| {
    //     b.iter(|| v1.iter().zip(&v2).map(|(i, j)| i + j).collect::<Vector>())
    // });

    c.bench_function("vector multiply", |b| b.iter(|| &v1 * &v2));
    c.bench_function("normal multiply", |b| {
        b.iter(|| v1.iter().zip(&v2).map(|(i, j)| i * j).collect::<Vector>())
    });

    c.bench_function("vector divide", |b| b.iter(|| &v1 / &v2));
    c.bench_function("normal divide", |b| {
        b.iter(|| v1.iter().zip(&v2).map(|(i, j)| i * j).collect::<Vector>())
    });

    // c.bench_function("vector ln", |b| b.iter(|| v1.ln()));
    // c.bench_function("normal ln", |b| {
    //     b.iter(|| v1.iter().map(|i| i.ln()).collect::<Vector>())
    // });

    // c.bench_function("vector sqrt", |b| b.iter(|| v1.sqrt()));
    // c.bench_function("normal sqrt", |b| {
    //     b.iter(|| v1.iter().map(|i| i.sqrt()).collect::<Vector>())
    // });

    // c.bench_function("vector exp", |b| b.iter(|| v1.exp()));
    // c.bench_function("normal exp", |b| {
    //     b.iter(|| v1.iter().map(|i| i.exp()).collect::<Vector>())
    // });

    c.bench_function("vector square", |b| b.iter(|| v1.powi(2)));
    c.bench_function("normal square", |b| {
        b.iter(|| Vector::from(v1.iter().map(|i| i.powi(2)).collect::<Vec<f64>>()))
    });

    c.bench_function("vector cube", |b| b.iter(|| v1.powi(3)));
    c.bench_function("normal cube", |b| {
        b.iter(|| v1.iter().map(|i| i.powi(3)).collect::<Vector>())
    });

    c.bench_function("vector float power", |b| b.iter(|| v1.powf(1.25)));
    c.bench_function("normal float power", |b| {
        b.iter(|| v1.iter().map(|i| i.powf(1.25)).collect::<Vector>())
    });

    // c.bench_function("unrolled vector-scalar addition", |b| b.iter(|| &v1 + s));
    // c.bench_function("normal vector-scalar addition", |b| {
    //     b.iter(|| v1.iter().map(|i| i + s).collect::<Vector>())
    // });

    // c.bench_function("unrolled vector-scalar division", |b| b.iter(|| &v1 / s));
    // c.bench_function("normal vector-scalar division", |b| {
    //     b.iter(|| v1.iter().map(|i| i / s).collect::<Vector>())
    // });

    // c.bench_function("unrolled scalar-vector subtraction", |b| b.iter(|| s - &v1));
    // c.bench_function("normal scalar-vector subtraction", |b| {
    //     b.iter(|| v1.iter().map(|i| s - i).collect::<Vector>())
    // });
}

pub fn criterion_matrix_sum(c: &mut Criterion) {
    let v = Normal::default().sample_matrix(100, 100);
    c.bench_function("sum rows 100x100", |b| b.iter(|| v.sum_rows()));
    c.bench_function("sum cols 100x100", |b| b.iter(|| v.sum_cols()));
}

pub fn criterion_ludecomp(c: &mut Criterion) {
    let v5 = Uniform::new(2., 50.).sample_n(5 * 5);
    c.bench_function("5x5 lu factorization", |b| b.iter(|| lu(&v5)));
    let v25 = Uniform::new(2., 50.).sample_n(25 * 25);
    c.bench_function("25x25 lu factorization", |b| b.iter(|| lu(&v25)));
}

pub fn criterion_solve(c: &mut Criterion) {
    let a5 = Uniform::new(2., 50.).sample_n(5 * 5);
    let b5 = Uniform::new(6., 30.).sample_n(5);
    c.bench_function("5 variable linear solve", |b| b.iter(|| solve(&a5, &b5)));
    let a10 = Uniform::new(2., 50.).sample_n(10 * 10);
    let b10 = Uniform::new(6., 30.).sample_n(10);
    c.bench_function("10 variable linear solve", |b| b.iter(|| solve(&a10, &b10)));
    let a30 = Uniform::new(2., 50.).sample_n(30 * 30);
    let b30 = Uniform::new(6., 30.).sample_n(30);
    c.bench_function("30 variable linear solve", |b| b.iter(|| solve(&a30, &b30)));
}

pub fn criterion_invert(c: &mut Criterion) {
    let a5 = Uniform::new(2., 50.).sample_n(5 * 5);
    let a10 = Uniform::new(2., 50.).sample_n(10 * 10);
    let a30 = Uniform::new(2., 50.).sample_n(30 * 30);
    c.bench_function("5x5 inversion", |b| b.iter(|| invert_matrix(&a5)));
    c.bench_function("10x10 inversion", |b| b.iter(|| invert_matrix(&a10)));
    c.bench_function("30x30 inversion", |b| b.iter(|| invert_matrix(&a30)));
}

pub fn criterion_matmul(c: &mut Criterion) {
    let normgen = Normal::new(2., 50.);
    let a1 = normgen.sample_n(5 * 15);
    let a2 = normgen.sample_n(15 * 10);
    let a3 = normgen.sample_n(512 * 512);

    c.bench_function("5x15x10 matmul", |b| {
        b.iter(|| matmul(&a1, &a2, 5, 15, false, false))
    });
    c.bench_function("5x15x10 matmul with transpose", |b| {
        b.iter(|| matmul(&a2, &a1, 15, 5, true, true))
    });
    c.bench_function("512x512 matmul", |b| {
        b.iter(|| matmul(&a3, &a3, 512, 512, false, false))
    });
    c.bench_function("512x512 matmul with transpose", |b| {
        b.iter(|| matmul(&a3, &a3, 512, 512, true, true))
    });
}

pub fn criterion_matmul_blocked(c: &mut Criterion) {
    let normgen = Normal::new(2., 50.);
    let a3 = normgen.sample_n(512 * 512);

    c.bench_function("512x512 blocked matmul, blocksize 25", |b| {
        b.iter(|| matmul_blocked(&a3, &a3, 512, 512, false, false, 25))
    });
    c.bench_function("512x512 blocked matmul, blocksize 100", |b| {
        b.iter(|| matmul_blocked(&a3, &a3, 512, 512, false, false, 100))
    });
    c.bench_function("512x512 blocked matmul, blocksize 250", |b| {
        b.iter(|| matmul_blocked(&a3, &a3, 512, 512, false, false, 250))
    });
}

pub fn criterion_xtx(c: &mut Criterion) {
    let a_20_6 = Uniform::new(2., 50.).sample_n(20 * 6);
    c.bench_function("20x6 xtx", |b| b.iter(|| xtx(&a_20_6, 20)));
}

criterion_group!(benches, criterion_vops);
criterion_main!(benches);
