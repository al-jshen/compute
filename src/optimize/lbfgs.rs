#[derive(Debug, Clone, Copy)]
pub struct LBFGS {}

impl LBFGS {
    fn linesearch() {}

    fn twoloop() {}

    fn optimize() {}
}

#[cfg(test)]
mod tests {
    use crate::prelude::{Float, F1};
    use std::f64::consts::PI;

    #[test]
    fn test_lbfgs() {
        fn loglikelihood(params: &[F1], data: &[&[f64]]) -> F1 {
            assert_eq!(params.len(), 3);
            assert_eq!(data.len(), 2);
            assert_eq!(data[0].len(), data[1].len());
            let (b, m, sigma) = (params[0], params[1], params[2]);
            let n = data[0].len() as f64;
            let mu: Vec<F1> = data[0].iter().map(|&v| m * v + b).collect();
            let ymusqsig: F1 = data[1]
                .iter()
                .zip(mu)
                .map(|(yv, muv)| (*yv - muv).powi(2) / (2. * sigma.powi(2)))
                .sum();
            return -n / 2. * (2. * PI * sigma.powi(2)) - ymusqsig;
        }

        dbg!(loglikelihood(
            &[2., 3., 1.].iter().map(|x| F1::var(*x)).collect::<Vec<_>>(),
            &[&[1., 2.], &[2., 4.]]
        ));
    }
}
