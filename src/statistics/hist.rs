use crate::linalg::Vector;

// Given the edges of some intervals, return the centers of the intervals. If `edges` is length n,
// then the resulting vector will be length n - 1.
pub fn hist_bin_centers(edges: &[f64]) -> Vector {
    let diff = Vector::from(edges).diff();

    diff.into_iter()
        .scan((edges[0] + edges[1]) / 2., |acc, x| {
            let temp = *acc;
            *acc += x;
            Some(temp)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::*;
    use crate::linalg::*;

    #[test]
    fn test_hist_bin_centers() {
        for _ in 0..10 {
            let lower = DiscreteUniform::new(0, 50).sample();
            let upper = DiscreteUniform::new(50, 100).sample();
            let n = DiscreteUniform::new(5, 500).sample() as usize;
            let arr = linspace(lower, upper, n);
            let answer = (Vector::from(&arr[1..]) + Vector::from(&arr[..arr.len() - 1])) / 2.;
            assert!(Vector::from(hist_bin_centers(&arr)).close_to(&answer, 1e-6));
        }
    }
}
