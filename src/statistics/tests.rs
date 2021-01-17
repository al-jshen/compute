/// Calculates the t-test for the mean of one set of data. It tests for the null hypothesis that
/// the mean of a sample of independent observations `data` is equal to the population mean `mu`.
/// It returns the t statistic and the two-sided p-value.
pub fn ttest_1s(data: &[f64], mu: f64) -> (f64, f64) {
    unimplemented!();
}

/// Calculates the Student's t-test for two independent samples, assuming equal variance. This
/// is less reliable than Welch's t-test. See <https://en.wikipedia.org/wiki/Student%27s_t-test>.
pub fn ttest_2s_student(x: &[f64], y: &[f64]) -> (f64, f64) {
    unimplemented!();
}

/// Calculates Welch's t-test for two independent samples, without assuming equal variance. It tests
/// the hypothesis that the two populations have equal means. This is more reliable when the two
/// samples have unequal variances and/or unequal sample sizes. See
/// <https://en.wikipedia.org/wiki/Welch%27s_t-test>.
pub fn ttest_2s_welch(x: &[f64], y: &[f64]) -> (f64, f64) {
    unimplemented!();
}
