use std::f64::consts::PI;

const G: f64 = 4.7421875 + 1.;

/// Coefficients from [here](https://my.fit.edu/~gabdo/gamma.txt).
const GAMMA_COEFFS: [f64; 14] = [
    57.156235665862923517,
    -59.597960355475491248,
    14.136097974741747174,
    -0.49191381609762019978,
    0.33994649984811888699e-4,
    0.46523628927048575665e-4,
    -0.98374475304879564677e-4,
    0.15808870322491248884e-3,
    -0.21026444172410488319e-3,
    0.21743961811521264320e-3,
    -0.16431810653676389022e-3,
    0.84418223983852743293e-4,
    -0.26190838401581408670e-4,
    0.36899182659531622704e-5,
];

/// Calculates the Gamma function using the [Lanczos
/// approximation](https://en.wikipedia.org/wiki/Lanczos_approximation). Has a typical precision of
/// 15 decimal places. Uses the reflection formula to extend the calculation to the entire complex
/// plane.
pub fn gamma(mut z: f64) -> f64 {
    if z < 0.5 {
        PI / ((PI * z).sin() * gamma(1. - z))
    } else {
        z -= 1.;
        let mut x = 0.99999999999999709182;
        for (idx, val) in GAMMA_COEFFS.iter().enumerate() {
            x += val / (z + (idx as f64) + 1.);
        }
        let t = z + G - 0.5;
        ((2. * PI) as f64).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
    }
}

#[test]
fn test_gamma() {
    assert!((gamma(0.1) - 9.513507698668731836292487).abs() / 9.513507698668731836292487 < 1e-10);
    assert!((gamma(0.5) - 1.7724538509551602798167).abs() / 1.7724538509551602798167 < 1e-10);
    assert!((gamma(6.) - 120.).abs() / 120. < 1e-10);
    assert!((gamma(20.) - 121645100408832000.).abs() / 121645100408832000. < 1e-10);
    assert!((gamma(-0.5) - -3.54490770181103205459).abs() / -3.54490770181103205459 < 1e-10);
}
