#![allow(dead_code)]

use super::convolve;

const SG35: &[f64] = &[-3., 12., 17.];
const SG37: &[f64] = &[-2., 3., 6., 7.];
const SG39: &[f64] = &[-21., 14., 39., 54., 59.];
const SG311: &[f64] = &[-36., 9., 44., 69., 84., 89.];
const SG313: &[f64] = &[-11., 0., 9., 16., 21., 24., 25.];
const SG315: &[f64] = &[-78., -13., 42., 87., 122., 147., 162., 167.];
const SG317: &[f64] = &[-21., -6., 7., 18., 27., 34., 39., 42., 43.];
const SG319: &[f64] = &[-136., -51., 24., 89., 144., 189., 224., 249., 264., 269.];
const SG321: &[f64] = &[
    -171., -76., 9., 84., 149., 204., 249., 284., 309., 324., 329.,
];
const SG323: &[f64] = &[-42., -21., 2., 15., 30., 43., 54., 63., 70., 75., 78., 89.];
const SG325: &[f64] = &[
    -253., -138., -33., 62., 147., 222., 287., 322., 387., 422., 447., 462., 467.,
];

const SG35S: u64 = 35;
const SG37S: u64 = 21;
const SG39S: u64 = 231;
const SG311S: u64 = 429;
const SG313S: u64 = 143;
const SG315S: u64 = 1105;
const SG317S: u64 = 323;
const SG319S: u64 = 2261;
const SG321S: u64 = 3059;
const SG323S: u64 = 8059;
const SG325S: u64 = 5175;

const SG57: &[f64] = &[5., -30., 75., 131.];
const SG59: &[f64] = &[15., -55., 30., 135., 179.];
const SG511: &[f64] = &[18., -45., -10., 60., 120., 143.];
const SG513: &[f64] = &[110., -198., -160., 110., 390., 600., 677.];
const SG515: &[f64] = &[2145., -2860., -2937., -165., 3755., 7500., 10125., 11063.];
const SG517: &[f64] = &[195., -195., -260., -117., 135., 415., 660., 825., 883.];
const SG519: &[f64] = &[
    340., -255., -420., -290., 18., 405., 790., 1110., 1320., 1393.,
];
const SG521: &[f64] = &[
    11628., -6460., -13005., -11220., -3940., 6378., 17655., 28190., 36660., 42120., 4403.,
];
const SG523: &[f64] = &[
    285., -114., -285., -285., -165., 30., 261., 495., 705., 870., 975., 1011.,
];
const SG525: &[f64] = &[
    1265., -345., -1122., -1255., -915., -255., 590., 1503., 2385., 3155., 3750., -4125., 4253.,
];

const SG57S: u64 = 231;
const SG59S: u64 = 429;
const SG511S: u64 = 429;
const SG513S: u64 = 2431;
const SG515S: u64 = 46189;
const SG517S: u64 = 4199;
const SG519S: u64 = 7429;
const SG521S: u64 = 260015;
const SG523S: u64 = 6555;
const SG525S: u64 = 30015;

/// Smooths a signal by applying a
/// [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter).
/// The `points` parameter specifies the number of points to use in the convolution,
/// and should be an odd number. The `order` parameter specifies the order of
/// polynomial to use in the smoothing.
///
/// # Remarks
/// This function currently uses Savitzky-Golay coefficients from the original paper, which means
/// that the number of points only goes up to 25, and the order only goes up to 5. In the future a
/// function to calculate the coefficients for an arbitrary number of points and order will be implemented.
pub fn savitzky_golay(signal: &[f64], points: usize, order: usize) -> Vec<f64> {
    let (coeffs, sum) = match order {
        2 | 3 => match points {
            5 => (SG35, SG35S),
            7 => (SG37, SG37S),
            9 => (SG39, SG39S),
            11 => (SG311, SG311S),
            13 => (SG313, SG313S),
            15 => (SG315, SG315S),
            17 => (SG317, SG317S),
            19 => (SG319, SG319S),
            21 => (SG321, SG321S),
            23 => (SG323, SG323S),
            _ => (SG325, SG325S),
        },
        _ => match points {
            7 => (SG57, SG57S),
            9 => (SG59, SG59S),
            11 => (SG511, SG511S),
            13 => (SG513, SG513S),
            15 => (SG515, SG515S),
            17 => (SG517, SG517S),
            19 => (SG519, SG519S),
            21 => (SG521, SG521S),
            23 => (SG523, SG523S),
            _ => (SG525, SG525S),
        },
    };
    let mut weights_backhalf = coeffs.to_owned();
    weights_backhalf.reverse();
    let mut weights = coeffs.to_vec();
    weights.extend_from_slice(&weights_backhalf[1..]);

    convolve(signal, &weights, 1. / sum as f64)
        .drain((points / 2)..)
        .collect::<Vec<_>>()
        .drain(..signal.len())
        .collect::<Vec<_>>()
}
