/// Utilities for 3D rotations.

pub enum Axis {
    X,
    Y,
    Z,
}

/// Returns a 3D rotation matrix along the `axis` axis (either X, Y, or Z) with a given angle in
/// degrees.
pub fn rotation_matrix(angle: f64, axis: Axis) -> Vec<f64> {
    let theta = angle.to_radians();
    match axis {
        Axis::X => vec![
            1.,
            0.,
            0.,
            0.,
            theta.cos(),
            -theta.sin(),
            0.,
            theta.sin(),
            -theta.cos(),
        ],
        Axis::Y => vec![
            theta.cos(),
            0.,
            theta.sin(),
            0.,
            1.,
            0.,
            -theta.sin(),
            0.,
            theta.cos(),
        ],
        Axis::Z => vec![
            theta.cos(),
            -theta.sin(),
            0.,
            theta.sin(),
            theta.cos(),
            0.,
            0.,
            0.,
            1.,
        ],
    }
}
