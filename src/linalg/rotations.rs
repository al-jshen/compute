/// Utilities for 3D rotations.

pub enum Axis {
    X,
    Y,
    Z,
}

/// Returns a 3D rotation matrix along the `axis` axis (either X, Y, or Z) with a given angle in
/// radians.
pub fn rotation_matrix(angle: f64, axis: Axis) -> Vec<f64> {
    match axis {
        Axis::X => vec![
            1.,
            0.,
            0.,
            0.,
            angle.cos(),
            -angle.sin(),
            0.,
            angle.sin(),
            -angle.cos(),
        ],
        Axis::Y => vec![
            angle.cos(),
            0.,
            angle.sin(),
            0.,
            1.,
            0.,
            -angle.sin(),
            0.,
            angle.cos(),
        ],
        Axis::Z => vec![
            angle.cos(),
            -angle.sin(),
            0.,
            angle.sin(),
            angle.cos(),
            0.,
            0.,
            0.,
            1.,
        ],
    }
}
