/// Utilities for 3D rotations.

pub enum Axis {
    X,
    Y,
    Z,
}
///
/// Returns a 3D clockwise rotation matrix along the `axis` axis (either X, Y, or Z) with a given angle in
/// radians.
///
/// # Remarks
/// These are clockwise rotation matrices. Use `rotation_matrix_ccw` to get counter-clockwise rotation
/// matrices.
pub fn rotation_matrix_cw(angle: f64, axis: Axis) -> Vec<f64> {
    match axis {
        Axis::X => vec![
            1.,
            0.,
            0.,
            0.,
            angle.cos(),
            angle.sin(),
            0.,
            -angle.sin(),
            angle.cos(),
        ],
        Axis::Y => vec![
            angle.cos(),
            0.,
            -angle.sin(),
            0.,
            1.,
            0.,
            angle.sin(),
            0.,
            angle.cos(),
        ],
        Axis::Z => vec![
            angle.cos(),
            angle.sin(),
            0.,
            -angle.sin(),
            angle.cos(),
            0.,
            0.,
            0.,
            1.,
        ],
    }
}

/// Returns a 3D counter-clockwise rotation matrix along the `axis` axis (either X, Y, or Z) with a given angle in
/// radians.
///
/// # Remarks
/// These are counter-clockwise rotation matrices. Use `rotation_matrix_cw` to get clockwise rotation
/// matrices.
pub fn rotation_matrix_ccw(angle: f64, axis: Axis) -> Vec<f64> {
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
            angle.cos(),
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
