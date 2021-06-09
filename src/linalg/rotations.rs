/// Utilities for 3D rotations.
use super::Matrix;

pub enum Axis {
    X,
    Y,
    Z,
}

/// Returns a 3D clockwise rotation matrix along the `axis` axis (either X, Y, or Z) with a given angle in
/// radians.
///
/// # Remarks
/// These are clockwise rotation matrices. Use `rotation_matrix_ccw` to get counter-clockwise rotation
/// matrices.
pub fn rotation_matrix_cw(angle: f64, axis: Axis) -> Matrix {
    let data = match axis {
        Axis::X => [
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
        Axis::Y => [
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
        Axis::Z => [
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
    };
    Matrix::new(data, 3, 3)
}

/// Returns a 3D counter-clockwise rotation matrix along the `axis` axis (either X, Y, or Z) with a given angle in
/// radians.
///
/// # Remarks
/// These are counter-clockwise rotation matrices. Use `rotation_matrix_cw` to get clockwise rotation
/// matrices.
pub fn rotation_matrix_ccw(angle: f64, axis: Axis) -> Matrix {
    let data = match axis {
        Axis::X => [
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
        Axis::Y => [
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
        Axis::Z => [
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
    };
    Matrix::new(data, 3, 3)
}
