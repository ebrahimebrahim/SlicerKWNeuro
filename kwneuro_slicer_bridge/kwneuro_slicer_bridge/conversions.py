"""Conversion helpers between numpy arrays + affines and VTK image data + matrices.

These helpers encode the representational contract between kwneuro (nibabel-style
IJK-ordered arrays with 4x4 RAS+ affines) and 3D Slicer (vtkImageData with an
IJKToRAS 4x4 matrix stored on the MRML volume node).

The helpers are intentionally pure: no module-level `slicer`/`vtk` imports, so
they can be exercised at unit-test collection time without a Slicer runtime.
Each function imports `vtk` lazily inside its body.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def numpy_to_vtk_image(array: NDArray[np.number]) -> Any:
    """Convert a 3D or 4D numpy array to a vtkImageData with identity transform.

    Slicer convention: the vtkImageData carries only voxel values; spacing,
    origin, and direction live on the volume node's IJKToRAS matrix. We
    therefore set identity spacing/origin on the image data itself.

    VTK's multi-component scalar layout interleaves components per voxel:
    `buffer[c + C * pointId]` where `pointId = i + nx * (j + ny * k)`. A
    4D numpy input `(nx, ny, nz, C)` is therefore transposed to
    `(C, nx, ny, nz)` before F-order flattening so the resulting flat
    buffer has component as the fastest-varying axis.
    """
    import vtk
    from vtk.util import numpy_support

    if array.ndim not in (3, 4):
        raise ValueError(f"Array must be 3D or 4D, got {array.ndim}D")

    nx, ny, nz = array.shape[:3]

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(nx, ny, nz)
    image_data.SetSpacing(1.0, 1.0, 1.0)
    image_data.SetOrigin(0.0, 0.0, 0.0)

    if array.ndim == 4:
        n_components = array.shape[3]
        transposed = np.asfortranarray(array.transpose(3, 0, 1, 2))
        flat = transposed.flatten(order="F")
    else:
        n_components = 1
        flat = np.asfortranarray(array).flatten(order="F")

    vtk_array = numpy_support.numpy_to_vtk(
        flat, deep=True, array_type=vtk.VTK_FLOAT,
    )
    vtk_array.SetNumberOfComponents(n_components)
    image_data.GetPointData().SetScalars(vtk_array)
    return image_data


def vtk_image_to_numpy(image_data: Any) -> NDArray[np.number]:
    """Convert a vtkImageData to a numpy array in IJK order.

    Single-component images return as a 3D `(nx, ny, nz)` array.
    Multi-component images return as `(nx, ny, nz, C)`.

    Note: `numpy_support.vtk_to_numpy` on a multi-component vtkDataArray
    returns an already-decoded 2D array of shape `(npoints, ncomponents)`.
    We reshape that to `(nx, ny, nz, C)` using F-order so that axis 0 is
    IJK's `i` (fastest-varying).
    """
    from vtk.util import numpy_support

    nx, ny, nz = image_data.GetDimensions()
    scalars = image_data.GetPointData().GetScalars()
    n_components = scalars.GetNumberOfComponents()
    raw = numpy_support.vtk_to_numpy(scalars)

    if n_components > 1:
        array = raw.reshape(nx, ny, nz, n_components, order="F")
    else:
        array = raw.reshape(nx, ny, nz, order="F")
    return array


def affine_to_ijk_to_ras_matrix(affine: NDArray[np.floating]) -> Any:
    """Copy a 4x4 numpy affine into a freshly-constructed vtkMatrix4x4."""
    import vtk

    matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i, j, affine[i, j])
    return matrix


def ijk_to_ras_matrix_to_affine(matrix: Any) -> NDArray[np.floating]:
    """Copy a vtkMatrix4x4 into a 4x4 numpy affine."""
    affine = np.eye(4)
    for i in range(4):
        for j in range(4):
            affine[i, j] = matrix.GetElement(i, j)
    return affine
