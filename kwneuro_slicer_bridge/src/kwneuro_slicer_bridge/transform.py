"""InSceneTransformResource: wrap kwneuro.reg.TransformResource as Slicer scene transforms.

kwneuro's `TransformResource` is backed by on-disk ANTs output files:
zero or more affine `.mat` files and zero or more displacement-field
`.nii` files, applied right-to-left in ANTs order. The Slicer scene
representation for the same information is a **list of transform nodes**:

* `.mat` -> `vtkMRMLLinearTransformNode`
* `.nii(.gz)` displacement field -> `vtkMRMLGridTransformNode`

`slicer.util.loadTransform` dispatches on file extension and produces
the correct node type.

Phase 1 scope: one-way conversion from kwneuro to Slicer (`from_transform`
and `from_affine_matrix`). Going the other direction — building a new
`TransformResource` from Slicer scene nodes — needs saving nodes to disk
and is deferred. Unlike the other InScene* classes this is a standalone
wrapper, not a subclass of its kwneuro analogue, because kwneuro's
`TransformResource` models the transform as file paths rather than as a
live handle you can populate from the scene.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import slicer

from kwneuro.reg import TransformResource


@dataclass
class InSceneTransformResource:
    """A forward-transform representation as a list of Slicer transform nodes.

    The list is in ANTs forward-transform order: the last element is
    applied first to a point (consistent with ANTs' `transformlist`
    convention).
    """

    node_ids: list[str]

    def get_nodes(self) -> list[Any]:
        resolved = []
        for nid in self.node_ids:
            node = slicer.mrmlScene.GetNodeByID(nid)
            if node is None:
                msg = f"MRML node {nid!r} not found in the scene"
                raise ValueError(msg)
            resolved.append(node)
        return resolved

    def get_linear_matrices(self) -> list[NDArray[np.floating]]:
        """Return the 4x4 RAS matrix of each backing linear transform node.

        Skips non-linear nodes silently. Order matches `node_ids`.
        """
        import vtk

        matrices: list[NDArray[np.floating]] = []
        for node in self.get_nodes():
            if not node.IsA("vtkMRMLLinearTransformNode"):
                continue
            mat = vtk.vtkMatrix4x4()
            node.GetMatrixTransformToParent(mat)
            arr = np.eye(4)
            for i in range(4):
                for j in range(4):
                    arr[i, j] = mat.GetElement(i, j)
            matrices.append(arr)
        return matrices

    @staticmethod
    def from_affine_matrix(
        affine: NDArray[np.floating],
        name: str = "kwneuro_affine_transform",
    ) -> InSceneTransformResource:
        """Create a vtkMRMLLinearTransformNode from a 4x4 RAS matrix."""
        import vtk

        arr = np.asarray(affine, dtype=np.float64)
        if arr.shape != (4, 4):
            msg = f"affine must be 4x4, got {arr.shape}"
            raise ValueError(msg)

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", name)
        mat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                mat.SetElement(i, j, arr[i, j])
        node.SetMatrixTransformToParent(mat)
        return InSceneTransformResource(node_ids=[node.GetID()])

    @staticmethod
    def from_transform(
        transform: TransformResource,
        name_prefix: str = "kwneuro_transform",
    ) -> InSceneTransformResource:
        """Load each forward ANTs file into a Slicer transform node.

        Uses `slicer.util.loadTransform` to dispatch on file extension:
        `.mat` produces a linear transform; `.nii(.gz)` produces a grid
        transform.
        """
        node_ids = []
        for idx, path_str in enumerate(transform._ants_fwd_paths):
            node = slicer.util.loadTransform(str(path_str))
            if node is None:
                msg = f"slicer.util.loadTransform failed on {path_str!r}"
                raise RuntimeError(msg)
            path = Path(path_str)
            node.SetName(f"{name_prefix}_{idx}_{path.stem}")
            node_ids.append(node.GetID())
        return InSceneTransformResource(node_ids=node_ids)
