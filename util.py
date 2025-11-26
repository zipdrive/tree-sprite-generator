from typing import Any
import numpy as np

def normalized(vec: np.typing.NDArray[np.floating[Any]]) -> np.typing.NDArray[np.floating[Any]]:
    '''
    Normalizes a vector.
    '''
    return vec / np.linalg.norm(vec)

def rotated(input: np.typing.NDArray[np.floating[Any]], axis: np.typing.NDArray[np.floating[Any]], angle: float):
    '''
    Rotates a vector around a particular axis.

    Args:
        input (np.typing.NDArray[np.floating[Any]]): The vector to be rotated.
        axis (np.typing.NDArray[np.floating[Any]]): The axis around which the vector is rotated.
        angle (float): The angle of rotation.
    '''
    axis = normalized(axis)
    parallel_factor: float = np.dot(input, axis)
    perpendicular_input: np.typing.NDArray[np.floating[Any]] = input - (parallel_factor * axis)
    perpendicular_input_magnitude: np.floating[Any] = np.linalg.norm(perpendicular_input)
    rot_axis1: np.typing.NDArray[np.floating[Any]] = perpendicular_input / perpendicular_input_magnitude
    rot_axis2: np.typing.NDArray[np.floating[Any]] = np.cross(axis, rot_axis1)
    rotated_input: np.typing.NDArray[np.floating[Any]] = (np.cos(angle) * rot_axis1 + np.sin(angle) * rot_axis2) * perpendicular_input_magnitude + (parallel_factor * axis)
    return rotated_input