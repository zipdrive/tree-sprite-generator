from typing import Any, Self
import numpy as np

class Vector:
    HORIZONTAL_INDEX: int = 0
    VERTICAL_INDEX: int = 2
    DEPTH_INDEX: int = 1

    vector: np.typing.NDArray[np.floating[Any]]
    '''
    The underlying vector.
    '''

    def __init__(self, vec: np.typing.NDArray[np.floating[Any]]):
        self.vector = vec

    @classmethod 
    def construct(cls, horizontal: float = 0.0, vertical: float = 0.0, depth: float = 0.0):
        '''
        Constructs a vector from its components.

        Args:
            horizontal (float): The horizontal component of the vector.
            vertical (float): The vertical component of the vector.
            depth (float): The depth component of the vector.
        '''
        res = cls(np.array([0.0, 0.0, 0.0]))
        res.horizontal = horizontal 
        res.vertical = vertical 
        res.depth = depth 
        return res

    @property 
    def horizontal(self) -> float:
        '''
        Retrieves the horizontal component of the vector.
        '''
        return self.vector[self.HORIZONTAL_INDEX]
    
    @horizontal.setter 
    def horizontal(self, value: float):
        '''
        Assigns the horizontal component of the vector.
        '''
        self.vector[self.HORIZONTAL_INDEX] = value
    
    @property 
    def vertical(self) -> float:
        '''
        Retrieves the vertical component of the vector.
        '''
        return self.vector[self.VERTICAL_INDEX]
    
    @vertical.setter
    def vertical(self, value: float):
        '''
        Assigns the vertical component of the vector.
        '''
        self.vector[self.VERTICAL_INDEX] = value 

    @property
    def depth(self) -> float:
        '''
        Retrieves the depth component of the vector.
        '''
        return self.vector[self.DEPTH_INDEX]
    
    @depth.setter 
    def depth(self, value: float):
        '''
        Assigns the depth component of the vector.
        '''
        self.vector[self.DEPTH_INDEX] = value

    def __add__(self, other: Self) -> Self:
        return Vector(self.vector + other.vector)
    
    def __sub__(self, other: Self) -> Self:
        return Vector(self.vector - other.vector)
    
    def __mul__(self, scalar: float) -> Self:
        return Vector(self.vector * scalar)
    
    def __rmul__(self, scalar: float) -> Self:
        return Vector(self.vector * scalar)
    
    def __truediv__(self, scalar: float) -> Self:
        return Vector(self.vector / scalar)
    
    def __rtruediv__(self, scalar: float) -> Self:
        return Vector(self.vector / scalar)
    
    def __neg__(self) -> Self:
        return Vector(-self.vector)

    def dot(self, other: Self) -> float:
        '''
        Computes the dot product with another vector.
        '''
        return np.dot(self.vector, other.vector)
    
    @property
    def length(self):
        '''
        Calculates the length of the vector.
        '''
        return np.linalg.norm(self.vector)
    
    def normalize(self) -> Self:
        '''
        Normalizes the vector.
        '''
        return Vector(self.vector / self.length)
    
    def cross(self, other: Self) -> Self:
        '''
        Computes the cross product with another vector.

        Args:
            other (Self): Another vector.
        '''
        return Vector(np.cross(self.vector, other.vector))

    def rotate(self: Self, axis: Self, angle: float) -> Self:
        '''
        Rotates the vector around a particular axis.

        Args:
            input (Vector): The vector to be rotated.
            axis (Vector): The axis around which the vector is rotated.
            angle (float): The angle of rotation.
        '''
        axis = axis.normalize()
        parallel_factor: float = self.dot(axis)
        perpendicular_input: Vector = self - (parallel_factor * axis)
        perpendicular_input_magnitude: float = perpendicular_input.length
        rot_axis1: Vector = perpendicular_input / perpendicular_input_magnitude
        rot_axis2: Vector = axis.cross(rot_axis1)
        rotated_input: Vector = (np.cos(angle) * rot_axis1 + np.sin(angle) * rot_axis2) * perpendicular_input_magnitude + (parallel_factor * axis)
        return rotated_input
