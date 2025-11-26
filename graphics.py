from __future__ import annotations
import math
import numpy as np
import png
import moderngl
from typing import Any, Generator
from util import Vector
from structure import TreeBranchSegment, TreeStructure

class Image:
    width: int 
    '''
    The width of the image.
    '''

    height: int
    '''
    The height of the image.
    '''

    pixels: list[int]
    '''
    The pixels of the image.
    '''

    renderers: list[TreeRenderer]
    '''
    The renderers for the image.
    '''

    def __init__(self, renderers: list[TreeRenderer], width: int, height: int):
        '''
        Constructs an empty image.
        '''
        self.renderers = renderers
        self.width = width
        self.height = height 
        self.pixels = [0 for _ in range(4 * width * height)]

    def __getitem__(self, key: tuple[int, int]) -> tuple[int, int, int, int]:
        x, y = key
        return tuple(self.pixels[(4 * ((self.width * y) + x)):(4 * ((self.width * y) + x + 1))])
    
    def __setitem__(self, key: tuple[int, int], value: tuple[int, int, int, int]):
        x, y = key
        r, g, b, a = value
        self.pixels[(4 * ((self.width * y) + x)):(4 * ((self.width * y) + x + 1))] = (r, g, b, a)

    def render(self, structure: TreeStructure):
        '''
        Renders the structure of a tree.
        '''
        for k in range(len(self.renderers)):
            self.renderers[k].render(structure, self)

    def save(self, filename: str):
        '''
        Saves the image to a file.

        Args:
            filename (str): The file path.
        '''
        # Construct rows
        rows = [self.pixels[(4 * self.width * i):(4 * self.width * (i + 1))] for i in range(self.height)]
        # Save to file
        data = png.from_array(rows, 'RGBA')
        data.save(filename)

class Vertex:
    pos: Vector 
    '''
    The position of the vertex.
    '''

    normal: Vector
    '''
    The normal to the vertex.
    '''

    def __init__(self, pos: Vector, normal: Vector):
        self.pos = pos
        self.normal = normal

class TreeRenderer:
    zoom: float 
    x: int 
    y: int 
    width: int 
    height: int 

    def __init__(self, zoom: float = 1.0, x: int = 0, y: int = 0, width: int = 300, height: int = 400):
        self.zoom = zoom
        self.x = x 
        self.y = y
        self.width = width 
        self.height = height

    def create_program(self, ctx: moderngl.Context) -> moderngl.Program:
        '''
        Creates the program used for rendering.
        '''
        pass


    def geometrize_branch_segment(self, segment: TreeBranchSegment, next_segment: TreeBranchSegment | None) -> list[Vertex]:
        '''
        Constructs vertices for the geometry of the branch segment.
        '''
        dir0: Vector = segment.vec.normalize()
        orth00: Vector = dir0.cross(Vector.construct(depth=1.0))
        orth01: Vector = dir0.cross(orth00) # orth01 is orth00 rotated 90 degrees counterclockwise around dir
        radial_segments: int = 8
        angle: float = 0.0
        vertices: list[Vertex] = []

        # Create geometry of cylinder walls in radial segments
        if next_segment == None or segment.is_end_cap:
            # Don't consider segment after it, just make a straight cylinder
            while angle < math.tau and not np.isclose(angle, math.tau):
                # Determine the vertices of the two triangles composing this radial segment
                next_angle: float = angle + (math.tau / radial_segments)
                normal00: Vector = (math.cos(angle) * orth00) + (math.sin(angle) * orth01)
                normal01: Vector = (math.cos(next_angle) * orth00) + (math.sin(next_angle) * orth01)
                corner00: Vector = segment.start + (normal00 * segment.radius_base)
                corner01: Vector = segment.start + (normal01 * segment.radius_base)
                corner10: Vector = segment.start + segment.vec + (normal00 * segment.radius_end)
                corner11: Vector = segment.start + segment.vec + (normal01 * segment.radius_end)
                end_center: Vector = segment.start + segment.vec
                
                # Compose triangles from [corner00, corner01, corner10] and [corner11, corner10, corner01]
                vertices += [
                    Vertex(corner01, normal01),
                    Vertex(corner00, normal00),
                    Vertex(corner10, normal00)
                ]
                vertices += [
                    Vertex(corner10, normal00),
                    Vertex(corner11, normal01),
                    Vertex(corner01, normal01)
                ]

                # Compose end cap triangle from [corner10, corner11, end_center]
                vertices += [
                    Vertex(corner11, dir0),
                    Vertex(corner10, dir0),
                    Vertex(end_center, dir0)
                ]

                # Advance iteration
                angle = next_angle
        else:
            dir1: Vector = next_segment.vec.normalize()
            orth10: Vector = dir1.cross(Vector.construct(depth=1.0))
            orth11: Vector = dir1.cross(orth10)

            while angle < math.tau and not np.isclose(angle, math.tau):
                # Determine the vertices of the two triangles composing this radial segment
                next_angle: float = angle + (math.tau / radial_segments)
                normal00: Vector = (math.cos(angle) * orth00) + (math.sin(angle) * orth01)
                normal01: Vector = (math.cos(next_angle) * orth00) + (math.sin(next_angle) * orth01)
                normal10: Vector = (math.cos(angle) * orth10) + (math.sin(angle) * orth11)
                normal11: Vector = (math.cos(next_angle) * orth10) + (math.sin(next_angle) * orth11)
                corner00: Vector = segment.start + (normal00 * segment.radius_base)
                corner01: Vector = segment.start + (normal01 * segment.radius_base)
                corner10: Vector = segment.start + segment.vec + (normal10 * segment.radius_end)
                corner11: Vector = segment.start + segment.vec + (normal11 * segment.radius_end)
                
                # Compose triangles from [corner00, corner01, corner10] and [corner11, corner10, corner01]
                vertices += [
                    Vertex(corner01, normal01),
                    Vertex(corner00, normal00),
                    Vertex(corner10, normal10),
                    Vertex(corner10, normal10),
                    Vertex(corner11, normal11),
                    Vertex(corner01, normal01)
                ]

                # Advance iteration
                angle = next_angle

        return vertices

    def render(self, structure: TreeStructure, img: Image):
        '''
        Renders the tree to a PNG file.

        Args:
            structure (TreeStructure): The structure of the tree to render.
            filename (str): The filename to render to.
        '''
        # Set up the context
        ctx: moderngl.Context = moderngl.create_context(standalone=True)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        prog: moderngl.Program = self.create_program(ctx)

        # Initialize the projection matrix
        width = img.width if self.width == 0 else self.width 
        height = img.height if self.height == 0 else self.height 
        prog['proj_matrix'].write(
            np.array([
                [2.0 * self.zoom / width, 0.0, 0.0, 0.0],
                [0.0, -2.0 * self.zoom / height, 0.0, 0.0],
                [0.0, 0.0, 0.002, 0.0],
                [0.0, 1.0 - (15.0 * self.zoom / height), 0.0, 1.0]
            ]).astype('f4').tobytes()
        )

        # Construct the vertices
        vertices: list[Vertex] = []
        for k in range(len(structure.branch_segments)):
            vertices += self.geometrize_branch_segment(structure.branch_segments[k], structure.branch_segments[k + 1] if k < len(structure.branch_segments) - 1 else None)
        vertex_data = np.dstack(
            [
                [vertices[k].pos.horizontal for k in range(len(vertices))],
                [vertices[k].pos.vertical for k in range(len(vertices))],
                [vertices[k].pos.depth for k in range(len(vertices))],
                [vertices[k].normal.vector[0] for k in range(len(vertices))],
                [vertices[k].normal.vector[1] for k in range(len(vertices))],
                [vertices[k].normal.vector[2] for k in range(len(vertices))]
            ]
        )
        vbo: moderngl.Buffer = ctx.buffer(vertex_data.astype('f4').tobytes())
        vao: moderngl.VertexArray = ctx.simple_vertex_array(prog, vbo, 'in_pos', 'in_normal')

        # Do the rendering in the standalone context
        fbo: moderngl.Framebuffer = ctx.simple_framebuffer((width, height))
        fbo.use()
        fbo.clear()
        vao.render(moderngl.TRIANGLES)

        # Write the render onto a PNG image
        pixels: np.typing.NDArray[np.floating[Any]] = np.frombuffer(fbo.read(components=4, dtype='f4'), dtype='f4').reshape((width * height, 4))
        x: int = 0
        y: int = 0
        for pixel in pixels:
            r: int = min(255, int(math.floor(pixel[0] * 256)))
            g: int = min(255, int(math.floor(pixel[1] * 256)))
            b: int = min(255, int(math.floor(pixel[2] * 256)))
            a: int = min(255, int(math.floor(pixel[3] * 256)))
            img[(x, y)] = (r, g, b, a)

            if x + 1 == width:
                x = 0
                y += 1
            else:
                x += 1

class TreeSimpleRenderer(TreeRenderer):
    '''
    A simple renderer that renders a tree in grayscale.
    '''

    def __init__(self, zoom = 1, x = 0, y = 0, width = 300, height = 400):
        super().__init__(zoom, x, y, width, height)

    def create_program(self, ctx):
        return ctx.program(
            vertex_shader='''
#version 330

uniform mat4 proj_matrix;
in vec3 in_pos;
in vec3 in_normal;

void main() {
    gl_Position = proj_matrix * vec4(in_pos, 1.0);
}
''',
            fragment_shader='''
#version 330

out vec4 f_color;

void main() {
    f_color = vec4(1.0, 1.0, 1.0, 1.0);
}
'''
        )

class TreeNormalmapRenderer(TreeRenderer):
    '''
    A normalmap renderer for a tree.
    '''

    def __init__(self, zoom = 1, x = 0, y = 0, width = 300, height = 400):
        super().__init__(zoom, x, y, width, height)

    def create_program(self, ctx):
        return ctx.program(
            vertex_shader='''
#version 330

uniform mat4 proj_matrix;
in vec3 in_pos;
in vec3 in_normal;
out vec3 v_normal;

void main() {
    gl_Position = proj_matrix * vec4(in_pos, 1.0);
    v_normal = in_normal;
}
''',
            fragment_shader='''
#version 330

in vec3 v_normal;
out vec4 f_color;

void main() {
    f_color = vec4(0.5 * (vec3(1.0, 1.0, 1.0) + v_normal), 1.0);
}
'''
        )