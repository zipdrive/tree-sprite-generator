from __future__ import annotations
import math
import random
import numpy as np
import png
from PIL import Image as PILImage
import moderngl
from typing import Any, Generator
from util import Vector
from structure import TreeBranchSegment, TreeStructure

def read_file_into_texture(ctx: moderngl.Context, filename: str) -> moderngl.Texture:
    '''
    Reads a texture file into a moderngl Texture object.
    '''
    img: PILImage.Image = PILImage.open(filename).convert('RGBA')
    return ctx.texture((img.width, img.height), 4, data=img.tobytes())

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

    uv: tuple[float, float]
    '''
    The vertex UV.
    '''

    def __init__(self, pos: Vector, normal: Vector, uv: tuple[float, float]):
        self.pos = pos
        self.normal = normal
        self.uv = uv

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

    def create_branch_program(self, ctx: moderngl.Context) -> moderngl.Program:
        '''
        Creates the program used for rendering branches.
        '''
        pass

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
        prog: moderngl.Program = self.create_branch_program(ctx)

        # Initialize the projection matrix
        width = img.width if self.width == 0 else self.width 
        height = img.height if self.height == 0 else self.height 
        prog['proj_matrix'].write(
            np.array([
                [2.0 * self.zoom / width, 0.0, 0.0, 0.0],
                [0.0, -2.0 * self.zoom / height, 0.0, 0.0],
                [0.0, -0.0005, 0.002, 0.0],
                [0.0, 1.0 - (15.0 * self.zoom / height), 0.0, 1.0]
            ]).astype('f4').tobytes()
        )

        branch_segment_uv: dict[TreeBranchSegment, tuple[float, float]] = {}
        def geometrize_branch_segment(segment: TreeBranchSegment) -> list[Vertex]:
            '''
            Constructs vertices for the geometry of the branch segment.
            '''
            dir0: Vector = segment.vec.normalize()
            orth00: Vector = dir0.cross(Vector.construct(depth=1.0))
            orth01: Vector = dir0.cross(orth00) # orth01 is orth00 rotated 90 degrees counterclockwise around dir
            radial_segments: int = 8
            angle: float = 0.0
            u: float
            v0: float
            u, v0 = branch_segment_uv[segment] if segment in branch_segment_uv else (random.random(), random.random())
            v1: float = v0 + (segment.vec.length / (math.tau * segment.radius_base))
            vertices: list[Vertex] = []

            # Create geometry of cylinder walls in radial segments
            if segment.next_segment == None:
                # Close end in a point
                while angle < math.tau and not np.isclose(angle, math.tau):
                    # Determine the vertices of the two triangles composing this radial segment
                    next_angle: float = angle + (math.tau / radial_segments)
                    normal00: Vector = (math.cos(angle) * orth00) + (math.sin(angle) * orth01)
                    normal01: Vector = (math.cos(next_angle) * orth00) + (math.sin(next_angle) * orth01)
                    corner00: Vector = segment.start + (normal00 * segment.radius_base)
                    corner01: Vector = segment.start + (normal01 * segment.radius_base)
                    corner1X: Vector = segment.start + segment.vec
                    diff00_1X: Vector = corner1X - corner00
                    diff01_1X: Vector = corner1X - corner01
                    tangent00: Vector = normal00.cross(diff00_1X)
                    tangent01: Vector = normal01.cross(diff01_1X)
                    normal00, normal01 = diff00_1X.cross(tangent00).normalize(), diff01_1X.cross(tangent01).normalize()
                    normal1X: Vector = dir0

                    # Determine the UV coordinates
                    u0: float = u
                    u1: float = u + (1.0 / radial_segments)
                    u = u1
                    
                    # Compose triangles from [corner00, corner01, corner10] and [corner11, corner10, corner01]
                    vertices += [
                        Vertex(corner01, normal01, (u1, v0)),
                        Vertex(corner00, normal00, (u0, v0)),
                        Vertex(corner1X, normal1X, (0.5 * (u0 + u1), v1))
                    ]

                    # Advance iteration
                    angle = next_angle
            else:
                dir1: Vector = segment.next_segment.vec.normalize()
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
                    
                    # Determine the UV coordinates
                    u0: float = u
                    u1: float = u + (1.0 / radial_segments)
                    u = u1
                    
                    # Compose triangles from [corner00, corner01, corner10] and [corner11, corner10, corner01]
                    vertices += [
                        Vertex(corner01, normal01, (u1, v0)),
                        Vertex(corner00, normal00, (u0, v0)),
                        Vertex(corner10, normal10, (u0, v1)),
                        Vertex(corner10, normal10, (u0, v1)),
                        Vertex(corner11, normal11, (u1, v1)),
                        Vertex(corner01, normal01, (u1, v0))
                    ]

                    # Advance iteration
                    angle = next_angle
                
                branch_segment_uv[segment.next_segment] = (u - math.floor(u), v1 - math.floor(v1))

            return vertices

        # Construct the vertices
        vertices: list[Vertex] = []
        for k in range(len(structure.branch_segments)):
            vertices += geometrize_branch_segment(structure.branch_segments[k])
        
        prog_attrs: list = []
        if 'in_pos' in prog:
            vertex_data = np.dstack(
                [
                    [vertices[k].pos.horizontal for k in range(len(vertices))],
                    [vertices[k].pos.vertical for k in range(len(vertices))],
                    [vertices[k].pos.depth for k in range(len(vertices))]
                ]
            )
            vbo: moderngl.Buffer = ctx.buffer(vertex_data.astype('f4').tobytes())
            prog_attrs.append((vbo, '3f', 'in_pos'))
        if 'in_normal' in prog:
            vertex_data = np.dstack(
                [
                    [vertices[k].normal.horizontal for k in range(len(vertices))],
                    [vertices[k].normal.depth for k in range(len(vertices))],
                    [vertices[k].normal.vertical for k in range(len(vertices))]
                ]
            )
            vbo: moderngl.Buffer = ctx.buffer(vertex_data.astype('f4').tobytes())
            prog_attrs.append((vbo, '3f', 'in_normal'))
        if 'in_uv' in prog:
            vertex_data = np.dstack(
                [
                    [vertices[k].uv[0] for k in range(len(vertices))],
                    [vertices[k].uv[1] for k in range(len(vertices))]
                ]
            )
            vbo: moderngl.Buffer = ctx.buffer(vertex_data.astype('f4').tobytes())
            prog_attrs.append((vbo, '2f', 'in_uv'))
        vao: moderngl.VertexArray = ctx.vertex_array(prog, prog_attrs)

        # Do the rendering in the standalone context
        fbo: moderngl.Framebuffer = ctx.simple_framebuffer((width, height))
        fbo.use()
        fbo.clear()
        vao.render(moderngl.TRIANGLES)

        # Write the render onto a PNG image
        pixels: np.typing.NDArray[np.floating[Any]] = np.frombuffer(fbo.read(components=4, dtype='f4'), dtype='f4').reshape((width * height, 4))
        x: int = self.x
        y: int = self.y
        for pixel in pixels:
            r: int = min(255, int(math.floor(pixel[0] * 256)))
            g: int = min(255, int(math.floor(pixel[1] * 256)))
            b: int = min(255, int(math.floor(pixel[2] * 256)))
            a: int = min(255, int(math.floor(pixel[3] * 256)))
            img[(x, y)] = (r, g, b, a)

            if x + 1 == self.x + width:
                x = self.x 
                y += 1
            else:
                x += 1

class TreeSimpleRenderer(TreeRenderer):
    '''
    A simple renderer that renders a tree in grayscale.
    '''

    def __init__(self, zoom = 1, x = 0, y = 0, width = 300, height = 400):
        super().__init__(zoom, x, y, width, height)

    def create_branch_program(self, ctx):
        return ctx.program(
            vertex_shader='''
#version 330

uniform mat4 proj_matrix;
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;

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
    
class TreeBarkRenderer(TreeRenderer):
    '''
    A renderer that renders the flat bark texture.
    '''

    barkmap: str 
    '''
    The filename of the bark texture.
    '''

    def __init__(self, barkmap: str, zoom = 1, x = 0, y = 0, width = 300, height = 400):
        super().__init__(zoom, x, y, width, height)
        self.barkmap = barkmap

    def create_branch_program(self, ctx):
        prog: moderngl.Program = ctx.program(
            vertex_shader='''
#version 330

uniform mat4 proj_matrix;
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
out vec2 v_uv;

void main() {
    gl_Position = proj_matrix * vec4(in_pos, 1.0);
    v_uv = in_uv;
}
''',
            fragment_shader='''
#version 330

uniform sampler2D barkmap_tex;
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec2 uv = vec2(v_uv.x + sqrt(3.0) * v_uv.y, v_uv.y);
    vec4 sampled_color = texture(barkmap_tex, uv);
    f_color = sampled_color;
}
'''
        )

        # Load the barkmap texture
        barkmap_tex: moderngl.Texture = read_file_into_texture(ctx, filename=self.barkmap)
        barkmap_sampler2D: moderngl.Sampler = ctx.sampler(texture=barkmap_tex)
        prog['barkmap_tex'] = 0
        barkmap_sampler2D.use(0)
        return prog 

class TreeNormalmapRenderer(TreeRenderer):
    '''
    A normalmap renderer for a tree.
    '''

    normalmap: str 
    '''
    The file for the normalmap texture.
    '''

    heightmap: str

    def __init__(self, normalmap: str, heightmap: str, zoom = 1, x = 0, y = 0, width = 300, height = 400):
        super().__init__(zoom, x, y, width, height)
        self.normalmap = normalmap
        self.heightmap = heightmap

    def create_branch_program(self, ctx):
        prog: moderngl.Program = ctx.program(
            vertex_shader='''
#version 330

uniform mat4 proj_matrix;
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    gl_Position = proj_matrix * vec4(in_pos, 1.0);
    v_normal = in_normal;
    v_uv = in_uv;
}
''',
            fragment_shader='''
#version 330

uniform sampler2D normalmap_tex;
uniform sampler2D heightmap_tex;
in vec3 v_normal;
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec2 uv = vec2(v_uv.x + sqrt(3.0) * v_uv.y, v_uv.y);
    vec3 sampled_normal = texture(normalmap_tex, uv).xyz;
    vec3 axis = cross(v_normal, sampled_normal);
    vec3 sampled_normal_parallel = axis * dot(axis, sampled_normal);
    vec3 sampled_normal_perpendicular = sampled_normal - sampled_normal_parallel;
    vec3 v_normal_parallel = axis * dot(axis, v_normal);
    vec3 v_normal_perpendicular = v_normal - v_normal_parallel;
    vec3 true_normal = sampled_normal_parallel + (length(sampled_normal_perpendicular) + normalize(v_normal_perpendicular));

    float sampled_height = texture(heightmap_tex, uv).r;

    f_color = vec4(0.5 * (vec3(1.0, 1.0, 1.0) + true_normal * clamp(2.0 * sampled_height, 0.0, 1.0)), 1.0);
}
'''
        )

        # Load the normalmap texture
        heightmap_tex: moderngl.Texture = read_file_into_texture(ctx, filename=self.normalmap)
        heightmap_sampler2D: moderngl.Sampler = ctx.sampler(texture=heightmap_tex)
        prog['normalmap_tex'] = 0
        heightmap_sampler2D.use(0)

        # Load the depthmap texture
        heightmap_tex: moderngl.Texture = read_file_into_texture(ctx, filename=self.heightmap)
        heightmap_sampler2D: moderngl.Sampler = ctx.sampler(texture=heightmap_tex)
        prog['heightmap_tex'] = 1
        heightmap_sampler2D.use(1)

        return prog 