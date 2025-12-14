from __future__ import annotations
import math
import random
import numpy as np
import png
from PIL import Image as PILImage
import moderngl
from typing import Any, Generator
from util import Vector
from structure import TreeBranchSegment, TreeLeaf, TreeStructure

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

    def __init__(self, renderers: list[TreeRenderer]):
        '''
        Constructs an empty image.
        '''
        self.renderers = renderers
        self.width = 0
        self.height = 0
        for k in range(len(renderers)):
            self.width = max(self.width, renderers[k].x + renderers[k].width)
            self.height = max(self.height, renderers[k].y + renderers[k].height)
        self.pixels = [0 for _ in range(4 * self.width * self.height)]

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

    leaf_texture_ratio: float = 1.0
    '''
    The ratio of height:width of the leaf texture.
    '''

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

    def create_leaf_program(self, ctx: moderngl.Context) -> moderngl.Program | None:
        '''
        Creates the program used for rendering leaves.
        '''
        return None

    def render(self, structure: TreeStructure, img: Image):
        '''
        Renders the tree to a PNG file.

        Args:
            structure (TreeStructure): The structure of the tree to render.
            filename (str): The filename to render to.
        '''
        # Set up the context
        ctx: moderngl.Context = moderngl.create_context(standalone=True)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        branch_prog: moderngl.Program = self.create_branch_program(ctx)

        # Initialize the projection matrix
        width = img.width if self.width == 0 else self.width 
        height = img.height if self.height == 0 else self.height 
        branch_prog['proj_matrix'].write(
            np.array([
                [2.0 * self.zoom / width, 0.0, 0.0, 0.0],
                [0.0, -2.0 * self.zoom / height, 0.0, 0.0],
                [0.0, -1.0 * self.zoom / height, 0.002, 0.0],
                [0.0, 1.0 - (structure.branch_segments[0].radius_base * self.zoom / height), 0.0, 1.0]
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
            radial_segments: int = max(8, int(round(segment.radius_base / 3)))
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
        branch_vertices: list[Vertex] = []
        for k in range(len(structure.branch_segments)):
            branch_vertices += geometrize_branch_segment(structure.branch_segments[k])
        
        branch_prog_attrs: list = []
        if 'in_pos' in branch_prog:
            vertex_data = np.dstack(
                [
                    [branch_vertices[k].pos.horizontal for k in range(len(branch_vertices))],
                    [branch_vertices[k].pos.vertical for k in range(len(branch_vertices))],
                    [branch_vertices[k].pos.depth for k in range(len(branch_vertices))]
                ]
            )
            vbo: moderngl.Buffer = ctx.buffer(vertex_data.astype('f4').tobytes())
            branch_prog_attrs.append((vbo, '3f', 'in_pos'))
        if 'in_normal' in branch_prog:
            vertex_data = np.dstack(
                [
                    [branch_vertices[k].normal.horizontal for k in range(len(branch_vertices))],
                    [branch_vertices[k].normal.depth for k in range(len(branch_vertices))],
                    [branch_vertices[k].normal.vertical for k in range(len(branch_vertices))]
                ]
            )
            vbo: moderngl.Buffer = ctx.buffer(vertex_data.astype('f4').tobytes())
            branch_prog_attrs.append((vbo, '3f', 'in_normal'))
        if 'in_uv' in branch_prog:
            vertex_data = np.dstack(
                [
                    [branch_vertices[k].uv[0] for k in range(len(branch_vertices))],
                    [branch_vertices[k].uv[1] for k in range(len(branch_vertices))]
                ]
            )
            vbo: moderngl.Buffer = ctx.buffer(vertex_data.astype('f4').tobytes())
            branch_prog_attrs.append((vbo, '2f', 'in_uv'))
        branch_vao: moderngl.VertexArray = ctx.vertex_array(branch_prog, branch_prog_attrs)

        # Construct program and vertices for leaves
        leaf_prog: moderngl.Program | None = None
        leaf_vao: moderngl.VertexArray | None = None 
        if len(structure.leaves) > 0:
            leaf_prog = self.create_leaf_program(ctx)
        if leaf_prog != None:
            # Set the projection matrix
            leaf_prog['proj_matrix'].write(branch_prog['proj_matrix'].read())

            # Create the leaf geometry
            def geometrize_leaf(leaf: TreeLeaf) -> list[Vertex]:
                normal: Vector = Vector.construct(depth=1.0)
                axis1: Vector = leaf.dir
                axis2: Vector = normal.cross(axis1)
                normal = axis2.cross(axis1)
                return [
                    Vertex(
                        pos=leaf.anchor - (0.5 * leaf.size * axis2),
                        normal=normal,
                        uv=(0.0, 1.0)
                    ),
                    Vertex(
                        pos=leaf.anchor + (leaf.size * axis1) + (0.5 * leaf.size * axis2),
                        normal=normal,
                        uv=(1.0, 0.0)
                    ),
                    Vertex(
                        pos=leaf.anchor + (0.5 * leaf.size * axis2),
                        normal=normal,
                        uv=(1.0, 1.0)
                    ),
                    Vertex(
                        pos=leaf.anchor - (0.5 * leaf.size * axis2),
                        normal=normal,
                        uv=(0.0, 1.0)
                    ),
                    Vertex(
                        pos=leaf.anchor + (leaf.size * axis1) - (0.5 * leaf.size * axis2),
                        normal=normal,
                        uv=(0.0, 0.0)
                    ),
                    Vertex(
                        pos=leaf.anchor + (leaf.size * axis1) + (0.5 * leaf.size * axis2),
                        normal=normal,
                        uv=(1.0, 0.0)
                    ),
                ]
            
            leaf_vertices: list[Vertex] = []
            for k in range(len(structure.leaves)):
                leaf_vertices += geometrize_leaf(structure.leaves[k])

            leaf_prog_attrs: list = []
            if 'in_pos' in leaf_prog:
                vertex_data = np.dstack(
                    [
                        [leaf_vertices[k].pos.horizontal for k in range(len(leaf_vertices))],
                        [leaf_vertices[k].pos.vertical for k in range(len(leaf_vertices))],
                        [leaf_vertices[k].pos.depth for k in range(len(leaf_vertices))]
                    ]
                )
                vbo: moderngl.Buffer = ctx.buffer(vertex_data.astype('f4').tobytes())
                leaf_prog_attrs.append((vbo, '3f', 'in_pos'))
            if 'in_normal' in leaf_prog:
                vertex_data = np.dstack(
                    [
                        [leaf_vertices[k].normal.horizontal for k in range(len(leaf_vertices))],
                        [leaf_vertices[k].normal.depth for k in range(len(leaf_vertices))],
                        [leaf_vertices[k].normal.vertical for k in range(len(leaf_vertices))]
                    ]
                )
                vbo: moderngl.Buffer = ctx.buffer(vertex_data.astype('f4').tobytes())
                leaf_prog_attrs.append((vbo, '3f', 'in_normal'))
            if 'in_uv' in leaf_prog:
                vertex_data = np.dstack(
                    [
                        [leaf_vertices[k].uv[0] for k in range(len(leaf_vertices))],
                        [leaf_vertices[k].uv[1] for k in range(len(leaf_vertices))]
                    ]
                )
                vbo: moderngl.Buffer = ctx.buffer(vertex_data.astype('f4').tobytes())
                leaf_prog_attrs.append((vbo, '2f', 'in_uv'))
            leaf_vao = ctx.vertex_array(leaf_prog, leaf_prog_attrs)
            
        # Do the rendering in the standalone context
        fbo: moderngl.Framebuffer = ctx.simple_framebuffer((width, height))
        fbo.use()
        fbo.clear()
        branch_vao.render(moderngl.TRIANGLES)
        if leaf_vao != None:
            leaf_vao.render(moderngl.TRIANGLES)

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
    
class TreeColormapRenderer(TreeRenderer):
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
    vec2 uv = vec2(v_uv.x + sqrt(0.001) * v_uv.y, v_uv.y);
    vec4 sampled_color = texture(barkmap_tex, uv);
    f_color = sampled_color;
}
'''
        )

        # Load the barkmap texture
        barkmap_tex: moderngl.Texture = read_file_into_texture(ctx, filename=self.barkmap)
        barkmap_sampler2D: moderngl.Sampler = ctx.sampler(texture=barkmap_tex, filter=(moderngl.NEAREST, moderngl.NEAREST), min_lod=0, max_lod=0)
        prog['barkmap_tex'] = 0
        barkmap_sampler2D.use(0)
        return prog 
    
class TreeLeafColormapRenderer(TreeColormapRenderer):
    '''
    A renderer that renders flat bark and leaf textures.
    '''

    leafmap: str
    '''
    The filename of the leaf texture.
    '''

    def __init__(self, barkmap: str, leafmap: str, zoom = 1, x = 0, y = 0, width = 300, height = 400):
        super().__init__(barkmap, zoom, x, y, width, height)
        self.leafmap = leafmap

    def create_leaf_program(self, ctx):
        prog: moderngl.Program = ctx.program(
            vertex_shader=
'''
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
            fragment_shader=
'''
#version 330

uniform sampler2D leafmap_tex;
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec4 sampled_color = texture(leafmap_tex, v_uv);
    f_color = sampled_color;
}
'''
        )

        # Load the leafmap texture
        img: PILImage.Image = PILImage.open(self.leafmap).convert('RGBA')
        self.leaf_texture_ratio = img.height / img.width
        leafmap_tex: moderngl.Texture = ctx.texture((img.width, img.height), 4, data=img.tobytes())
        leafmap_sampler2D: moderngl.Sampler = ctx.sampler(texture=leafmap_tex)
        prog['leafmap_tex'] = 1
        leafmap_sampler2D.use(1)
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

#define PI 3.1415926538

uniform sampler2D normalmap_tex;
uniform sampler2D heightmap_tex;
uniform sampler2D noise_tex;
in vec3 v_normal;
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec2 uv = vec2(v_uv.x + sqrt(0.001) * v_uv.y, v_uv.y);
    float noise = texture(noise_tex, uv).r;
    
    vec3 sampled_normal = vec3(0.0, 1.0, 0.0);// normalize(texture(normalmap_tex, uv).xzy - 0.5);
    sampled_normal.y = -sampled_normal.y;

    float v_normal_xy_angle = atan(v_normal.x, -v_normal.y) + (noise * PI / 8.0) - (PI / 16.0);
    float v_normal_xy_magn = length(v_normal.xy);
    float v_normal_z_angle = atan(v_normal.z, v_normal_xy_magn);
    float v_normal_posterized_xy_angle = round(v_normal_xy_angle * 4.0 / PI) * PI / 4.0;
    float v_normal_posterized_z_angle = round(v_normal_z_angle * 4.0 / PI) * PI / 4.0;
    vec3 surface_normal = normalize(vec3(sin(v_normal_posterized_xy_angle), -cos(v_normal_posterized_xy_angle), sin(v_normal_posterized_z_angle)));

    vec3 axis = cross(vec3(1.0, 0.0, 0.0), surface_normal);
    vec3 sampled_normal_parallel = axis * dot(axis, sampled_normal);
    vec3 sampled_normal_perpendicular = sampled_normal - sampled_normal_parallel;
    vec3 true_normal = sampled_normal_parallel + (length(sampled_normal_perpendicular) * surface_normal);

    float sampled_height = texture(heightmap_tex, uv).r;

    f_color = vec4(0.5 * (vec3(1.0, 1.0, 1.0) + true_normal * clamp(2.0 * sampled_height, 0.0, 1.0)), 1.0);
}
'''
        )

        # Load the normalmap texture
        normalmap_tex: moderngl.Texture = read_file_into_texture(ctx, filename=self.normalmap)
        normalmap_sampler2D: moderngl.Sampler = ctx.sampler(texture=normalmap_tex, filter=(moderngl.NEAREST, moderngl.NEAREST), min_lod=0, max_lod=0)
        #prog['normalmap_tex'] = 0
        normalmap_sampler2D.use(0)

        # Load the depthmap texture
        heightmap_tex: moderngl.Texture = read_file_into_texture(ctx, filename=self.heightmap)
        heightmap_sampler2D: moderngl.Sampler = ctx.sampler(texture=heightmap_tex, filter=(moderngl.NEAREST, moderngl.NEAREST), min_lod=0, max_lod=0)
        prog['heightmap_tex'] = 1
        heightmap_sampler2D.use(1)

        # Load the noise texture
        noise_tex: moderngl.Texture = read_file_into_texture(ctx, filename='assets/noise.png')
        noise_sampler2D: moderngl.Sampler = ctx.sampler(texture=noise_tex, filter=(moderngl.NEAREST, moderngl.NEAREST), min_lod=0, max_lod=0)
        prog['noise_tex'] = 4
        noise_sampler2D.use(4)

        return prog 
    
class TreeLeafNormalmapRenderer(TreeNormalmapRenderer):
    '''
    A renderer that renders the normalmap for bark and leaf textures.
    '''

    leafmap: str
    '''
    The filename of the leaf texture.
    '''

    def __init__(self, bark_normalmap: str, bark_heightmap: str, leafmap: str, zoom = 1, x = 0, y = 0, width = 300, height = 400):
        super().__init__(bark_normalmap, bark_heightmap, zoom, x, y, width, height)
        self.leafmap = leafmap

    def create_leaf_program(self, ctx):
        prog: moderngl.Program = ctx.program(
            vertex_shader=
'''
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
            fragment_shader=
'''
#version 330

uniform sampler2D leafmap_tex;
in vec3 v_normal;
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec4 sampled_color = texture(leafmap_tex, v_uv);
    f_color = vec4(0.5 * (vec3(1.0, 1.0, 1.0) + v_normal), sampled_color.a);
}
'''
        )

        # Load the leafmap texture
        img: PILImage.Image = PILImage.open(self.leafmap).convert('RGBA')
        self.leaf_texture_ratio = img.height / img.width
        leafmap_tex: moderngl.Texture = ctx.texture((img.width, img.height), 4, data=img.tobytes())
        leafmap_sampler2D: moderngl.Sampler = ctx.sampler(texture=leafmap_tex)
        prog['leafmap_tex'] = 2
        leafmap_sampler2D.use(2)
        return prog 
    
class TreeSampleRenderer(TreeRenderer):
    '''
    Renders what the tree will look like with a color palette and lighting applied.
    '''

    bark_colormap: str 
    '''
    The filename of the bark color texture.
    '''

    bark_normalmap: str 
    '''
    The filename of the bark normal texture.
    '''

    bark_heightmap: str 
    '''
    The filename of the bark height texture.
    '''

    leaf_colormap: str
    '''
    The filename of the leaf color texture.
    '''

    palette_matrix: np.typing.NDArray[np.floating[Any]]
    '''
    Matrix mapping the color palette of the tree.
    '''

    def __init__(self, bark_colormap: str, bark_normalmap: str, bark_heightmap: str, leaf_colormap: str, primary_bark_color: tuple[int, int, int], secondary_bark_color: tuple[int, int, int], leaf_color: tuple[int, int, int], zoom = 1, x = 0, y = 0, width = 300, height = 400):
        super().__init__(zoom, x, y, width, height)
        self.bark_colormap = bark_colormap
        self.bark_normalmap = bark_normalmap
        self.bark_heightmap = bark_heightmap
        self.leaf_colormap = leaf_colormap
        self.palette_matrix = np.array([
            [primary_bark_color[0] / 255.0, primary_bark_color[1] / 255.0, primary_bark_color[2] / 255.0, 0.0],
            [secondary_bark_color[0] / 255.0, secondary_bark_color[1] / 255.0, secondary_bark_color[2] / 255.0, 0.0],
            [leaf_color[0] / 255.0, leaf_color[1] / 255.0, leaf_color[2] / 255.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

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

#define PI 3.1415926538

uniform sampler2D colormap_tex;
uniform sampler2D normalmap_tex;
uniform sampler2D heightmap_tex;
uniform sampler2D noise_tex;
uniform mat4 palette;
in vec3 v_normal;
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec2 uv = vec2(v_uv.x + sqrt(0.001) * v_uv.y, v_uv.y);
    float noise = texture(noise_tex, uv).r;
    
    vec4 untransformed_sampled_color = texture(colormap_tex, uv);
    vec4 sampled_color = palette * vec4(untransformed_sampled_color.rgb, 1.0);
    vec3 sampled_normal = vec3(0.0, 1.0, 0.0);// normalize(texture(normalmap_tex, uv).xzy - 0.5);
    sampled_normal.y = -sampled_normal.y;

    float v_normal_xy_angle = atan(v_normal.x, -v_normal.y) + (noise * PI / 8.0) - (PI / 16.0);
    float v_normal_xy_magn = length(v_normal.xy);
    float v_normal_z_angle = atan(v_normal.z, v_normal_xy_magn);
    float v_normal_posterized_xy_angle = round(v_normal_xy_angle * 4.0 / PI) * PI / 4.0;
    float v_normal_posterized_z_angle = round(v_normal_z_angle * 4.0 / PI) * PI / 4.0;
    vec3 surface_normal = normalize(vec3(sin(v_normal_posterized_xy_angle), -cos(v_normal_posterized_xy_angle), sin(v_normal_posterized_z_angle)));

    vec3 axis = cross(vec3(1.0, 0.0, 0.0), surface_normal);
    vec3 sampled_normal_parallel = axis * dot(axis, sampled_normal);
    vec3 sampled_normal_perpendicular = sampled_normal - sampled_normal_parallel;
    vec3 true_normal = sampled_normal_parallel + (length(sampled_normal_perpendicular) * surface_normal);

    float sampled_height = texture(heightmap_tex, uv).r;
    float depth_shadow_factor = clamp(2.0 * sampled_height, 0.0, 1.0);

    vec3 ambient = vec3(15.0 / 255.0, 15.0 / 255.0, 105.0 / 255.0);
    f_color = vec4(ambient + (sampled_color.rgb - ambient) * dot(true_normal, normalize(vec3(-0.5, -1.0, 1.0))) * depth_shadow_factor, untransformed_sampled_color.a);
}
'''
        )

        # Set the color palette
        prog['palette'].write(self.palette_matrix.astype('f4').tobytes())

        # Load the colormap texture
        colormap_tex: moderngl.Texture = read_file_into_texture(ctx, filename=self.bark_colormap)
        colormap_sampler2D: moderngl.Sampler = ctx.sampler(texture=colormap_tex, filter=(moderngl.NEAREST, moderngl.NEAREST), min_lod=0, max_lod=0)
        prog['colormap_tex'] = 0
        colormap_sampler2D.use(0)

        # Load the normalmap texture
        normalmap_tex: moderngl.Texture = read_file_into_texture(ctx, filename=self.bark_normalmap)
        normalmap_sampler2D: moderngl.Sampler = ctx.sampler(texture=normalmap_tex, filter=(moderngl.NEAREST, moderngl.NEAREST), min_lod=0, max_lod=0)
        #prog['normalmap_tex'] = 1
        normalmap_sampler2D.use(1)

        # Load the depthmap texture
        heightmap_tex: moderngl.Texture = read_file_into_texture(ctx, filename=self.bark_heightmap)
        heightmap_sampler2D: moderngl.Sampler = ctx.sampler(texture=heightmap_tex, filter=(moderngl.NEAREST, moderngl.NEAREST), min_lod=0, max_lod=0)
        prog['heightmap_tex'] = 2
        heightmap_sampler2D.use(2)

        # Load the noise texture
        noise_tex: moderngl.Texture = read_file_into_texture(ctx, filename='assets/noise.png')
        noise_sampler2D: moderngl.Sampler = ctx.sampler(texture=noise_tex, filter=(moderngl.NEAREST, moderngl.NEAREST), min_lod=0, max_lod=0)
        prog['noise_tex'] = 4
        noise_sampler2D.use(4)

        return prog 
    
    def create_leaf_program(self, ctx):
        prog: moderngl.Program = ctx.program(
            vertex_shader=
'''
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
            fragment_shader=
'''
#version 330

uniform sampler2D leafmap_tex;
uniform mat4 palette;
in vec3 v_normal;
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec4 untransformed_sampled_color = texture(leafmap_tex, v_uv);
    vec4 sampled_color = palette * vec4(untransformed_sampled_color.rgb, 1.0);

    f_color = vec4(sampled_color.rgb * dot(v_normal, normalize(vec3(-0.5, -1.0, 1.0))), untransformed_sampled_color.a);
}
'''
        )

        # Set the color palette
        prog['palette'].write(self.palette_matrix.astype('f4').tobytes())

        # Load the leafmap texture
        img: PILImage.Image = PILImage.open(self.leaf_colormap).convert('RGBA')
        self.leaf_texture_ratio = img.height / img.width
        leafmap_tex: moderngl.Texture = ctx.texture((img.width, img.height), 4, data=img.tobytes())
        leafmap_sampler2D: moderngl.Sampler = ctx.sampler(texture=leafmap_tex)
        prog['leafmap_tex'] = 3
        leafmap_sampler2D.use(3)
        return prog 