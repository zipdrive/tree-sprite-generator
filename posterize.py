import functools
from PIL import Image 
import moderngl 
import numpy as np
from sklearn import cluster
from typing import Any, Literal

original_file: str = 'assets/birch/wood_0027_color_1k.jpg'
output_file: str = 'assets/birch/color.png'

# Load the image
original_img: Image.Image = Image.open(original_file).convert('RGBA')
width: int = original_img.width
height: int = original_img.height

# Perform k-means clustering on the pixels of the image to get the colors that will be used for posterization
original_img_bytes: bytes = original_img.tobytes()
original_pixel_data: np.typing.NDArray[np.floating[Any]] = np.frombuffer(original_img_bytes, dtype=np.uint8).reshape(width * height, 4)
color_centroids, _, _ = cluster.k_means(original_pixel_data, n_clusters=4)

# Remove (more-or-less) colinear colors from the centroids
def test_colinear(a: np.typing.NDArray[np.floating[Any]], b: np.typing.NDArray[np.floating[Any]], tolerance=0.1) -> Literal['not colinear', 'start in middle', 'a_end in middle', 'b_end in middle']:
    a = a[0:3]
    b = b[0:3]
    a_length: float = np.linalg.norm(a)
    b_length: float = np.linalg.norm(b)
    a_norm = a / a_length
    b_norm = b / b_length
    perpendicularity: float = np.linalg.norm(np.cross(a_norm, b_norm))
    if perpendicularity < tolerance or np.isclose(perpendicularity, tolerance):
        if np.dot(a, b) < 0.0:
            return 'start in middle'
        elif a_length < b_length:
            return 'a_end in middle'
        else:
            return 'b_end in middle'
    else:
        return 'not colinear'

colors = [
    color_centroids[0, :] / 255.0,
    color_centroids[1, :] / 255.0,
    color_centroids[2, :] / 255.0,
    color_centroids[3, :] / 255.0
]

# Repeatedly reduce color set to a minimal set of non-colinear colors
final_colors = sorted(colors, key=lambda centroid: functools.reduce(lambda a, b: a + b, [np.linalg.norm(pixel - centroid) for pixel in original_pixel_data], 0.0))
def try_remove_colinear_color() -> bool:
    for k in range(2, len(final_colors)):
        for j in range(1, k):
            for i in range(0, j):
                lineij = final_colors[j] - final_colors[i]
                lineik = final_colors[k] - final_colors[i]
                colinearity_result = test_colinear(lineij, lineik)
                if colinearity_result == 'start in middle':
                    final_colors.pop(i)
                    return True 
                elif colinearity_result == 'a_end in middle':
                    final_colors.pop(j)
                    return True 
                elif colinearity_result == 'b_end in middle':
                    final_colors.pop(k)
                    return True 
    return False 
while try_remove_colinear_color():
    pass 
final_color_mappings = [
    np.array([1.0, 0.0, 0.0, 1.0]),
    np.array([0.0, 1.0, 0.0, 1.0]),
    np.array([0.0, 0.0, 1.0, 1.0]),
    np.array([0.0, 0.0, 0.0, 1.0])
][0:len(final_colors)]

if len(final_colors) == 2:
    final_colors.append(0.5 * (final_colors[0] + final_colors[1]))
    final_color_mappings.append(0.5 * (final_color_mappings[0] + final_color_mappings[1]))


# Render the transformed color space
ctx: moderngl.Context = moderngl.create_context(standalone=True)
ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
prog: moderngl.Program = ctx.program(
    vertex_shader= 
'''
#version 330

in vec2 in_uv;
out vec2 v_uv;

void main() {
    gl_Position = vec4((2.0 * in_uv) - 1.0, 0.0, 1.0);
    v_uv = in_uv;
}
''',
    fragment_shader=
f'''
#version 330

uniform sampler2D tex;

uniform vec4 colors[{len(final_colors)}]; // The posterization colors.
uniform vec4 mapped_colors[{len(final_color_mappings)}]; // The colors that they map to.

in vec2 v_uv;
out vec4 f_color;

void main() {{
    vec4 sampled_color = texture(tex, v_uv);
    int best_k = 0;
    float best_k_distance = length(colors[0] - sampled_color);
    for (int k = 1; k < {len(final_colors)}; ++k) {{
        float k_distance = length(colors[k] - sampled_color);
        if (k_distance < best_k_distance) {{
            best_k = k;
            best_k_distance = k_distance;
        }}
    }}

    f_color = vec4(mapped_colors[best_k].rgb, sampled_color.a);
}}
'''
)

# Bind each color
prog['colors'].write(np.array(final_colors).astype('f4').tobytes())
prog['mapped_colors'].write(np.array(final_color_mappings).astype('f4').tobytes())

# Construct and bind the texture
tex: moderngl.Texture = ctx.texture((original_img.width, original_img.height), 4, data=original_img.tobytes())
samp: moderngl.Sampler = ctx.sampler(texture=tex)
prog['tex'] = 0
samp.use(0)

# Construct and bind the vertex data
uv_data = np.dstack(
    [
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    ]
)
vbo: moderngl.Buffer = ctx.buffer(uv_data.astype('f4').tobytes())
vao: moderngl.VertexArray = ctx.vertex_array(prog, [(vbo, '2f', 'in_uv')])

# Do the rendering in the standalone context
fbo: moderngl.Framebuffer = ctx.simple_framebuffer((width, height))
fbo.use()
fbo.clear()
vao.render(moderngl.TRIANGLES)

# Save rendered image to PNG
output_bytes: bytes = fbo.read(components=4)
output_img: Image.Image = Image.frombytes(
    "RGBA", (width, height), output_bytes
)
output_img.save(output_file)
print("Done.")