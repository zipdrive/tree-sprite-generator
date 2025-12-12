import functools
from PIL import Image 
import moderngl 
import numpy as np
from sklearn import cluster
from typing import Any, Literal

original_file: str = 'assets/ash/leaf_untreated.png'
output_file: str = 'assets/ash/leaf.png'

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
colors = sorted(colors, key=lambda centroid: functools.reduce(lambda a, b: a + b, [np.linalg.norm(pixel - centroid) for pixel in original_pixel_data], 0.0))

final_colors = []
final_color_mappings = []

line01 = colors[1] - colors[0]
line02 = colors[2] - colors[0]
colinearity_01_02 = test_colinear(line01, line02)
if colinearity_01_02 == 'not colinear':
    for i in range(0, 2):
        for j in range(i + 1, 3):
            if len(final_colors) == 0:
                k: int = 3 - (i + j)
                lineij = colors[j] - colors[i]
                linei3 = colors[3] - colors[i]
                colinearity_ij_i3 = test_colinear(lineij, linei3)
                if colinearity_ij_i3 == 'not colinear':
                    continue 

                orth = colors[k]
                orth_mapping: np.typing.NDArray[np.floating[Any]]
                end1: np.typing.NDArray[np.floating[Any]]
                end1_mapping: np.typing.NDArray[np.floating[Any]]
                end2: np.typing.NDArray[np.floating[Any]]
                end2_mapping: np.typing.NDArray[np.floating[Any]]
                middle: np.typing.NDArray[np.floating[Any]]

                if colinearity_ij_i3 == 'start in middle':
                    end1 = colors[j]
                    end2 = colors[3]
                    end2_mapping = np.array([0.0, 0.0, 1.0, 1.0])
                    if j < k:
                        end1_mapping = np.array([1.0, 0.0, 0.0, 1.0])
                        orth_mapping = np.array([0.0, 1.0, 0.0, 1.0])
                    else:
                        orth_mapping = np.array([1.0, 0.0, 0.0, 1.0])
                        end1_mapping = np.array([0.0, 1.0, 0.0, 1.0])
                elif colinearity_ij_i3 == 'a_end in middle':
                    end1 = colors[i]
                    end2 = colors[3]
                    end2_mapping = np.array([0.0, 0.0, 1.0, 1.0])
                    if i < k:
                        end1_mapping = np.array([1.0, 0.0, 0.0, 1.0])
                        orth_mapping = np.array([0.0, 1.0, 0.0, 1.0])
                    else:
                        orth_mapping = np.array([1.0, 0.0, 0.0, 1.0])
                        end1_mapping = np.array([0.0, 1.0, 0.0, 1.0])
                elif colinearity_ij_i3 == 'b_end in middle':
                    end1 = colors[i]
                    end2 = colors[j]
                    if j < k:
                        end1_mapping = np.array([1.0, 0.0, 0.0, 1.0])
                        end2_mapping = np.array([0.0, 1.0, 0.0, 1.0])
                        orth_mapping = np.array([0.0, 0.0, 1.0, 1.0])
                    elif i < k:
                        end1_mapping = np.array([1.0, 0.0, 0.0, 1.0])
                        orth_mapping = np.array([0.0, 1.0, 0.0, 1.0])
                        end2_mapping = np.array([0.0, 0.0, 1.0, 1.0])
                    else:
                        orth_mapping = np.array([1.0, 0.0, 0.0, 1.0])
                        end1_mapping = np.array([0.0, 1.0, 0.0, 1.0])
                        end2_mapping = np.array([0.0, 0.0, 1.0, 1.0])
                
                middle_blend = np.linalg.norm(middle - end1) / np.linalg.norm(end2 - end1)
                final_colors = [
                    end1,
                    end2,
                    orth,
                    middle
                ]
                final_color_mappings = [
                    end1_mapping, 
                    end2_mapping,
                    orth_mapping,
                    end1_mapping + ((end2_mapping - end1_mapping) * middle_blend)
                ]
    
    if len(final_colors) == 0:
        final_colors = colors 
        final_color_mappings = [
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 0.0, 1.0])
        ]
else:
    intermediate_colors = []
    middle1: np.typing.NDArray[np.floating[Any]]
    if colinearity_01_02 == 'start in middle':
        intermediate_colors.append(colors[1])
        intermediate_colors.append(colors[2])
        middle1 = colors[0]
    elif colinearity_01_02 == 'a_end in middle':
        intermediate_colors.append(colors[0])
        intermediate_colors.append(colors[2])
        middle1 = colors[1]
    elif colinearity_01_02 == 'b_end in middle':
        intermediate_colors.append(colors[0])
        intermediate_colors.append(colors[1])
        middle1 = colors[2]
    
    line01 = intermediate_colors[1] - intermediate_colors[0]
    line03 = colors[3] - intermediate_colors[0]
    colinearity_01_03 = test_colinear(line01, line03)
    if colinearity_01_03 == 'not colinear':
        middle1_blend: float = np.linalg.norm(middle1 - intermediate_colors[0]) / np.linalg.norm(intermediate_colors[1] - intermediate_colors[0])
        final_colors = [intermediate_colors[0], intermediate_colors[1], colors[3], middle1]
        final_color_mappings = [
            np.array([1.0, 0.0, 0.0, 1.0]), 
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([1.0 - middle1_blend, middle1_blend, 0.0, 1.0])
        ] 
    else:
        end1: np.typing.NDArray[np.floating[Any]]
        end2: np.typing.NDArray[np.floating[Any]]
        middle2: np.typing.NDArray[np.floating[Any]]
        if colinearity_01_03 == 'start in middle':
            end1, end2 = intermediate_colors[1], colors[3]
            middle2 = intermediate_colors[0]
        elif colinearity_01_03 == 'a_end in middle':
            end1, end2 = intermediate_colors[0], colors[3]
            middle2 = intermediate_colors[1]
        elif colinearity_01_03 == 'b_end in middle':
            end1, end2 = intermediate_colors[0], intermediate_colors[1]
            middle2 = colors[3]
        middle1_blend: float = np.linalg.norm(middle1 - end1) / np.linalg.norm(end2 - end1)
        middle2_blend: float = np.linalg.norm(middle2 - end1) / np.linalg.norm(end2 - end1)
        final_colors = [end1, end2, middle1, middle2]
        final_color_mappings = [
            np.array([1.0, 0.0, 0.0, 1.0]), 
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([1.0 - middle1_blend, middle1_blend, 0.0, 1.0]),
            np.array([1.0 - middle2_blend, middle2_blend, 0.0, 1.0])
        ] 
    

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
'''
#version 330

uniform sampler2D tex;

uniform vec4 colors[4]; // The posterization colors.
uniform vec4 mapped_colors[4]; // The colors that they map to.

in vec2 v_uv;
out vec4 f_color;

void main() {
    vec4 sampled_color = texture(tex, v_uv);
    int best_k = 0;
    float best_k_distance = length(colors[0] - sampled_color);
    for (int k = 1; k < 4; ++k) {
        float k_distance = length(colors[k] - sampled_color);
        if (k_distance < best_k_distance) {
            best_k = k;
            best_k_distance = k_distance;
        }
    }

    f_color = vec4(mapped_colors[best_k].rgb, sampled_color.a);
}
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