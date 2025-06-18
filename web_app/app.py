import sys
import os
import streamlit as st
import torch
import numpy as np
import trimesh
from skimage import measure
from scipy.ndimage import binary_closing
sys.path.append(os.path.abspath('../src'))
from autoencoder_model import Decoder  # Your trained model
import plotly.graph_objects as go

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load decoder
decoder = Decoder()
decoder.load_state_dict(torch.load("../checkpoints/decoder.pth", map_location=device))
decoder.to(device)
decoder.eval()

# Load latent vectors
latents = {
    "grid": torch.load("../latents/grid.pt"),
    "flowing": torch.load("../latents/flowing.pt"),
    "holes": torch.load("../latents/holes.pt"),
    "complex": torch.load("../latents/complex.pt"),
    "lines": torch.load("../latents/lines.pt"),
    "point": torch.load("../latents/point.pt"),
}

# Streamlit sliders
st.title("Latent Space 3D Morphing")
st.markdown("Drag sliders to morph between shape categories.")

weights = {
    k: st.slider(k.capitalize(), 0.0, 1.0, 0.0 if k != "complex" else 1.0)
    for k in latents
}

# Normalize weights
w_tensor = torch.tensor([weights[k] for k in latents])
if w_tensor.sum() == 0:
    w_tensor[0] = 1.0
w_tensor /= w_tensor.sum()

# Interpolate latent
final_latent = sum(w * latents[k] for w, k in zip(w_tensor, latents))

# Decode SDF
with torch.no_grad():
    voxels = decoder(final_latent.unsqueeze(0).to(device)).squeeze().cpu().numpy()

# Convert to mesh
# verts, faces, normals, _ = measure.marching_cubes(sdf, level=0)
# mesh = trimesh.Trimesh(vertices=verts, faces=faces)

level = (voxels.min() + voxels.max()) / 2.0

# Binarize voxels (adjust threshold if needed)
binary_voxels = voxels > level

# Apply morphological closing
closed_voxels = binary_closing(binary_voxels, structure=np.ones((1,1,1)))

# Convert voxel to mesh
verts, faces, normals, _ = measure.marching_cubes(closed_voxels.astype(np.float32), level=0.5)

mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

# Split into components and keep the largest
components = mesh.split(only_watertight=False)
mesh = max(components, key=lambda m: m.vertices.shape[0])

# Plotly 3D viewer
x, y, z = verts.T
i, j, k = faces.T

fig = go.Figure(
    data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color="lightblue",
            opacity=0.8,
            flatshading=True,
            lighting=dict(ambient=0.3, diffuse=0.6, specular=0.5),
        )
    ]
)
fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig)

# Optional download
st.download_button("Download OBJ", mesh.export(file_type='obj'), file_name="generated_model.obj")