import torch
from autoencoder_model import Decoder
import numpy as np
from skimage import measure
import trimesh
import os
import open3d as o3d

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Load decoder
decoder = Decoder(latent_size=128).to(device)
decoder.load_state_dict(torch.load("../checkpoints/decoder.pth", map_location=device))
decoder.eval()

# Load average latents(adjust here)
base_dir = "../latents"
weights = {
    "flowing": 0.0,
    "holes": 0.8,
    "grid": 0.0,
    "complex": 0.8,
    "point": 0.0,
    "lines": 0.0,
}

latent = torch.zeros((128,), device=device)
for cat, weight in weights.items():
    vec = torch.load(f"{base_dir}/{cat}.pt", map_location=device)
    latent += weight * vec

# Decode
with torch.no_grad():
    voxels = decoder(latent.unsqueeze(0)).squeeze().cpu().numpy()

print("Voxel min:", voxels.min(), "Voxel max:", voxels.max())

level = (voxels.min() + voxels.max()) / 2.0
print(f"Auto-selected marching cubes level: {level}")


from scipy.ndimage import binary_closing

# Binarize voxels (adjust threshold if needed)
binary_voxels = voxels > level

# Apply morphological closing
closed_voxels = binary_closing(binary_voxels, structure=np.ones((1,1,1)))

# Use closed_voxels for marching cubes
verts, faces, normals, _ = measure.marching_cubes(closed_voxels.astype(np.float32), level=0.5)


# # Convert voxel to mesh(without postprocess)
# verts, faces, normals, _ = measure.marching_cubes(voxels, level=level)


mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
# Split into components and keep the largest
components = mesh.split(only_watertight=False)
mesh = max(components, key=lambda m: m.vertices.shape[0])

o3d_mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(mesh.vertices),
    o3d.utility.Vector3iVector(mesh.faces)
)
# # Smooth mesh
# o3d_mesh = o3d_mesh.filter_smooth_simple(number_of_iterations=5)
# mesh = trimesh.Trimesh(
#     vertices=np.asarray(o3d_mesh.vertices),
#     faces=np.asarray(o3d_mesh.triangles)
# )

# Save as OBJ
os.makedirs("generated_meshes", exist_ok=True)
mesh.export("generated_meshes/blended.obj")
print("✅ Mesh generated from blended latent vector.")