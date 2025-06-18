import os
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure
from tqdm import tqdm
import gc

# Parameters
GRID_SIZE = 32
SDF_CLIP = 0.1  # limit the signed distance range
SAVE_PATH = 'sdf_dataset'  # folder to save npz files

def mesh_to_sdf_grid(mesh, grid_size=GRID_SIZE):
    bounds = mesh.bounds
    min_bound = bounds[0] - 0.1
    max_bound = bounds[1] + 0.1
    coords = np.linspace(min_bound, max_bound, grid_size)
    X, Y, Z = np.meshgrid(*coords.T, indexing="ij")
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    sdf = mesh.nearest.signed_distance(points)
    sdf = sdf.reshape((grid_size, grid_size, grid_size))
    return np.clip(sdf, -SDF_CLIP, SDF_CLIP)

def process_folder(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    categories = os.listdir(input_root)
    for cat in categories:
        cat_path = os.path.join(input_root, cat)
        if not os.path.isdir(cat_path): continue
        save_dir = os.path.join(output_root, cat)
        os.makedirs(save_dir, exist_ok=True)

        for fname in tqdm(os.listdir(cat_path), desc=f"Processing {cat}"):
            if not fname.endswith('.obj'): continue
            fpath = os.path.join(cat_path, fname)
            mesh = trimesh.load(fpath)
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = mesh.dump().sum()
            sdf = mesh_to_sdf_grid(mesh)
            save_name = os.path.splitext(fname)[0] + '.npz'
            np.savez_compressed(os.path.join(save_dir, save_name), sdf=sdf)
            del mesh, sdf
            gc.collect()

if __name__ == "__main__":
    INPUT_DIR = 'your_meshes'  # change to your .obj folder
    OUTPUT_DIR = 'sdf_dataset'
    process_folder(INPUT_DIR, OUTPUT_DIR)
    print("✅ All meshes converted to SDFs!")