import pickle
import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
import os
import sys
import palm_simulator


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ── Load dataset ─────────────────────────────────────────────────────────────
with open('data/simdata_size3.pkl', 'rb') as f:
    data = pickle.load(f)

# Load mesh connectivity from pickle (to get triangle surface)
with open('palm_size3.pickle', 'rb') as f:
    mesh_data = pickle.load(f)
tetrahedra = np.array(mesh_data['tetrahedra'])

# Build surface triangle list (unique triangles from all tets)
triangle_set = set()
for tet in tetrahedra:
    for i in range(4):
        for j in range(i+1, 4):
            for k in range(j+1, 4):
                tri = tuple(sorted([tet[i], tet[j], tet[k]]))
                triangle_set.add(tri)
triangle_list = np.array(list(triangle_set))

# ── Build displacement matrix ─────────────────────────────────────────────────
# Stack all deformed vertex arrays: shape (N, num_vertices, 3)
verts_all = np.array([d['deformed_vertices'] for d in data])  # (944, 241, 3)
N, nV, _ = verts_all.shape

# Mean shape (reference for displacement)
mean_verts = verts_all.mean(axis=0)  # (241, 3)

# Displacement matrix: (N, 241*3)
D = (verts_all - mean_verts).reshape(N, nV * 3)

# ── PCA ───────────────────────────────────────────────────────────────────────
n_modes = 10
pca = PCA(n_components=n_modes)
pca.fit(D)

modes = pca.components_          # (n_modes, 241*3)
explained = pca.explained_variance_ratio_

print("Explained variance ratio:")
for i in range(n_modes):
    print(f"  Mode {i+1}: {explained[i]*100:.2f}%  (cumulative: {explained[:i+1].sum()*100:.2f}%)")

# ── Build PyVista surface mesh helper ─────────────────────────────────────────
def make_surface_mesh(verts):
    """Build pyvista PolyData surface from vertices and triangle list."""
    faces = np.hstack([np.full((len(triangle_list), 1), 3), triangle_list]).astype(np.int_).flatten()
    return pv.PolyData(verts.astype(np.float32), faces)

# ── Visualize first 3 modes ───────────────────────────────────────────────────
# For each mode, show mean shape ± 2σ displacement coloured by mode displacement magnitude
n_show = 3
scale = 2.0  # ±2 std deviations along each mode

plotter = pv.Plotter(shape=(1, n_show), off_screen=False, window_size=(1800, 600))

for col in range(n_show):
    mode_vec = modes[col].reshape(nV, 3)          # (241, 3) displacement direction
    std = np.sqrt(pca.explained_variance_[col])   # std along this mode

    # Positive extreme shape
    verts_pos = mean_verts + scale * std * mode_vec
    # Negative extreme shape
    verts_neg = mean_verts - scale * std * mode_vec

    # Displacement magnitude at each vertex for colouring
    disp_mag = np.linalg.norm(mode_vec, axis=1)  # (241,)

    mesh_pos = make_surface_mesh(verts_pos)
    mesh_pos['disp_magnitude'] = disp_mag
    mesh_neg = make_surface_mesh(verts_neg)
    mesh_neg['disp_magnitude'] = disp_mag
    mesh_mean = make_surface_mesh(mean_verts)

    plotter.subplot(0, col)
    plotter.add_text(
        f"Mode {col+1}  ({explained[col]*100:.1f}%)",
        position='upper_edge', font_size=12, color='black'
    )
    # Show positive extreme (solid, coloured)
    plotter.add_mesh(
        mesh_pos, scalars='disp_magnitude', cmap='plasma',
        show_scalar_bar=(col == n_show - 1), opacity=0.85,
        scalar_bar_args={'title': 'Disp. magnitude (m)'}
    )
    # Show negative extreme as wireframe for reference
    plotter.add_mesh(mesh_neg, style='wireframe', color='steelblue', opacity=0.4, line_width=0.5)
    # Mean shape wireframe
    plotter.add_mesh(mesh_mean, style='wireframe', color='gray', opacity=0.2, line_width=0.3)
    plotter.view_isometric()

plotter.add_text(
    "Palm deformation modes (solid=+2σ, blue wireframe=−2σ, gray=mean)",
    position='lower_right', font_size=9, color='black'
)
plotter.show(title="Palm FEM Modal Analysis (PCA)")

pickleFilename = 'models/palm_size3.pickle'
palm = palm_simulator.PalmSimulator(pickleFilename)
mode_vec = modes[0].reshape(nV, 3) 
vert = mean_verts + 0.0 * std * mode_vec
palm.draw_mesh_with_initial(vert)
print("cable length for mode 1")
print(palm.get_cable_length(palm.vertices_2_Q(vert), palm.pulley_location))
