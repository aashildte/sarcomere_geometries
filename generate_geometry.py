# create_geometry_with_nucleus.py
import os
import gmsh
import numpy as np
from gmshnics.interopt import msh_gmsh_model, mesh_from_gmsh
import dolfin as df
import matplotlib.pyplot as plt
import sys

# -------------------------
# Parameters
# -------------------------

cell_ID = sys.argv[1]
num_nuclei = int(sys.argv[2])
idealized = eval(sys.argv[3])

H = 1.0
zline_width = 0.2
row_offset = 1.2
sarco_length_mean = 2.0

if idealized:
    num_sarcomeres = 50
    num_rows = 15
    amp = 0
    phase_shift = np.pi / 3
    length_variation = 0.0
    length_variation_top = 0.0
    length_variation_bottom = 0.0
else:
    np.random.seed(int(sys.argv[1]))
    num_sarcomeres = np.random.randint(45, 57)
    num_rows = np.random.randint(14, 23)
    amp = np.random.uniform(0, 0.015)
    phase_shift = np.pi / 3
    length_variation = 0.005
    length_variation_top = 0.025
    length_variation_bottom = 0.025


mesh_size = 0.7
nucleus_mesh_size = 0.4
num_circle_pts = 50
sizefield_dist_max = 0.7

# Physical tags
ZLINE_TAG = 2000
CYTO_TAG = 3000
CONN_TAG = 4000
NUCLEUS1_TAG = 5001
NUCLEUS2_TAG = 5002

# -------------------------
# Init gmsh
# -------------------------
gmsh.initialize(['', '-v', '2'])
occ = gmsh.model.occ

# -------------------------
# Helper functions
# -------------------------
def find_line_between(p1, p2):
    for lid, (a, b) in line_endpoints.items():
        if (a == p1 and b == p2) or (a == p2 and b == p1):
            return lid
    return None

def safe_add_line(start_id, end_id):
    existing = find_line_between(start_id, end_id)
    if existing is not None:
        return existing
    s, e = np.array(coords_dict[start_id]), np.array(coords_dict[end_id])
    assert np.linalg.norm(e - s) > 1e-12
    for p in (start_id, end_id):
        line_count.setdefault(p, 0)
        line_count[p] += 1
        assert line_count[p] <= 4
    lid = gmsh.model.occ.add_line(start_id, end_id)
    line_endpoints[lid] = (start_id, end_id)
    return lid

def add_rectangle_from_ids(left_line_id, bl_id, br_id, tr_id, tl_id):
    right_line = safe_add_line(br_id, tr_id)
    top_line = safe_add_line(tl_id, tr_id)
    bottom_line = safe_add_line(bl_id, br_id)
    loop = gmsh.model.occ.add_curve_loop([left_line_id, top_line, right_line, bottom_line])
    surf = gmsh.model.occ.add_plane_surface([loop])
    corners = np.vstack([coords_dict[bl_id], coords_dict[br_id], coords_dict[tr_id], coords_dict[tl_id]])
    original_polygons[surf] = corners
    return surf, top_line, right_line, bottom_line

def surface_centroid_2d(tag):
    try:
        com3 = gmsh.model.occ.getCenterOfMass(2, tag)
        return np.array(com3[:2])
    except Exception:
        return None

def point_in_poly(point, poly):
    x, y = point
    verts = poly
    inside = False
    n = len(verts)
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[(i+1)%n]
        intersect = ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi+1e-30)+xi)
        if intersect:
            inside = not inside
    return inside

def build_nucleus_ellipse(row_idx, sarc_idx):
    top_line = row_top_lines[row_idx][sarc_idx]
    bottom_line = row_bottom_lines[row_idx][sarc_idx]
    bt1, bt2 = line_endpoints[bottom_line]
    tt1, tt2 = line_endpoints[top_line]
    b1, b2 = coords_dict[bt1], coords_dict[bt2]
    t1, t2 = coords_dict[tt1], coords_dict[tt2]

    center = 0.25 * (b1 + b2 + t1 + t2)
    mid_vec = b2 - b1
    theta_rot = np.arctan2(mid_vec[1], mid_vec[0])

    ellipse_pts = []
    theta = np.linspace(0, 2 * np.pi, num_circle_pts, endpoint=False)
    for th in theta:
        x = rx * np.cos(th)
        y = ry * np.sin(th)
        px = center[0] + x * np.cos(theta_rot) - y * np.sin(theta_rot)
        py = center[1] + x * np.sin(theta_rot) + y * np.cos(theta_rot)
        pid = gmsh.model.occ.add_point(px, py, 0.0, nucleus_mesh_size)
        ellipse_pts.append(pid)

    ellipse_lines = [
        gmsh.model.occ.add_line(ellipse_pts[i], ellipse_pts[(i + 1) % num_circle_pts])
        for i in range(num_circle_pts)
    ]
    loop = gmsh.model.occ.add_curve_loop(ellipse_lines)
    surf_id = gmsh.model.occ.add_plane_surface([loop])

    return {
        "surface": surf_id,
        "lines": ellipse_lines,
        "center": center,
        "theta_rot": theta_rot,
    }

# -------------------------
# Output folder
# -------------------------
output_folder = "cell_geometries"
os.makedirs(output_folder, exist_ok=True)
fig, axes = plt.subplots(1,2)

# -------------------------
# Build geometry
# -------------------------
coords_dict = {}
line_count = {}
line_endpoints = {}
original_polygons = {}
sarcomere_surfaces = []
zline_surfaces = []
cytoskeleton_surfaces = []
connection_surfaces = []

row_top_lines = []
row_bottom_lines = []

freq = 2.0*np.pi/num_sarcomeres
base_bottom_lengths = [sarco_length_mean*(1+amp*np.sin(freq*i)) for i in range(num_sarcomeres)]
base_top_lengths = [sarco_length_mean*(1+amp*np.sin(freq*i+phase_shift)) for i in range(num_sarcomeres)]

bottom_lengths_ref, top_lengths_ref = [], []
for Lb, Lt in zip(base_bottom_lengths, base_top_lengths):
    delta_shared = 3*length_variation*(np.random.randn()-0.5)
    delta_bottom = 1/3*length_variation_bottom*(np.random.randn()-0.5)
    delta_top = 1/3*length_variation_top*(np.random.randn()-0.5)
    bottom_lengths_ref.append(Lb + delta_shared + delta_bottom)
    top_lengths_ref.append(Lt + delta_shared + delta_top)

all_angles = []

for r in range(num_rows):
    top_lines_row, bottom_lines_row = [], []
    bl_coords = np.array([0.0, r*row_offset])
    tl_coords = np.array([0.0, r*row_offset+H])
    bl_id = gmsh.model.occ.add_point(*bl_coords, 0, mesh_size)
    tl_id = gmsh.model.occ.add_point(*tl_coords, 0, mesh_size)
    coords_dict[bl_id], coords_dict[tl_id] = bl_coords, tl_coords
    left_line = safe_add_line(bl_id, tl_id)

    prev_bl, prev_tl, prev_left_line = bl_id, tl_id, left_line

    for i in range(num_sarcomeres):
        bl_coord, tl_coord = coords_dict[prev_bl], coords_dict[prev_tl]
        edge_vec = tl_coord - bl_coord
        perp_angle = np.arctan2(edge_vec[1], edge_vec[0]) + np.pi/2
        bottom_len, top_len = bottom_lengths_ref[i], top_lengths_ref[i]

        dx_b, dy_b = bottom_len*np.cos(perp_angle), bottom_len*np.sin(perp_angle)
        dx_t, dy_t = top_len*np.cos(perp_angle), top_len*np.sin(perp_angle)
        br_coords = bl_coord + np.array([dx_b, dy_b])
        tr_coords = tl_coord + np.array([dx_t, dy_t])
        if i == num_sarcomeres-1:
            tr_coords[0] = br_coords[0]

        br_id = gmsh.model.occ.add_point(*br_coords, 0, mesh_size)
        tr_id = gmsh.model.occ.add_point(*tr_coords, 0, mesh_size)
        coords_dict[br_id], coords_dict[tr_id] = br_coords, tr_coords

        surf, top_line, right_line, bottom_line = add_rectangle_from_ids(prev_left_line, prev_bl, br_id, tr_id, prev_tl)
        sarcomere_surfaces.append(surf)
        top_lines_row.append(top_line)
        bottom_lines_row.append(bottom_line)

        bottom_vec = br_coords - bl_coord
        angle = np.arctan2(bottom_vec[1], bottom_vec[0])
        all_angles.append(angle)

        if i < num_sarcomeres-1:
            right_vec = tr_coords - br_coords
            perp_vec = np.array([-right_vec[1], right_vec[0]])/np.linalg.norm(right_vec)
            offset = perp_vec*zline_width
            zbr_coords, ztr_coords = br_coords + offset, tr_coords + offset
            zbr_id = gmsh.model.occ.add_point(*zbr_coords, 0, mesh_size)
            ztr_id = gmsh.model.occ.add_point(*ztr_coords, 0, mesh_size)
            coords_dict[zbr_id], coords_dict[ztr_id] = zbr_coords, ztr_coords
            zsurf, z_top_line, z_right_line, z_bottom_line = add_rectangle_from_ids(
                right_line, br_id, zbr_id, ztr_id, tr_id
            )
            zline_surfaces.append(zsurf)
            top_lines_row.append(z_top_line)
            bottom_lines_row.append(z_bottom_line)
            prev_bl, prev_tl, prev_left_line = zbr_id, ztr_id, z_right_line
        else:
            prev_bl, prev_tl, prev_left_line = br_id, tr_id, right_line

    row_top_lines.append(top_lines_row)
    row_bottom_lines.append(bottom_lines_row)

# Cytoskeleton & connections
for r in range(num_rows-1):
    top_edges = row_top_lines[r]
    bottom_edges = row_bottom_lines[r+1]
    for k in range(len(top_edges)):
        t_start, t_end = line_endpoints[top_edges[k]]
        b_start, b_end = line_endpoints[bottom_edges[k]]
        bl_id, br_id, tr_id, tl_id = b_start, b_end, t_end, t_start
        left_vert = find_line_between(bl_id, tl_id)
        if left_vert is None:
            left_vert = safe_add_line(bl_id, tl_id)
        csurf, _, _, _ = add_rectangle_from_ids(left_vert, bl_id, br_id, tr_id, tl_id)
        if k%2==0:
            cytoskeleton_surfaces.append(csurf)
        else:
            connection_surfaces.append(csurf)

# -------------------------
# Compute nucleus size
# -------------------------

if num_nuclei == 1:
    
    all_points = np.array(list(coords_dict.values()))
    min_xy, max_xy = np.min(all_points, axis=0), np.max(all_points, axis=0)
    bbox_size = max_xy - min_xy
    nucleus_radius = 3.0 #*min(bbox_size[0], bbox_size[1])

    rx = 3.0 * nucleus_radius + np.random.uniform(0, 0.2)
    ry = 1.0 * nucleus_radius + np.random.uniform(0, 0.2)

    r_mid = num_rows // 2
    i1 = int(0.5 * 2*num_sarcomeres)

    nucleus_defs = [
        build_nucleus_ellipse(r_mid, i1),
    ]

    nucleus_surfaces = [nd["surface"] for nd in nucleus_defs]
    all_ellipse_lines = [l for nd in nucleus_defs for l in nd["lines"]]

elif num_nuclei == 2:

    all_points = np.array(list(coords_dict.values()))
    min_xy, max_xy = np.min(all_points, axis=0), np.max(all_points, axis=0)
    bbox_size = max_xy - min_xy
    nucleus_radius = 3.0 #*min(bbox_size[0], bbox_size[1])

    rx = 3.0 * nucleus_radius + np.random.uniform(0, 0.2)
    ry = 1.0 * nucleus_radius + np.random.uniform(0, 0.2)

    # -------------------------
    # Build two rotated ellipses 
    # -------------------------
    r_mid = num_rows // 2
    i1 = int(0.7 * 2*num_sarcomeres)
    i2 = int(0.3 * 2*num_sarcomeres)


    nucleus_defs = [
        build_nucleus_ellipse(r_mid, i1),
        build_nucleus_ellipse(r_mid, i2),
    ]

    nucleus_surfaces = [nd["surface"] for nd in nucleus_defs]
    all_ellipse_lines = [l for nd in nucleus_defs for l in nd["lines"]]


else:
    raise NotImplementedError

# -------------------------
# Mesh size field
# -------------------------
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "EdgesList", all_ellipse_lines)
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", nucleus_mesh_size)
gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_size)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(2, "DistMax", sizefield_dist_max)
gmsh.model.mesh.field.setAsBackgroundMesh(2)

# -------------------------
# Fragment & synchronize
# -------------------------
all_original_surfaces = (
    sarcomere_surfaces + zline_surfaces + cytoskeleton_surfaces + connection_surfaces
)
gmsh.model.occ.synchronize()

orig_centroids = {}
for s in all_original_surfaces:
    c = surface_centroid_2d(s)
    if c is not None:
        orig_centroids[s] = c
    else:
        orig_centroids[s] = np.mean(
            original_polygons.get(s, np.array([[1e9, 1e9]])), axis=0
        )

sarco_tag_map = {s: i + 1 for i, s in enumerate(sarcomere_surfaces)}
entities = [(2, s) for s in all_original_surfaces + nucleus_surfaces]
gmsh.model.occ.fragment(entities, [])
gmsh.model.occ.synchronize()
all_surfs_after = [t for d, t in gmsh.model.getEntities(2)]


frag_centroids = {s: surface_centroid_2d(s) for s in all_surfs_after}
nucleus_fragments = []
sarco_fragments_map = {tag: [] for tag in sarco_tag_map.values()}
zline_fragments, cyto_fragments, conn_fragments = [], [], []

def which_nucleus(com):
    for idx, nd in enumerate(nucleus_defs):
        center = nd["center"]
        theta_rot = nd["theta_rot"]

        xp = (com[0] - center[0]) * np.cos(theta_rot) + (com[1] - center[1]) * np.sin(theta_rot)
        yp = -(com[0] - center[0]) * np.sin(theta_rot) + (com[1] - center[1]) * np.cos(theta_rot)

        if (xp / rx)**2 + (yp / ry)**2 < 1.0 - 1e-9:
            return idx
    return None

nucleus_fragments_0 = []
nucleus_fragments_1 = []

for frag in all_surfs_after:
    com = frag_centroids[frag]
    if com is None:
        nearest_orig = min(
            orig_centroids.keys(),
            key=lambda s: np.linalg.norm(orig_centroids[s] - np.array([0, 0])),
        )
        if nearest_orig in sarco_tag_map:
            sarco_fragments_map[sarco_tag_map[nearest_orig]].append(frag)
        else:
            sarco_fragments_map[next(iter(sarco_frag_map))].append(frag)
        continue

    nid = which_nucleus(com)
    if nid == 0:
        nucleus_fragments_0.append(frag)
        continue
    elif nid == 1:
        nucleus_fragments_1.append(frag)
        continue

    assigned = False
    for orig_surf, poly in original_polygons.items():
        if point_in_poly(com, poly):
            if orig_surf in sarcomere_surfaces:
                sarco_fragments_map[sarco_tag_map[orig_surf]].append(frag)
            elif orig_surf in zline_surfaces:
                zline_fragments.append(frag)
            elif orig_surf in cytoskeleton_surfaces:
                cyto_fragments.append(frag)
            elif orig_surf in connection_surfaces:
                conn_fragments.append(frag)
            assigned = True
            break
    if assigned:
        continue

    nearest_orig = min(
        orig_centroids.keys(),
        key=lambda s: np.linalg.norm(orig_centroids[s] - com),
    )
    if nearest_orig in sarco_tag_map:
        sarco_fragments_map[sarco_tag_map[nearest_orig]].append(frag)
    elif nearest_orig in zline_surfaces:
        zline_fragments.append(frag)
    elif nearest_orig in cytoskeleton_surfaces:
        cyto_fragments.append(frag)
    elif nearest_orig in connection_surfaces:
        conn_fragments.append(frag)
    else:
        sarco_fragments_map[next(iter(sarco_frag_map))].append(frag)

# -------------------------
# Physical groups
# -------------------------
for orig_surf, tag in sarco_tag_map.items():
    frags = sarco_fragments_map.get(tag, [])
    if frags:
        try:
            gmsh.model.addPhysicalGroup(2, frags, tag=tag)
        except:
            pass

#gmsh.fltk.run()

if zline_fragments:
    gmsh.model.addPhysicalGroup(2, zline_fragments, tag=ZLINE_TAG)
if cyto_fragments:
    gmsh.model.addPhysicalGroup(2, cyto_fragments, tag=CYTO_TAG)
if conn_fragments:
    gmsh.model.addPhysicalGroup(2, conn_fragments, tag=CONN_TAG)
if nucleus_fragments_0:
    gmsh.model.addPhysicalGroup(2, nucleus_fragments_0, tag=NUCLEUS1_TAG)
if nucleus_fragments_1:
    gmsh.model.addPhysicalGroup(2, nucleus_fragments_1, tag=NUCLEUS2_TAG)

#gmsh.fltk.run()

# -------------------------
# Mesh & export
# -------------------------
nodes, topologies = msh_gmsh_model(gmsh.model, 2)
mesh, entity_fs = mesh_from_gmsh(nodes, topologies)

fname_core = f"{output_folder}/cell_{cell_ID}_with_{num_nuclei}_nuclei"
fname_h5 = fname_core+".h5"
with df.HDF5File(mesh.mpi_comm(), fname_h5, "w") as out:
    out.write(mesh, "mesh")
    out.write(entity_fs[2], "subdomains")

fname_pvd = fname_core+".pvd"
df.File(fname_pvd) << entity_fs[2]

all_angles = (np.array(all_angles)+np.pi/2)%np.pi - np.pi/2
fname_npy = fname_core+"_angles.npy"
np.save(fname_npy, all_angles)



theta_values = [nd["theta_rot"] for nd in nucleus_defs]
angles_nuclei = np.array(theta_values)
radi_nuclei = np.array([rx, ry])

nuclei_data = {"angles" : angles_nuclei, "radi" : radi_nuclei}
print("nucleus data", nuclei_data)
fname_nuclei = fname_core+"_nuclei_angles.npy"
np.save(fname_nuclei, nuclei_data)
