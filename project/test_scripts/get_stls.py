import open3d as o3d

# Загрузка объединённого облака
merged_pcd = o3d.io.read_point_cloud("combined_cloud.ply")

# Повторная выборка и нормали
pcd_down = merged_pcd.voxel_down_sample(voxel_size=0.005)
pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

### === Меш 1: Poisson ===
mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_down, depth=9)
bbox = merged_pcd.get_axis_aligned_bounding_box()
mesh_poisson_crop = mesh_poisson.crop(bbox)
mesh_poisson_crop.compute_vertex_normals()

o3d.io.write_triangle_mesh("combined_mesh_poisson.stl", mesh_poisson_crop)
o3d.io.write_triangle_mesh("combined_mesh_poisson.ply", mesh_poisson_crop)
print("[✓] Poisson mesh сохранён")

### === Меш 2: Alpha Shapes ===
alpha = 0.03  # Можно подбирать вручную
mesh_alpha = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_down, alpha)
mesh_alpha.compute_vertex_normals()

o3d.io.write_triangle_mesh("combined_mesh_alpha.stl", mesh_alpha)
o3d.io.write_triangle_mesh("combined_mesh_alpha.ply", mesh_alpha)
print("[✓] Alpha Shape mesh сохранён")

### === Меш 3: Ball Pivoting ===
radii = [0.005, 0.01, 0.02]
mesh_bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd_down,
    o3d.utility.DoubleVector(radii)
)
mesh_bpa.compute_vertex_normals()

o3d.io.write_triangle_mesh("combined_mesh_bpa.stl", mesh_bpa)
o3d.io.write_triangle_mesh("combined_mesh_bpa.ply", mesh_bpa)
print("[✓] Ball Pivoting mesh сохранён")
