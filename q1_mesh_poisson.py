#!/usr/bin/env python3
import open3d as o3d
import numpy as np
from pathlib import Path

IN_PLY = Path("q1/output/points3D.ply")
OUT_MESH = Path("q1/output/mesh_poisson.ply")
SNAPSHOT = Path("q1/output/mesh.png")

USE_VOXEL_DOWNSAMPLE = False   
VOXEL_SIZE = 0.01 
POISSON_DEPTH = 10 
DENSITY_TRIM_Q = 0.08          # 砍最外圍 10% 低密度頂點
SMOOTH_ITERS = 3 
TARGET_FACES = 150_000 


def largest_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    labels = np.array(mesh.cluster_connected_triangles()[0])
    if labels.size == 0:
        return mesh
    keep = np.argmax(np.bincount(labels))
    mask = labels != keep
    mesh.remove_triangles_by_mask(mask)
    mesh.remove_unreferenced_vertices()
    return mesh

def save_snapshot(mesh, png_path: Path):
    try:
        from open3d.visualization import rendering
        renderer = rendering.OffscreenRenderer(1280, 960)
        mat = rendering.MaterialRecord(); mat.shader = "defaultLit"
        renderer.scene.add_geometry("mesh", mesh, mat)
        img = renderer.render_to_image()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_image(str(png_path), img)
        print(f"🖼️ 已存圖：{png_path}")
    except Exception:
        print("⚠️ 無法離線渲染，改開視窗手動截圖")

def auto_normal_radius(pcd: o3d.geometry.PointCloud) -> float:
    # 以最近鄰距離的分位數估尺度，比 AABB 更穩
    dists = pcd.compute_nearest_neighbor_distance()
    if len(dists) == 0:
        return 0.05
    base = np.median(dists)  # 中位數距離
    return max(base * 6.0, 1e-4)  # 半徑取 6×鄰距（經驗值）

def main():
    if not IN_PLY.exists():
        raise FileNotFoundError(f"找不到點雲：{IN_PLY}")

    print("="*60)
    print(" Q1-2 : Poisson Mesh Reconstruction (GUI保守版)")
    print("="*60)
    print(f"📥 輸入：{IN_PLY}")

    # 讀點雲
    pcd = o3d.io.read_point_cloud(str(IN_PLY))
    n0 = len(pcd.points)
    if n0 == 0:
        raise RuntimeError("點雲為空，請先確認 Q1-1 的 points3D.ply。")
    print(f"📊 原始點數：{n0}")

    # 可視化「點雲本體」先檢查（按需：註解掉就不顯示）
    # o3d.visualization.draw_geometries([pcd], window_name="原始點雲（先檢查）")

    # 下採樣（預設關閉，避免丟資訊）
    if USE_VOXEL_DOWNSAMPLE:
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)
        print(f"🔻 下採樣：{len(pcd.points)} 點")

    # 法向：先用自動半徑（Hybrid），失敗則回退 KNN
    normal_radius = auto_normal_radius(pcd)
    print(f"🧭 法向估計半徑：{normal_radius:.5f}")
    try:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=60))
    except Exception as e:
        print(f"⚠️ Hybrid 失敗，改 KNN：{e}")
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
    # 一致取向非必須，嘗試就好
    try:
        pcd.orient_normals_consistent_tangent_plane(30)
    except Exception as e:
        print(f"ℹ️ 法向一致取向略過：{e}")

    # Poisson
    print(f"🧱 Poisson (depth={POISSON_DEPTH}) ...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH
    )
    densities = np.asarray(densities)

    # 密度裁切（保守 10%）
    thr = np.quantile(densities, DENSITY_TRIM_Q)
    mesh.remove_vertices_by_mask(densities < thr)
    print(f"✂️ 砍低密度 {DENSITY_TRIM_Q*100:.0f}% 頂點 (thr={thr:.4f})")

    # 清理 + 保留最大連通塊
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh = largest_component(mesh)

    # 輕微平滑 + 簡化
    if SMOOTH_ITERS > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=SMOOTH_ITERS)
        print(f"🫧 平滑 {SMOOTH_ITERS} 次")
    if TARGET_FACES and len(mesh.triangles) > TARGET_FACES:
        mesh = mesh.simplify_quadric_decimation(TARGET_FACES)
        print(f"🔻 簡化至 ≈{TARGET_FACES} 面")

    mesh.compute_vertex_normals()

    # 輸出
    OUT_MESH.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(OUT_MESH), mesh)
    print(f"✅ 已輸出網格：{OUT_MESH}")

    print("🪟 開啟 Open3D GUI 視窗，請旋轉/縮放檢查成品")
    o3d.visualization.draw_geometries([mesh], window_name="Q1-2 Poisson Mesh Viewer")

    print("🎉 完成")
    print("="*60)


if __name__ == "__main__":
    main()

