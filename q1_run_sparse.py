#!/usr/bin/env python3
# ============================================
# Q1-1 : COLMAP Sparse Reconstruction（桌面 GPU 加速版, 固定 FPS=2）
# - 抽幀：ffmpeg（fps=2）
# - COLMAP：feature_extractor → sequential_matcher → mapper
# - 匯出：TXT（cameras/images/points3D）→ 轉 points3D.ply
# - 顯示：Open3D 點雲
# ============================================

import subprocess
import sys
import shutil
from pathlib import Path
import numpy as np
import open3d as o3d

# ========= 可調參數 =========
VIDEO = Path("q1/video/scene.MOV")     # 可改成 .mp4
IMG_DIR = Path("q1/images")
WORK_DIR = Path("q1/colmap_work")
SPARSE_DIR = WORK_DIR / "sparse"
OUT_DIR = Path("q1/output")
TXT_OUT_DIR = OUT_DIR / "txt"

FPS = 2               # 固定為 2 fps
MAX_IMG_SIZE = 2000       # SIFT 最大影像邊長
OVERWRITE_FRAMES = False  # True：重新抽幀（會清空 q1/images/*.jpg）
# ===========================


def die(msg: str):
    print(f"\n❌ {msg}")
    sys.exit(1)


def run_cmd(cmd, desc):
    print(f"\n[執行] {desc}\n{' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        die(f"指令失敗：{' '.join(cmd)}")
    print("✅ 完成")


def ensure_bins():
    for b in ["ffmpeg", "colmap"]:
        if shutil.which(b) is None:
            die(f"找不到 `{b}`，請先安裝並加入 PATH。")


def extract_frames(video: Path, img_dir: Path, fps: float):
    if OVERWRITE_FRAMES and img_dir.exists():
        print(f"🧹 清理舊影像：{img_dir}")
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    if list(img_dir.glob("*.jpg")):
        print(f"📸 {img_dir} 已有影像，略過抽幀（如需重抽設 OVERWRITE_FRAMES=True）")
        return

    if not video.exists():
        die(f"找不到影片：{video}")

    run_cmd([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", f"fps={fps:.3f}",
        str(img_dir / "%04d.jpg")
    ], f"抽幀（fps={fps:.2f}）→ {img_dir}")

    n = len(list(img_dir.glob("*.jpg")))
    print(f"📷 抽幀完成，共 {n} 張")


def export_txt(model_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "colmap", "model_converter",
        "--input_path", str(model_dir),
        "--output_path", str(out_dir),
        "--output_type", "TXT"
    ], "匯出 TXT（cameras/images/points3D）")


def points3d_txt_to_ply(txt_path: Path, out_ply: Path):
    """把 COLMAP points3D.txt 轉成 points3D.ply（純點雲）"""
    if not txt_path.exists():
        die(f"找不到 {txt_path}，無法轉換 PLY")

    pts, cols = [], []
    with open(txt_path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            toks = line.strip().split()
            if len(toks) < 8:
                continue
            x, y, z = map(float, toks[1:4])
            r, g, b = map(int, toks[4:7])
            pts.append([x, y, z])
            cols.append([r/255.0, g/255.0, b/255.0])

    if not pts:
        die("points3D.txt 內沒有任何點")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(cols))
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(out_ply), pcd):
        die("寫入 points3D.ply 失敗")
    print(f"🟢 已輸出點雲：{out_ply}")


def main():
    ensure_bins()

    # 建立資料夾
    for d in [VIDEO.parent, IMG_DIR, WORK_DIR, SPARSE_DIR, OUT_DIR, TXT_OUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" Q1-1 : Sparse Reconstruction（桌面 GPU 版）")
    print("=" * 60)
    print(f"🎥 影片：{VIDEO}")
    print(f"📂 影像輸出：{IMG_DIR}")
    print(f"📦 結果輸出：{OUT_DIR}")
    print("-" * 60)

    # Step 1) 抽幀（固定 2 fps）
    extract_frames(VIDEO, IMG_DIR, FPS)

    # Step 2) 特徵擷取（啟用 GPU）
    run_cmd([
        "colmap", "feature_extractor",
        "--database_path", str(WORK_DIR / "database.db"),
        "--image_path", str(IMG_DIR),
        "--SiftExtraction.max_image_size", str(MAX_IMG_SIZE),
        "--ImageReader.single_camera", "1",
    ], "COLMAP 特徵擷取（GPU 啟用）")

    # Step 3) 特徵匹配（sequential）
    run_cmd([
        "colmap", "sequential_matcher",
        "--database_path", str(WORK_DIR / "database.db"),
    ], "COLMAP 特徵匹配（sequential, GPU 啟用）")

    # Step 4) 建立稀疏模型
    run_cmd([
        "colmap", "mapper",
        "--database_path", str(WORK_DIR / "database.db"),
        "--image_path", str(IMG_DIR),
        "--output_path", str(SPARE_DIR := SPARSE_DIR),
    ], "COLMAP 建立稀疏模型")

    MODEL_DIR = SPARSE_DIR / "0"
    if not MODEL_DIR.exists():
        die(f"Mapper 未輸出模型：{MODEL_DIR}")

    # Step 5) 匯出 TXT 並轉成 points3D.ply
    export_txt(MODEL_DIR, TXT_OUT_DIR)
    points_txt = TXT_OUT_DIR / "points3D.txt"
    points_ply = OUT_DIR / "points3D.ply"
    points3d_txt_to_ply(points_txt, points_ply)

    # Step 6) 顯示結果
    print("\n📈 顯示點雲（Open3D）...")
    pcd = o3d.io.read_point_cloud(str(points_ply))
    if pcd.has_points():
        o3d.visualization.draw_geometries([pcd], window_name="Q1-1 Sparse Cloud")
    else:
        print("⚠️ 點雲為空，請檢查重建是否成功")

    print("\n🎉 Q1-1 完成！")
    print(f"🔹 點雲：{points_ply}")
    print(f"🔹 TXT：{TXT_OUT_DIR}（cameras.txt / images.txt / points3D.txt）")
    print("=" * 60)


if __name__ == "__main__":
    main()


