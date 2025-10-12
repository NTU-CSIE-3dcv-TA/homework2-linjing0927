#!/usr/bin/env python3
# Q2-2: AR cube (points version, colorful faces) + Painter's Algorithm
# - Uses poses from data/poses_est.csv
# - Uses cube_transform_mat.npy from transform_cube.py (if present)
# - Draws cube as colored points per face
# - Writes video with each frame exactly once (no padding)

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- Config ----------
OUT_DIR = "q2_ar_out"
VIDEO_OUT = "q2_ar_result.mp4"
FPS = 10  # 影片播放幀率；總時長 = len(frames) / FPS

# points density & style
POINTS_PER_EDGE = 18   # 每面邊上的點數（越大越密）
POINT_RADIUS = 3       # 影像上圓點半徑（像素）
COLOR_MODE = "per_face"  # "per_face" | "rainbow"

# camera intrinsics & distortion (k1,k2,p1,p2,k3=0)
K = np.array([[1868.27, 0, 540],
              [0, 1869.18, 960],
              [0, 0, 1]], dtype=np.float64)
DIST = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352, 0.0], dtype=np.float64)


# ---------- Helpers ----------
def write_video_once(frames, video_out_path, fps=10):
    """將 frames 依序各寫入一次；總時長 = len(frames) / fps。"""
    if len(frames) == 0:
        print("⚠️ 沒有可寫入的影格")
        return
    first = cv2.imread(frames[0])
    if first is None:
        print(f"⚠️ 無法讀取影格：{frames[0]}")
        return
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_out_path), fourcc, fps, (w, h))
    try:
        for f in frames:
            img = cv2.imread(f)
            if img is None:
                continue
            vw.write(img)
    finally:
        vw.release()
    duration = len(frames) / float(fps)
    print(f"🎬 影片完成：{video_out_path} (fps={fps}, frames={len(frames)}, 時長≈{duration:.2f}s)")


def load_cube_transform(default_center, default_size=0.6):
    """
    優先讀 transform_cube.py 的 3x4 變換矩陣；若不存在，用 default center + 單位縮放。
    transform 形式: [sR | t] (3x4)
    """
    path = "cube_transform_mat.npy"
    if os.path.exists(path):
        T = np.load(path)
        if T.shape == (3, 4):
            return T
        else:
            print("⚠️ cube_transform_mat.npy 形狀不是 3x4，改用預設。")
    # fallback: identity rotation + scale=1, translate to center
    R = np.eye(3)
    t = default_center.reshape(3)
    s = 1.0
    return np.hstack([s * R, t.reshape(3, 1)])


def make_cube_points_unit(n):
    """
    在 unit cube [0,1]^3 的 6 個面建立 n×n 個等距點。
    回傳:
      X_local: (M,3), face_id: (M,), uv_face: (M,2)
    面順序: z-, z+, y-, y+, x+, x-
    """
    lin = np.linspace(0.0, 1.0, n)
    uu, vv = np.meshgrid(lin, lin)  # (n,n)

    Xs, fids, UVs = [], [], []

    # z- (0)
    X = np.stack([uu, vv, np.zeros_like(uu)], axis=-1)
    Xs.append(X.reshape(-1, 3)); fids.append(np.full(n*n, 0)); UVs.append(np.stack([uu, vv], -1).reshape(-1, 2))
    # z+ (1)
    X = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
    Xs.append(X.reshape(-1, 3)); fids.append(np.full(n*n, 1)); UVs.append(np.stack([uu, vv], -1).reshape(-1, 2))
    # y- (2)
    X = np.stack([uu, np.zeros_like(uu), vv], axis=-1)
    Xs.append(X.reshape(-1, 3)); fids.append(np.full(n*n, 2)); UVs.append(np.stack([uu, vv], -1).reshape(-1, 2))
    # y+ (3)
    X = np.stack([uu, np.ones_like(uu), vv], axis=-1)
    Xs.append(X.reshape(-1, 3)); fids.append(np.full(n*n, 3)); UVs.append(np.stack([uu, vv], -1).reshape(-1, 2))
    # x+ (4)
    X = np.stack([np.ones_like(uu), uu, vv], axis=-1)
    Xs.append(X.reshape(-1, 3)); fids.append(np.full(n*n, 4)); UVs.append(np.stack([uu, vv], -1).reshape(-1, 2))
    # x- (5)
    X = np.stack([np.zeros_like(uu), uu, vv], axis=-1)
    Xs.append(X.reshape(-1, 3)); fids.append(np.full(n*n, 5)); UVs.append(np.stack([uu, vv], -1).reshape(-1, 2))

    X_local = np.vstack(Xs)
    face_id = np.concatenate(fids)
    uv_face = np.vstack(UVs)
    return X_local, face_id, uv_face


def apply_transform_points(X_local, T34):
    """(N,3) local → world with 3x4 affine"""
    Xh = np.hstack([X_local, np.ones((X_local.shape[0], 1))])  # (N,4)
    Xw = (T34 @ Xh.T).T
    return Xw


def face_colormap(face_id, uv, mode="per_face"):
    """
    (M,3) BGR uint8 colors.
    - per_face: 固定 6 面顏色
    - rainbow : HSV 漸層 (face-based hue + u-grad)
    """
    if mode == "per_face":
        table = np.array([
            [ 40,  40, 255],  # z-  (red-ish)
            [ 40, 255,  40],  # z+  (green-ish)
            [255,  40,  40],  # y-  (blue-ish)
            [255, 180,  40],  # y+  (teal-ish)
            [180,  40, 255],  # x+  (magenta-ish)
            [ 40, 255, 255],  # x-  (yellow-ish)
        ], dtype=np.uint8)
        return table[face_id]

    # rainbow (NumPy 2.0-safe: use np.ptp)
    hsv = np.zeros((face_id.shape[0], 3), dtype=np.float32)
    hue_base = np.array([0, 30, 60, 120, 180, 240], dtype=np.float32) / 360.0
    hsv[:, 0] = hue_base[face_id]
    # 以 u 方向做些微 hue 漸層
    u = uv[:, 0]
    u_norm = (u - u.min()) / (np.ptp(u) + 1e-9)
    hsv[:, 0] = (hsv[:, 0] + 0.15 * u_norm) % 1.0
    hsv[:, 1] = 1.0
    hsv[:, 2] = 1.0
    bgr = cv2.cvtColor((hsv.reshape(-1, 1, 3) * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)[:, 0, :]
    return bgr


def project_world_points(Xw, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(Xw.reshape(-1, 1, 3), rvec, tvec, K, dist)
    return proj.reshape(-1, 2)


def draw_points_painter(image, proj, depths, colors, radius=3):
    """Painter’s Algorithm for points: sort by depth (far→near), draw circles."""
    order = np.argsort(depths)[::-1]  # far → near
    H, W = image.shape[:2]
    out = image.copy()
    for i in order:
        u, v = proj[i]
        if 0 <= u < W and 0 <= v < H:
            c = tuple(int(x) for x in colors[i])
            cv2.circle(out, (int(u), int(v)), radius, c, -1, lineType=cv2.LINE_AA)
    return out


# ---------- Main ----------
def main():
    print("============================================================")
    print(" Q2-2 : AR Cube (Colorful Points) + Painter's Algorithm")
    print("============================================================")

    os.makedirs(OUT_DIR, exist_ok=True)

    # 讀姿態 & 影像資訊
    poses_csv = "data/poses_est.csv"
    images_pkl = "data/images.pkl"
    if not os.path.exists(poses_csv):
        raise FileNotFoundError("找不到 data/poses_est.csv，請先完成 Q2-1 (2d3dmatching.py)。")
    poses_df = pd.read_csv(poses_csv)
    images_df = pd.read_pickle(images_pkl)

    # 估一個場景中心（若 transform 不存在）
    points3d_pkl = "data/points3D.pkl"
    if os.path.exists(points3d_pkl):
        p3d = pd.read_pickle(points3d_pkl)
        scene_center = np.vstack(p3d["XYZ"]).mean(axis=0)
    else:
        scene_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # 讀/建 cube 變換矩陣 (3x4)
    T34 = load_cube_transform(scene_center, default_size=0.6)

    # 建 unit cube 面點 → 世界座標
    X_local, face_id, uv_face = make_cube_points_unit(POINTS_PER_EDGE)
    X_world = apply_transform_points(X_local, T34)
    colors = face_colormap(face_id, uv_face, mode=COLOR_MODE)

    frames = []
    for i, row in tqdm(poses_df.iterrows(), total=len(poses_df)):
        img_name = images_df.loc[images_df["IMAGE_ID"] == row["IMAGE_ID"], "NAME"].values
        if len(img_name) == 0:
            continue
        img_path = os.path.join("data/frames", img_name[0])
        img = cv2.imread(img_path)
        if img is None:
            print(f"skip: {img_path}")
            continue

        rvec = np.array([row["rvec_x"], row["rvec_y"], row["rvec_z"]], dtype=np.float64).reshape(3, 1)
        tvec = np.array([row["t_x"], row["t_y"], row["t_z"]], dtype=np.float64).reshape(3, 1)

        # 投影點
        proj = project_world_points(X_world, rvec, tvec, K, DIST)

        # 用相機座標 z 當深度
        Rm, _ = cv2.Rodrigues(rvec)
        Xc = (Rm @ X_world.T + tvec).T
        depths = Xc[:, 2]

        out = draw_points_painter(img, proj, depths, colors, radius=POINT_RADIUS)
        out_path = os.path.join(OUT_DIR, f"{i:06d}.jpg")
        cv2.imwrite(out_path, out)
        frames.append(out_path)

    # 影片（每張影格各寫一次）
    if len(frames) > 0:
        write_video_once(frames, VIDEO_OUT, fps=FPS)
    else:
        print("⚠️ 沒有任何輸出影格，請檢查 poses_est.csv 與 data/frames 內容。")

    print("✅ Q2-2 完成！")

if __name__ == "__main__":
    main()

