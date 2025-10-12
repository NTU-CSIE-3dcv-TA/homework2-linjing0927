from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
from tqdm import tqdm
import open3d as o3d

np.random.seed(1428) # do not change this seed
random.seed(1428)    # do not change this seed

# ------------------- 可調參數（符合題意，預設穩健） -------------------
LOWE_RATIO = 0.75         # Lowe ratio test
MIN_CORR   = 20           # 進入RANSAC的最少匹配數
RANSAC_MAX_ITERS = 4000
RANSAC_REPROJ_ERR = 4.0   # 以「去畸變像素」評分（像素閾值）
TARGET_INLIER_RATIO = 0.5 # 內點比例達標即提前終止
LOCAL_REFINE_ITERS = 10   # 以GN對內點做小幅收斂；0=不收斂
# --------------------------------------------------------------------

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

# -------------- 工具：FLANN + ratio test 建立 2D-3D 對應 --------------
def _build_flann():
    index_params = dict(algorithm=1, trees=8)  # KD-Tree
    search_params = dict(checks=64)
    return cv2.FlannBasedMatcher(index_params, search_params)

def _match_2d_3d(kp_query, desc_query, kp_model, desc_model, ratio=LOWE_RATIO):
    desc_query = desc_query.astype(np.float32)
    desc_model = desc_model.astype(np.float32)
    flann = _build_flann()
    matches = flann.knnMatch(desc_query, desc_model, k=2)

    pts2d, pts3d = [], []
    for m_n in matches:
        if len(m_n) < 2: continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            pts2d.append(kp_query[m.queryIdx])   # (x,y) 像素（畸變後）
            pts3d.append(kp_model[m.trainIdx])   # (X,Y,Z)
    if len(pts2d) == 0:
        return np.empty((0,2)), np.empty((0,3))
    return np.asarray(pts2d, np.float64), np.asarray(pts3d, np.float64)

# -------------- 工具：去畸變（反向迭代），bearing 計算 ---------------
def _undistort_points_px(pts_px, K, dist, iters=8):
    """把『畸變後像素』反解為『去畸變後像素』與其相機座標歸一化點。"""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    k1, k2, p1, p2 = dist[0], dist[1], dist[2], dist[3]
    k3 = 0.0

    # 轉到歸一化座標 (distorted normalized)
    x_d = (pts_px[:,0] - cx) / fx
    y_d = (pts_px[:,1] - cy) / fy

    # 初始估（去畸變）取等於畸變座標，迭代修正
    x_u = x_d.copy()
    y_u = y_d.copy()

    for _ in range(iters):
        r2 = x_u*x_u + y_u*y_u
        radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        x_tang = 2*p1*x_u*y_u + p2*(r2 + 2*x_u*x_u)
        y_tang = p1*(r2 + 2*y_u*y_u) + 2*p2*x_u*y_u
        x_est = x_u*radial + x_tang
        y_est = y_u*radial + y_tang
        # 以固定步長往使 x_est→x_d、y_est→y_d 的方向修正
        x_u += (x_d - x_est)
        y_u += (y_d - y_est)

    # 回像素平面（去畸變後像素）
    u = fx * x_u + cx
    v = fy * y_u + cy
    px_ud = np.stack([u, v], axis=1)
    # 歸一化相機座標（去畸變）
    xn = x_u; yn = y_u
    return px_ud, np.stack([xn, yn], axis=1)

def _bearings_from_px_ud(px_ud, K):
    """由『去畸變像素』得到單位視線向量 f = [x, y, 1]/||.||。"""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (px_ud[:,0] - cx) / fx
    y = (px_ud[:,1] - cy) / fy
    f = np.stack([x, y, np.ones_like(x)], axis=1)
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    return f

# P3P
def _solve_depths_by_distances(f1, f2, f3, P1, P2, P3, max_iters=50, tol=1e-9):
    """
    解 s1,s2,s3，使 || s1 f1 - s2 f2 || = ||P1 - P2|| 等距離方程成立。
    以 Levenberg-Marquardt / Gauss-Newton 在 R^3 上迭代。
    """
    d12 = np.linalg.norm(P1 - P2)
    d23 = np.linalg.norm(P2 - P3)
    d13 = np.linalg.norm(P1 - P3)

    s = np.full(3, (d12 + d23 + d13)/3.0, dtype=np.float64)  # 初值同量級

    def residuals(s):
        X1 = s[0]*f1; X2 = s[1]*f2; X3 = s[2]*f3
        return np.array([
            np.linalg.norm(X1 - X2) - d12,
            np.linalg.norm(X2 - X3) - d23,
            np.linalg.norm(X1 - X3) - d13
        ], dtype=np.float64)

    def jacobian(s):
        X1 = s[0]*f1; X2 = s[1]*f2; X3 = s[2]*f3
        d12v = X1 - X2; n12 = np.linalg.norm(d12v) + 1e-12
        d23v = X2 - X3; n23 = np.linalg.norm(d23v) + 1e-12
        d13v = X1 - X3; n13 = np.linalg.norm(d13v) + 1e-12

        J = np.zeros((3,3), dtype=np.float64)
        J[0,0] = (d12v/n12) @ f1; J[0,1] = -(d12v/n12) @ f2
        J[1,1] = (d23v/n23) @ f2; J[1,2] = -(d23v/n23) @ f3
        J[2,0] = (d13v/n13) @ f1; J[2,2] = -(d13v/n13) @ f3
        return J

    lam = 1e-3
    for _ in range(max_iters):
        r = residuals(s)
        if np.linalg.norm(r) < tol:
            break
        J = jacobian(s)
        H = J.T @ J + lam*np.eye(3)
        g = J.T @ r
        try:
            dx = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return False, None, None, None
        s_new = s + dx
        s = np.maximum(s_new, 1e-6)  # 深度不可為負
    if not np.all(s > 0):
        return False, None, None, None
    return True, s[0], s[1], s[2]

def _kabsch(Pw, Pc):
    """Umeyama/Kabsch：求 R,t 使 R Pw + t ≈ Pc。"""
    Pc_mu = Pc.mean(axis=0); Pw_mu = Pw.mean(axis=0)
    X = Pw - Pw_mu; Y = Pc - Pc_mu
    H = X.T @ Y
    U, S, VT = np.linalg.svd(H)
    R_est = U @ VT
    if np.linalg.det(R_est) < 0:
        U[:,-1] *= -1
        R_est = U @ VT
    t_est = Pc_mu - R_est @ Pw_mu
    return R_est, t_est

def _p3p_minimal(f123, P123):
    """
    純手刻 P3P：
      1) 以三個 bearing 與三個世界點，解 s1,s2,s3 使兩兩距離相符
      2) Xc = s_i f_i，與 Pw 對齊 → 得 (R,t)
    可能 0 或 1 組解（此實作通常穩定產生一組）
    """
    f1,f2,f3 = f123
    P1,P2,P3 = P123
    ok, s1,s2,s3 = _solve_depths_by_distances(f1,f2,f3,P1,P2,P3)
    if not ok:
        return []
    Xc = np.stack([s1*f1, s2*f2, s3*f3], axis=0)
    Pw = np.stack([P1, P2, P3], axis=0)
    R_est, t_est = _kabsch(Pw, Xc)
    return [(R_est, t_est)]

# --------------------- RANSAC（手刻）+ GN 微調 ---------------------
def _project_no_dist_px(Xw, R_est, t_est, K):
    """世界點投影到像素（不加畸變）。"""
    Xc = (R_est @ Xw.T + t_est.reshape(3,1)).T
    z = Xc[:,2:3]
    z = np.where(z==0, 1e-9, z)
    xnorm = Xc[:,:2] / z
    u = K[0,0]*xnorm[:,0] + K[0,2]
    v = K[1,1]*xnorm[:,1] + K[1,2]
    return np.stack([u,v], axis=1), Xc

def _gauss_newton_refine(R0, t0, Xw, obs_px_ud, K, iters=10):
    """以去畸變像素為觀測，最小化 reprojection error（不加畸變），對 (R,t) 做小旋量更新。"""
    Rm = R0.copy(); tm = t0.copy()
    fx, fy = K[0,0], K[1,1]

    for _ in range(iters):
        U, Xc = _project_no_dist_px(Xw, Rm, tm, K)
        err = (U - obs_px_ud).reshape(-1,2)

        J_list, r_list = [], []
        for i in range(Xw.shape[0]):
            x, y, z = Xc[i]
            if z <= 1e-9: continue
            du_dX = np.array([[fx/z, 0, -fx*x/(z*z)],
                              [0, fy/z, -fy*y/(z*z)]], dtype=np.float64)  # 2x3
            # dXc/dw = -[Xc]_x
            Xcx = np.array([[0, -z,  y],
                            [z,  0, -x],
                            [-y, x,  0]], dtype=np.float64)
            dXc_dw = -Xcx
            dXc_dt = np.eye(3)
            Ji = np.hstack([du_dX @ dXc_dw, du_dX @ dXc_dt])  # 2x6
            J_list.append(Ji)
            r_list.append(err[i])

        if not J_list:
            break
        J = np.vstack(J_list)
        rvec = np.hstack(r_list).reshape(-1,1)

        H = J.T @ J + 1e-6*np.eye(6)
        g = J.T @ rvec
        try:
            dx = -np.linalg.solve(H, g).reshape(-1)
        except np.linalg.LinAlgError:
            break
        dw, dt = dx[:3], dx[3:]
        dR = R.from_rotvec(dw).as_matrix()
        Rm = dR @ Rm
        tm = tm + dt
        if np.linalg.norm(dx) < 1e-6:
            break
    return Rm, tm

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    """
    加分題
    流程：
      1) Descriptor matching（FLANN + ratio test）
      2) 去畸變 → 由去畸變像素建立 bearing vectors
      3) RANSAC：每次隨機取3對應 → P3P minimal → 以無畸變投影在去畸變像素平面打分
      4) （可選）用內點做 Gauss-Newton 小幅收斂
    回傳：retval(bool), rvec(3,1), tvec(3,1), inliers(np.ndarray[bool])
    """
    kp_query, desc_query = query
    kp_model, desc_model = model

    # 題目相機參數
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]], dtype=np.float64)
    distCoeffs   = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352], dtype=np.float64)

    # 1) 先做 2D-3D 對應
    pts2d_px, pts3d = _match_2d_3d(kp_query, desc_query, kp_model, desc_model, LOWE_RATIO)
    if len(pts2d_px) < max(MIN_CORR, 3):
        return False, None, None, None

    # 2) 去畸變像素 & bearings
    pts2d_px_ud, pts2d_norm_ud = _undistort_points_px(pts2d_px, cameraMatrix, distCoeffs, iters=8)
    f = np.stack([pts2d_norm_ud[:,0], pts2d_norm_ud[:,1], np.ones_like(pts2d_norm_ud[:,0])], axis=1)
    f = f / np.linalg.norm(f, axis=1, keepdims=True)

    # 3) 手刻 RANSAC + P3P
    n = len(pts2d_px)
    best_inliers = None
    best_R, best_t = None, None
    best_cnt = -1
    best_mean_err = np.inf

    for _ in range(RANSAC_MAX_ITERS):
        idx = np.random.choice(n, 3, replace=False)
        f123 = f[idx]
        P123 = pts3d[idx]

        cands = _p3p_minimal(f123, P123)
        if not cands:
            continue
        for (Rc, tc) in cands:
            proj_px, _ = _project_no_dist_px(pts3d, Rc, tc, cameraMatrix)
            err = np.linalg.norm(proj_px - pts2d_px_ud, axis=1)
            inliers = (err < RANSAC_REPROJ_ERR)
            cnt = int(inliers.sum())
            mean_err = err[inliers].mean() if cnt > 0 else np.inf

            update = False
            if cnt > best_cnt: update = True
            elif cnt == best_cnt and mean_err < best_mean_err: update = True
            if update:
                best_inliers = inliers
                best_R, best_t = Rc, tc
                best_cnt = cnt
                best_mean_err = mean_err

        if best_cnt >= int(TARGET_INLIER_RATIO * n):
            break

    if best_inliers is None or best_cnt < 3:
        return False, None, None, None

    # 4) 以內點做純手刻 GN 小收斂（可關）
    if LOCAL_REFINE_ITERS > 0:
        obs = pts2d_px_ud[best_inliers]
        obj = pts3d[best_inliers]
        best_R, best_t = _gauss_newton_refine(best_R, best_t, obj, obs, cameraMatrix, iters=LOCAL_REFINE_ITERS)

    rvec = R.from_matrix(best_R).as_rotvec().reshape(3,1)
    tvec = best_t.reshape(3,1)
    return True, rvec, tvec, best_inliers

def rotation_error(R1, R2):
    """
    計算旋轉誤差（度）：以四元數相對旋轉角為準。
    期待輸入 shape 為 (N,4)，順序 [QX,QY,QZ,QW]（和資料一致）。
    """
    def _norm(q):
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        return q
    q1 = _norm(np.asarray(R1, np.float64))
    q2 = _norm(np.asarray(R2, np.float64))
    r_rel = (R.from_quat(q1)).inv() * R.from_quat(q2)
    ang = np.linalg.norm(r_rel.as_rotvec(), axis=-1)
    return np.degrees(ang)

def translation_error(t1, t2):
    """平移誤差（歐氏距離），輸入 shape (N,3)。"""
    t1 = np.asarray(t1, np.float64).reshape(-1,3)
    t2 = np.asarray(t2, np.float64).reshape(-1,3)
    return np.linalg.norm(t1 - t2, axis=-1)


def visualization(Camera2World_Transform_Matrixs, points3D_df):
    import open3d as o3d
    import numpy as np

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Camera Pose & Trajectory", width=1280, height=720)

    # === 1️⃣ 顯示 3D 點雲 ===
    xyz = np.vstack(points3D_df["XYZ"])
    rgb = np.vstack(points3D_df["RGB"]) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    vis.add_geometry(pcd)

    # === 2️⃣ 建立相機錐體 (四角錐) ===
    def make_camera_pyramid(scale=0.15, color=[1, 0, 0]):
        apex = np.array([[0, 0, 0]])
        base = np.array([
            [ scale,  scale,  1.5*scale],
            [-scale,  scale,  1.5*scale],
            [-scale, -scale,  1.5*scale],
            [ scale, -scale,  1.5*scale],
        ])
        vertices = np.vstack([apex, base])
        lines = [
            [0,1],[0,2],[0,3],[0,4],
            [1,2],[2,3],[3,4],[4,1]
        ]
        colors = [color for _ in lines]
        cam = o3d.geometry.LineSet()
        cam.points = o3d.utility.Vector3dVector(vertices)
        cam.lines = o3d.utility.Vector2iVector(lines)
        cam.colors = o3d.utility.Vector3dVector(colors)
        return cam

    cam_centers = []
    for i, T in enumerate(Camera2World_Transform_Matrixs):
        R = T[:3,:3]
        t = T[:3,3]
        cam_centers.append(t)

        # 紅藍交替
        color = [1, 0, 0] if i % 2 == 0 else [0, 0, 1]
        cam = make_camera_pyramid(color=color)

        # 將錐體轉到世界座標系
        pts = np.asarray(cam.points)
        pts_world = (R @ pts.T).T + t
        cam.points = o3d.utility.Vector3dVector(pts_world)
        vis.add_geometry(cam)

    cam_centers = np.array(cam_centers)

    # === 3️⃣ 畫出相機軌跡線（綠色） ===
    if len(cam_centers) > 1:
        lines = [[i, i+1] for i in range(len(cam_centers)-1)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(cam_centers),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.paint_uniform_color([0, 1.0, 0])  # 🟢 綠色軌跡線
        vis.add_geometry(line_set)

    # === 4️⃣ 座標軸 ===
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(axes)

    vis.run()
    vis.destroy_window()



# ＝＝＝＝
if __name__ == "__main__":
    # Load data
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    # IMAGE_ID_LIST = [200,201]  
    val_ids = []
    for i, name in enumerate(images_df["NAME"]):
        if "val" in name or i >= 163:
            val_ids.append(images_df.loc[i, "IMAGE_ID"])

    IMAGE_ID_LIST = val_ids
    print(f"[INFO] Total validation images selected: {len(IMAGE_ID_LIST)}")



    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []

    c2w_list = []

    for idx in tqdm(IMAGE_ID_LIST):
        # Load query image（僅讀取路徑；若需要可用於顯示）
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        _ = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp（手刻 RANSAC + P3P）
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        r_list.append(rvec)
        t_list.append(tvec)

        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values  # [x,y,z,w]
        tvec_gt = ground_truth[["TX","TY","TZ"]].values

        # Calculate error（使用估計 vs GT）
        if retval:
            rotq_est = R.from_rotvec(rvec.reshape(1,3)).as_quat()   # [x,y,z,w]
            t_est = tvec.reshape(1,3)
            r_error = rotation_error(rotq_est, rotq_gt)[0]
            t_error = translation_error(t_est, tvec_gt)[0]
        else:
            r_error, t_error = np.nan, np.nan
        rotation_error_list.append(r_error)
        translation_error_list.append(t_error)

        # 累積 c2w for visualization（c2w = [R^T | -R^T t]）
        if retval:
            R_wc = R.from_rotvec(rvec.reshape(1,3)).as_matrix()[0].T
            C = -R_wc @ tvec.reshape(3)
            c2w = np.eye(4)
            c2w[:3,:3] = R_wc
            c2w[:3, 3] = C
            c2w_list.append(c2w)

    # 統計中位數誤差（去除 NaN）
    rot_err_np  = np.array([e for e in rotation_error_list if np.isfinite(e)], dtype=np.float64)
    trans_err_np= np.array([e for e in translation_error_list if np.isfinite(e)], dtype=np.float64)
    if rot_err_np.size > 0 and trans_err_np.size > 0:
        print("\n=== Q2-1 Pose Error (Median) — Self P3P + RANSAC ===")
        print(f"Rotation (deg)   median = {np.median(rot_err_np):.3f}")
        print(f"Translation (m)  median = {np.median(trans_err_np):.3f}")
        print(f"(valid frames: {len(rot_err_np)})")
    else:
        print("\n⚠️ 沒有有效姿態；可嘗試：LOWE_RATIO=0.8、RANSAC_REPROJ_ERR=6.0、RANSAC_MAX_ITERS=8000、LOCAL_REFINE_ITERS=0")

    # 結果視覺化（自動開啟 Open3D GUI）
    Camera2World_Transform_Matrixs = c2w_list
    if len(Camera2World_Transform_Matrixs) > 0:
        visualization(Camera2World_Transform_Matrixs, points3D_df)

    # === 輸出每張影像的姿態到 CSV（Q2-2
    rows = []
    for idx, (rvec, tvec) in zip(IMAGE_ID_LIST, zip(r_list, t_list)):
        if rvec is None or tvec is None:
            continue
        rows.append({
            "IMAGE_ID": idx,
            "rvec_x": float(rvec[0]), "rvec_y": float(rvec[1]), "rvec_z": float(rvec[2]),
            "t_x": float(tvec[0]), "t_y": float(tvec[1]), "t_z": float(tvec[2]),
        })
    poses_df = pd.DataFrame(rows).sort_values("IMAGE_ID")
    poses_df.to_csv("data/poses_est.csv", index=False)
    print("💾 已輸出姿態：data/poses_est.csv")
