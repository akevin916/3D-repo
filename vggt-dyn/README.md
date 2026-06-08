# vggt-dyn

**VGGT-Dyn** 是一個針對動態場景的幾何重建後處理 pipeline。
以 VGGT 的 feed-forward 輸出作為高品質初始值，取代 MonST3R global optimization 的 MST 初始化，並引入 optical flow consistency loss 對靜態區域施加物理約束，從而在不重新訓練任何模型的條件下強化動態影像的深度與相機姿態品質。

---

## 動機

| 問題 | 說明 |
|---|---|
| VGGT 缺乏動態感知 | VGGT 的 `world_points` loss 對所有 pixel 一視同仁，動態物體產生矛盾的監督訊號 |
| MonST3R 初始化耗時 | MonST3R 的 MST 初始化從 pairwise prediction 拼接，累積誤差大，需要 ~300 次迭代優化 |
| 重新訓練成本過高 | 修改 VGGT 訓練 pipeline 需要帶有 `dynamic_mask` 標注的大量動態資料集 |

**核心洞察**：VGGT `pose_enc[S,9]` 直接包含 `(T, quat, fov_h, fov_w)`，可完整初始化 MonST3R optimizer 的所有參數（`im_poses`、`im_focals`、`im_depthmaps`），完整跳過 MST 步驟，優化迭代數從 300 降至 ~40。

---

## 架構概覽

```
Input: video frames [S, 3, H, W]
         │
         ▼
┌─────────────────────┐
│   VGGT (frozen)     │  feed-forward, 不修改任何參數
│                     │
│  pose_enc [S,9]     │──► T, R(quat), fov_h, fov_w
│  depth    [S,H,W]   │──► per-frame 深度圖
│  depth_conf[S,H,W]  │──► 作為優化權重
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  RAFT (frozen)      │  預先計算 optical flow
│                     │
│  flow_ij [E,2,H,W]  │  E = frame pair 數量
│  flow_ji [E,2,H,W]  │
│  valid_mask         │  occlusion mask
└─────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  VGGTInitializer                   │
│                                    │
│  pose_enc → extrinsics, intrinsics │  pose_encoding_to_extri_intri()
│  im_poses    ← inv(extrinsics)     │  cam-to-world
│  im_focals   ← fx from intrinsics  │  (W/2) / tan(fov_w/2)
│  im_depthmaps← log(depth)          │  log-space，對應 MonST3R 格式
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  VGGTDynOptimizer                  │  ~40 次迭代（vs MonST3R 的 300 次）
│                                    │
│  優化變數：                         │
│    delta_depth [S,H,W]  (殘差)     │
│    delta_pose  [S,7]    (殘差)     │
│                                    │
│  Loss：                            │
│    L_recon  = conf-weighted 3D pt  │  只在 ~dynamic_mask 的 static pixel
│    L_flow   = smooth_l1(           │  ego-flow vs RAFT flow
│               ego_flow,            │  只在 ~dynamic_mask 的 static pixel
│               raft_flow,           │
│               ~dynamic_mask)       │
│    L_depth  = depth regularization │  動態 pixel 上的 scale-inv 正則化
│                                    │
│  Dynamic Mask 更新（每 N 步）：     │
│    Stage 1: flow residual > thre   │
│    Stage 2: SAM2 refine (optional) │
└────────────────────────────────────┘
         │
         ▼
Output:
  refined_depth  [S, H, W]
  refined_poses  [S, 4, 4]
  dynamic_mask   [S, H, W]
  world_points   [S, H, W, 3]  (dynamic 區域已標記)
```

---

## 模組規劃

```
vggt-dyn/
├── README.md
├── vggt_dyn/
│   ├── __init__.py
│   ├── pipeline.py          # 主流程：VGGT → init → optimize → output
│   ├── initializer.py       # VGGT 輸出 → MonST3R optimizer 初始狀態
│   ├── optimizer.py         # 繼承 MonST3R PointCloudOptimizer，覆蓋初始化
│   ├── dynamic_mask.py      # flow residual → dynamic_mask，SAM2 refine
│   └── utils/
│       ├── __init__.py
│       ├── pose_utils.py    # pose_enc ↔ (R, T, K) 轉換工具
│       └── flow_utils.py    # RAFT wrapper、OccMask、ego-flow 計算
├── run.py                   # CLI 入口
└── eval.py                  # 與 SCARED / DTU 等 benchmark 對接
```

### 各模組職責

#### `initializer.py` — VGGTInitializer

接收 VGGT 的 raw 輸出，產生 MonST3R `PointCloudOptimizer` 所需的初始化參數：

- `im_poses[i]`：從 `pose_enc` 解碼 extrinsics，取 inverse 得 cam-to-world
- `im_focals[i]`：從 `fov_w` 計算 $f_x = \frac{W/2}{\tan(\text{fov\_w}/2)}$
- `im_depthmaps[i]`：取 `log(depth)` 對應 MonST3R 的 log-depth 格式
- `conf_weights[i]`：直接使用 VGGT 的 `depth_conf`，作為 pixel 優化權重

#### `optimizer.py` — VGGTDynOptimizer

繼承 MonST3R 的 `PointCloudOptimizer`，主要覆蓋：

- `__init__`：跳過 `rand_pose` 隨機初始化，直接載入 `VGGTInitializer` 的結果
- `forward`：flow loss 從第 0 步就啟用（無需 warm-up，因為初始值已夠好）
- 優化殘差而非絕對值：`depth = vggt_depth * exp(delta_depth)`，`pose = vggt_pose ⊕ delta_pose`

#### `dynamic_mask.py` — DynamicMaskEstimator

兩階段動態 mask 生成：

1. **Flow residual mask**：`|ego_flow - raft_flow| > pixel_threshold`，對應 MonST3R 的 `get_motion_mask_from_pairs`
2. **SAM2 refine**（選用）：以 Stage 1 mask 為 prompt，輸入 SAM2 得到語意邊界精確的 mask，對應 MonST3R 的 `refine_motion_mask_w_sam2`

#### `pipeline.py` — VGGTDynPipeline

整合上述模組的主流程，輸入影片序列，輸出精化結果。

---

## 與原始方法的差異對照

| 項目 | MonST3R (原始) | VGGT-Dyn (本專案) |
|---|---|---|
| 模型前向 | DUSt3R pairwise prediction | VGGT 全幀同時 feed-forward |
| 初始化方式 | MST 從 pairwise 拼接，有累積誤差 | VGGT 直接輸出，無累積誤差 |
| Focal 初始化 | `estimate_focal(depth)` 近似反推 | VGGT `fov_h/fov_w` 精確解碼 |
| 優化迭代數 | ~300 | ~40 |
| Flow loss warm-up | 需 warm-up（初始值差，過早加 flow loss 不穩定） | 從第 0 步啟用 |
| 重新訓練 | 否 | 否 |
| 模型修改 | 否 | 否 |

---

## 依賴

- VGGT（`../vggt`）
- MonST3R（`../monst3r`）：借用 `DepthBasedWarping`、`depth_regularization_si_weighted`、`OccMask`、RAFT wrapper
- RAFT（`../monst3r/third_party/RAFT`）
- SAM2（`../monst3r/third_party/sam2`）（選用）

---

## 使用方法

本專案建議以「先重建、再評估、最後診斷」的流程使用。

### 1) 先跑重建（run.py）

```bash
cd vggt-dyn
python run.py \
  --images "../data/kitti/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync*.png" \
  --ckpt ../vggt/checkpoints/VGGT-1B.pt \
  --raft ../Endo3R/checkpoints/raft-things.pth \
  --output outputs/kitti_long/2011_09_26_drive_0002_sync \
  --preprocess long_edge \
  --niter 50 \
  --device cuda \
  --verbose
```

輸出目錄會包含：

- `depth/`：每幀深度
- `depth_orig_res/`：轉回原始解析度後的深度（評估通常讀這個）
- `dynamic_mask/`：每幀動態遮罩
- `extrinsics.npy`, `intrinsics.npy`, `pts3d.npy`, `metrics.json`

前處理選項：

- `letterbox`：短邊補邊到正方形
- `center_crop`：中心裁切正方形
- `long_edge`：長邊縮放到 518、短邊等比例（KITTI 常用）

### 2) 跑評估（eval.py）

KITTI 輸出的準確率指標中：`d1` 即為 `δ < 1.25`（另外 `d2`/`d3` 對應 `δ < 1.25^2` / `δ < 1.25^3`）。

#### KITTI（標準協議）

```bash
cd vggt-dyn
python eval.py kitti \
  --output_dir outputs/kitti_long/2011_09_26_drive_0002_sync \
  --gt_dir ../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth \
  --drive 2011_09_26_drive_0002_sync \
  --align_scale_mode median \
  --max_depth 80
```

#### KITTI（Single-frame depth evaluation，對齊 MonST3R）

```bash
cd vggt-dyn
python eval.py kitti \
  --output_dir outputs/kitti_long/2011_09_26_drive_0002_sync \
  --gt_dir ../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth \
  --drive 2011_09_26_drive_0002_sync \
  --align_scale_mode single_frame \
  --max_depth 0 \
  --no_eigen_crop
```

`single_frame` 的定義：

- 每張 frame 各自做 median scale alignment
- 序列內用 `valid_pixels` 加權平均（與 MonST3R single-frame notebook 聚合方式一致）

#### KITTI（MonST3R 對齊協議）

```bash
cd vggt-dyn
python eval.py kitti \
  --output_dir outputs/kitti_long/2011_09_26_drive_0002_sync \
  --gt_dir ../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth \
  --drive 2011_09_26_drive_0002_sync \
  --align_scale_mode scale_and_shift \
  --max_depth 0 \
  --no_eigen_crop
```

#### KITTI（per-sequence scale only）

```bash
cd vggt-dyn
python eval.py kitti \
  --output_dir outputs/kitti_long/2011_09_26_drive_0002_sync \
  --gt_dir ../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth \
  --drive 2011_09_26_drive_0002_sync \
  --align_scale_mode scale_only \
  --max_depth 80
```

#### Bonn（MonST3R 對齊協議）

```bash
cd vggt-dyn
python eval.py bonn \
  --output_dir outputs/bonn_eval/rgbd_bonn_balloon2 \
  --scene_dir ../data/bonn/rgbd_bonn_dataset/rgbd_bonn_balloon2 \
  --align_scale_mode scale_and_shift \
  --max_depth 70
```

#### Sintel（MonST3R 對齊協議）

```bash
cd vggt-dyn
python eval.py sintel \
  --output_dir outputs/sintel_eval/alley_2 \
  --scene_dir ../data/sintel/depth/alley_2 \
  --align_scale_mode scale_and_shift \
  --max_depth 70 \
  --post_clip_max 70
```

Sintel 也支援：

- `--align_scale_mode scale_only`
- `--align_scale_mode single_frame`

其中 `--post_clip_max 70` 對齊 MonST3R notebook 的 post-clip 行為（`0` 可關閉）。

#### SCARED

```bash
cd vggt-dyn
python eval.py scared \
  --output_dir outputs/scared_eval/my_case \
  --gt_cloud ../data/SCARED/dataset_8/keyframe_0/point_cloud.obj \
  --align_scale_mode median
```

#### DTU

```bash
cd vggt-dyn
python eval.py dtu \
  --output_dir outputs/dtu_eval/scan24 \
  --stl_dir ../data/DTU/Points/stl \
  --scan_id 24 \
  --calib_dir "../data/DTU/SampleSet/MVS Data/Calibration/cal18" \
  --images "../data/DTU/Rectified/scan24/rect_*_3_r5000.png" \
  --align_scale_mode sim3
```

### 2.1) KITTI 一次跑完整序列（run + eval）

新增 `kitti_batch.py`，可一次對所有 KITTI drive 序列執行 `run.py` 與 `eval.py`，並輸出整體 summary。

```bash
cd vggt-dyn
python kitti_batch.py \
  --stage all \
  --image_dir ../data/kitti/depth_selection/val_selection_cropped/image \
  --gt_dir ../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth \
  --output_root outputs/kitti_batch_dyn \
  --ckpt ../vggt/checkpoints/VGGT-1B.pt \
  --raft ../Endo3R/checkpoints/raft-things.pth \
  --preprocess long_edge \
  --niter 50 \
  --loss_version dyn \
  --align_scale_mode scale_and_shift \
  --device cuda
```

```bash
cd vggt-dyn
python kitti_batch.py \
  --stage eval \
  --output_root outputs/kitti_batch_dyn \
  --gt_dir ../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth \
  --align_scale_mode scale_only \
  --max_depth 0
```

`kitti_batch.py` 的評估預設已經是 MonST3R 對標：

- `--max_depth 0`（無深度上限）
- `--no_eigen_crop`（預設關閉 Eigen crop）

因此你通常只需要調整：

- `--align_scale_mode scale_and_shift`（MonST3R 常用）
- 或 `--align_scale_mode scale_only`
- 或 `--align_scale_mode single_frame`（MonST3R single-frame depth protocol）
- `--loss_version mon|dyn`（控制 run.py 使用哪種 loss）

此外，批次 summary 會以每個序列的 `valid_pixels` 做加權平均，與 MonST3R notebook 的聚合方式一致。

常用批次選項：

- `--stage run`：只做重建
- `--stage eval`：只做評估（讀既有輸出）
- `--sequences a,b,c`：只跑指定序列
- `--skip_existing`：跳過已有 `metrics.json` 的序列
- `--continue_on_error`：單一序列失敗時繼續跑後續序列
- `--dry_run`：只印出命令，不真的執行
- `--use_eigen_crop`：改回使用 Eigen crop（預設不開）

### 2.2) Bonn / Sintel 一次跑完整序列（run + eval）

新增 `depth_batch.py`，可對 Bonn 或 Sintel 批次執行 `run.py` + `eval.py`，並輸出加權 summary。

預設資料切分對齊 MonST3R：

- Bonn：預設 5 條序列 `balloon2,crowd2,crowd3,person_tracking2,synchronous`
- Sintel：預設 14 條序列（MonST3R notebook 子集）
- 若要全序列，使用 `--full_seq`

#### Bonn（MonST3R 子集，run + eval）

```bash
cd vggt-dyn
python depth_batch.py \
  --dataset bonn \
  --stage all \
  --output_root outputs/bonn_batch_dyn \
  --ckpt ../vggt/checkpoints/VGGT-1B.pt \
  --raft ../Endo3R/checkpoints/raft-things.pth \
  --preprocess long_edge \
  --niter 50 \
  --loss_version dyn \
  --align_scale_mode scale_and_shift \
  --max_depth 70 \
  --device cuda
```

#### Bonn（全序列，run + eval）

```bash
cd vggt-dyn
python depth_batch.py \
  --dataset bonn \
  --stage all \
  --full_seq \
  --output_root outputs/bonn_batch_full \
  --ckpt ../vggt/checkpoints/VGGT-1B.pt \
  --raft ../Endo3R/checkpoints/raft-things.pth \
  --preprocess long_edge \
  --niter 50 \
  --loss_version mon \
  --align_scale_mode scale_and_shift \
  --max_depth 70 \
  --device cuda
```

#### Sintel（MonST3R 子集，run + eval）

```bash
cd vggt-dyn
python depth_batch.py \
  --dataset sintel \
  --stage all \
  --output_root outputs/sintel_batch_dyn \
  --ckpt ../vggt/checkpoints/VGGT-1B.pt \
  --raft ../Endo3R/checkpoints/raft-things.pth \
  --preprocess long_edge \
  --niter 50 \
  --loss_version dyn \
  --align_scale_mode scale_and_shift \
  --max_depth 70 \
  --post_clip_max 70 \
  --device cuda
```

```bash
  cd vggt-dyn
  python depth_batch.py \
    --dataset sintel \
    --stage eval \
    --output_root outputs/sintel_batch_dyn \
    --align_scale_mode scale_only \
    --max_depth 70 \
    --post_clip_max 70 \
    --device cuda
```

#### Sintel（全序列，run + eval）

```bash
cd vggt-dyn
python depth_batch.py \
  --dataset sintel \
  --stage all \
  --full_seq \
  --output_root outputs/sintel_batch_full \
  --ckpt ../vggt/checkpoints/VGGT-1B.pt \
  --raft ../Endo3R/checkpoints/raft-things.pth \
  --preprocess long_edge \
  --niter 50 \
  --loss_version mon \
  --align_scale_mode scale_and_shift \
  --max_depth 70 \
  --post_clip_max 70 \
  --device cuda
```

`depth_batch.py` 常用選項：

- `--stage run`：只做重建
- `--stage eval`：只做評估（讀既有輸出）
- `--sequences a,b,c`：只跑指定序列
- `--align_scale_mode scale_only|scale_and_shift|single_frame`
- `--loss_version mon|dyn`：控制 run.py 使用哪種 loss（僅 stage=run/all）
- `--skip_existing`：跳過已有 `metrics.json` 的序列
- `--continue_on_error`：單一序列失敗時繼續跑後續序列
- `--dry_run`：只印出命令，不真的執行

### 2.3) 原版 VGGT Single-frame 批次評估（Bonn / Sintel / KITTI）

若你想要「不做任何 TTO 優化」，直接用原版 VGGT feed-forward 深度做 single-frame depth evaluation，可使用 `vggt_single_frame_batch.py`：

- 推理：呼叫 `run_vggt_single_frame.py`（每幀獨立前向）
- 評估：固定 `align_scale_mode=single_frame`
- Bonn/Sintel 預設序列清單與 MonST3R 子集一致；`--full_seq` 可改跑全序列
- KITTI 預設會自動掃描 `image_dir` 內所有 drive（可用 `--sequences` 指定）

#### Bonn（MonST3R 子集，run + eval）

```bash
cd vggt-dyn
python vggt_single_frame_batch.py \
  --dataset bonn \
  --stage all \
  --output_root outputs/vggt_single_frame_bonn \
  --checkpoint ../vggt/checkpoints/VGGT-1B.pt \
  --max_depth 70 \
  --device cuda
```

#### Bonn（全序列，run + eval）

```bash
cd vggt-dyn
python vggt_single_frame_batch.py \
  --dataset bonn \
  --stage all \
  --full_seq \
  --output_root outputs/vggt_single_frame_bonn_full \
  --checkpoint ../vggt/checkpoints/VGGT-1B.pt \
  --max_depth 70 \
  --device cuda
```

#### Sintel（MonST3R 子集，run + eval）

```bash
cd vggt-dyn
python vggt_single_frame_batch.py \
  --dataset sintel \
  --stage all \
  --output_root outputs/vggt_single_frame_sintel \
  --checkpoint ../vggt/checkpoints/VGGT-1B.pt \
  --max_depth 70 \
  --post_clip_max 70 \
  --device cuda
```

#### Sintel（全序列，run + eval）

```bash
cd vggt-dyn
python vggt_single_frame_batch.py \
  --dataset sintel \
  --stage all \
  --full_seq \
  --output_root outputs/vggt_single_frame_sintel_full \
  --checkpoint ../vggt/checkpoints/VGGT-1B.pt \
  --max_depth 70 \
  --post_clip_max 70 \
  --device cuda
```

#### KITTI（全 drive，run + eval）

```bash
cd vggt-dyn
python vggt_single_frame_batch.py \
  --dataset kitti \
  --stage all \
  --image_dir ../data/kitti/depth_selection/val_selection_cropped/image \
  --gt_dir ../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth \
  --output_root outputs/vggt_single_frame_kitti \
  --checkpoint ../vggt/checkpoints/VGGT-1B.pt \
  --max_depth 0 \
  --device cuda
```

#### KITTI（指定 drive，eval only）

```bash
cd vggt-dyn
python vggt_single_frame_batch.py \
  --dataset kitti \
  --stage eval \
  --sequences 2011_09_26_drive_0002_sync,2011_09_26_drive_0005_sync \
  --gt_dir ../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth \
  --output_root outputs/vggt_single_frame_kitti \
  --max_depth 0
```

`vggt_single_frame_batch.py` 常用選項：

- `--stage run`：只做原版 VGGT 推理
- `--stage eval`：只做 single-frame 評估（讀既有輸出）
- `--sequences a,b,c`：只跑指定序列
- `--max_frames N`：每序列只取前 N 幀（快速測試）
- `--skip_existing`：跳過已有 `manifest.json` 的序列
- `--continue_on_error`：單一序列失敗時繼續跑後續序列
- `--dry_run`：只印出命令，不真的執行
- `--use_eigen_crop`：KITTI 評估開啟 Eigen crop（預設關閉以對齊 MonST3R）

### 2.4) Sintel Pose 評估（MonST3R 對齊，含 batch）

新增兩支工具：

- `sintel_pose_eval.py`：單一序列 pose 評估（讀 `extrinsics.npy` + GT `camdata_left/<seq>`）
- `sintel_pose_batch.py`：批次 `run/eval/all`，預設序列子集與 MonST3R 一致

評估指標對齊 MonST3R：

- ATE（translation RMSE, Sim(3) 對齊 + scale correction）
- RPE trans / RPE rot（delta=1 frame）
- 最終 summary 為跨序列 unweighted mean

#### Sintel pose（MonST3R 子集，run + eval）

```bash
cd vggt-dyn
python sintel_pose_batch.py \
  --stage all \
  --image_dir ../data/sintel/training/final \
  --gt_dir ../data/sintel/training/camdata_left \
  --output_root outputs/sintel_pose_batch_dyn \
  --ckpt ../vggt/checkpoints/VGGT-1B.pt \
  --raft ../Endo3R/checkpoints/raft-things.pth \
  --preprocess center_crop \
  --niter 40 \
  --loss_version dyn \
  --pose_eval_stride 1 \
  --device cuda
```

#### Sintel pose（全序列，run + eval）

```bash
cd vggt-dyn
python sintel_pose_batch.py \
  --stage all \
  --full_seq \
  --image_dir ../data/sintel/training/final \
  --gt_dir ../data/sintel/training/camdata_left \
  --output_root outputs/sintel_pose_batch_full \
  --ckpt ../vggt/checkpoints/VGGT-1B.pt \
  --raft ../Endo3R/checkpoints/raft-things.pth \
  --preprocess center_crop \
  --niter 300 \
  --loss_version mon \
  --pose_eval_stride 1 \
  --device cuda
```

#### 只重跑 pose 評估（重用既有 run 輸出）

```bash
cd vggt-dyn
python sintel_pose_batch.py \
  --stage eval \
  --image_dir ../data/sintel/training/final \
  --gt_dir ../data/sintel/training/camdata_left \
  --output_root outputs/sintel_pose_batch \
  --pose_eval_stride 1
```

`sintel_pose_batch.py` 常用選項：

- `--stage run|eval|all`
- `--full_seq`：由預設 14 條 MonST3R 子集切換為全序列
- `--sequences a,b,c`：只跑指定序列
- `--pose_eval_stride N`：按 stride 抽幀計算 pose 指標
- `--loss_version mon|dyn`：run 階段的優化 loss 版本
- `--skip_existing`、`--continue_on_error`、`--dry_run`

### 3) 做元件診斷（inspect_components.py）

有輸出結果時可直接診斷（Mode A）：

```bash
cd vggt-dyn
python inspect_components.py \
  --images "../data/kitti/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync*.png" \
  --from_run outputs/kitti_batch_dyn/2011_09_26_drive_0002_sync \
  --raft ../Endo3R/checkpoints/raft-things.pth \
  --preprocess long_edge \
  --save_dir outputs/inspect_kitti_0002 \
  --max_frames 20 \
  --gif
```

診斷用途：

- 比對 `RAFT flow` 與 `ego-flow` 是否一致
- 觀察 `residual` 的分布與閾值敏感度
- 觀察 `dynamic mask` 是否誤抓到靜態區域

### 4) 建議的最小實驗流程

1. 先用 `--max_frames 20` 跑小規模重建，確認不會 OOM。
2. 用 `inspect_components.py` 檢查 flow/residual/mask 是否合理。
3. 再跑完整序列，最後用 `eval.py` 產生最終指標。

---

## TODO

- [ ] `initializer.py`：VGGT 輸出 → optimizer 初始狀態
- [ ] `optimizer.py`：繼承 MonST3R optimizer，覆蓋初始化邏輯
- [ ] `dynamic_mask.py`：flow residual mask + SAM2 refine
- [ ] `pipeline.py`：整合主流程
- [ ] `run.py`：CLI 入口
- [ ] `eval.py`：對接 SCARED / DTU benchmark
