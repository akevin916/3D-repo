# evaluators — 評估器使用說明

`evaluators/` 包含 VGGT-Dyn 所有 benchmark 的評估邏輯。
透過根目錄的 `eval.py` 統一分派，也可直接 import 到程式中使用。

---

## 目錄結構

```
evaluators/
├── base.py          — BaseEvaluator 抽象基底類別、共用 I/O helper
├── metrics.py       — 共用 metric 函式（深度、點雲、對齊）
├── bonn.py          — Bonn RGB-D 動態室內場景
├── sintel.py        — MPI-Sintel 深度
├── sintel_pose.py   — MPI-Sintel 相機姿態
├── kitti.py         — KITTI Eigen split 單目深度
├── scared.py        — SCARED 手術場景
└── dtu.py           — DTU 多視角重建
```

---

## 共用參數

所有子命令都接受以下共用參數：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--output_dir` | （必填）| `run.py` 的輸出目錄 |
| `--align_scale_mode` | `none` | 對齊模式（見下表） |

### align_scale_mode 說明

| 模式 | 說明 | 適用場景 |
|---|---|---|
| `none` | 不做對齊，直接計算 | 有 metric scale 的輸出 |
| `single_frame` | 每幀各自 median scale 對齊，valid_pixels 加權平均 | MonST3R single-frame 評估協定 |
| `median` | 全序列 median scale 對齊 | 快速對齊估計 |
| `scale_only` | 全序列單一 scale（無 shift），魯棒迭代 | KITTI 標準 scale-only |
| `scale_and_shift` | 全序列 scale + shift 最小二乘 | MonST3R 常用對齊 |
| `sim3` | Umeyama Sim(3) 透過 GT 相機中心 | DTU、SCARED |

---

## 各評估器使用方式

### Bonn RGB-D（動態室內深度）

**指標**：AbsRel、SqRel、RMSE、RMSElog、δ<1.25/1.25²/1.25³

**GT 格式**：`depth/` 目錄下的 PNG uint16（除以 5000 = 公尺）

```bash
cd vggt-dyn
python eval.py bonn \
  --output_dir outputs/bonn_eval/rgbd_bonn_balloon2 \
  --scene_dir  ../data/bonn/rgbd_bonn_dataset/rgbd_bonn_balloon2 \
  --align_scale_mode scale_and_shift \
  --max_depth 70 \
  --min_depth 0.1
```

**輸出 JSON**：`eval_bonn_rgbd_bonn_balloon2_scale_shift.json`

| 參數 | 預設 | 說明 |
|---|---|---|
| `--scene_dir` | （必填）| Bonn 場景目錄（含 `rgb/`、`depth/`、`groundtruth.txt`） |
| `--max_depth` | `10.0` | 深度上限（公尺），超過則忽略 |
| `--min_depth` | `0.1` | 深度下限（公尺） |

---

### MPI-Sintel 深度

**指標**：AbsRel、SqRel、RMSE、RMSElog、δ<1.25/1.25²/1.25³

**GT 格式**：`.dpt` 二進位深度圖（Sintel 專用格式）

```bash
cd vggt-dyn
python eval.py sintel \
  --output_dir outputs/sintel_eval/alley_2 \
  --scene_dir  ../data/sintel/training/depth/alley_2 \
  --align_scale_mode scale_and_shift \
  --max_depth 70 \
  --post_clip_max 70
```

**輸出 JSON**：`eval_sintel_alley_2_scale_shift.json`

| 參數 | 預設 | 說明 |
|---|---|---|
| `--scene_dir` | （必填）| Sintel 序列深度目錄（含 `.dpt` 檔） |
| `--max_depth` | `70.0` | 深度上限（公尺），`0` 表示無上限 |
| `--min_depth` | `0.0` | 深度下限 |
| `--post_clip_max` | `70.0` | 計算前 clip 預測深度（`0` 可關閉，對齊 MonST3R notebook） |

---

### MPI-Sintel 相機姿態

**指標**：ATE RMSE（Sim(3) 對齊）、RPE 平移 RMSE、RPE 旋轉 RMSE（deg，delta=1 frame）

**GT 格式**：`camdata_left/<seq>/frame_XXXX.cam`（Sintel 二進位格式）

```bash
cd vggt-dyn
python eval.py sintel_pose \
  --output_dir outputs/sintel_pose_batch/alley_2 \
  --gt_cam_dir ../data/sintel/training/camdata_left/alley_2 \
  --pose_eval_stride 1
```

**輸出**：
- `eval_sintel_pose.json` — 數值指標
- `pred_traj.txt`、`pred_traj_aligned.txt`、`gt_traj.txt` — TUM 格式軌跡
- `pose_eval_metric.txt` — 人類可讀摘要

| 參數 | 預設 | 說明 |
|---|---|---|
| `--gt_cam_dir` | （必填）| Sintel `camdata_left/<seq>` 目錄 |
| `--pose_eval_stride` | `1` | 抽幀 stride（對齊 MonST3R 預設 1） |
| `--align_scale_mode` | （共用，此處不使用）| sintel_pose 固定使用 Sim(3) 內部對齊 |

---

### KITTI Eigen split（單目深度）

**指標**：AbsRel、SqRel、RMSE、RMSElog、δ<1.25/1.25²/1.25³

**GT 格式**：
- `.npy` float32（公尺）
- 或 `.png` uint16（除以 256 = 公尺，KITTI raw 格式）

```bash
cd vggt-dyn
# 標準協議（per-sequence median scale）
python eval.py kitti \
  --output_dir outputs/kitti_long/2011_09_26_drive_0002_sync \
  --gt_dir     ../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth \
  --drive      2011_09_26_drive_0002_sync \
  --align_scale_mode median \
  --max_depth 80

# MonST3R 對齊協議（scale + shift）
python eval.py kitti \
  --output_dir outputs/kitti_long/2011_09_26_drive_0002_sync \
  --gt_dir     ../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth \
  --drive      2011_09_26_drive_0002_sync \
  --align_scale_mode scale_and_shift \
  --max_depth 0 \
  --no_eigen_crop

# MonST3R single-frame 協議（每幀各自對齊）
python eval.py kitti \
  --output_dir outputs/kitti_long/2011_09_26_drive_0002_sync \
  --gt_dir     ../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth \
  --drive      2011_09_26_drive_0002_sync \
  --align_scale_mode single_frame \
  --max_depth 0 \
  --no_eigen_crop

# per-sequence scale only（無 shift）
python eval.py kitti \
  --output_dir outputs/kitti_long/2011_09_26_drive_0002_sync \
  --gt_dir     ../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth \
  --drive      2011_09_26_drive_0002_sync \
  --align_scale_mode scale_only \
  --max_depth 80
```

`single_frame` 的定義：
- 每張 frame 各自做 median scale alignment
- 序列內用 `valid_pixels` 加權平均（與 MonST3R single-frame notebook 聚合方式一致）

**輸出 JSON**：`eval_kitti_median.json` / `eval_kitti_scale_shift.json` / `eval_kitti_single_frame.json` / `eval_kitti_scale_only.json`

| 參數 | 預設 | 說明 |
|---|---|---|
| `--gt_dir` | （必填）| GT 深度目錄 |
| `--max_depth` | `80.0` | 深度上限（公尺），`0` 表示無上限（MonST3R 常用 `0`） |
| `--no_eigen_crop` | `False` | 關閉 Eigen crop（MonST3R 評估用此選項） |
| `--drive` | `None` | 用來過濾 GT 檔名的 drive 字串，例如 `2011_09_26_drive_0002_sync` |

> **Eigen crop**：標準 KITTI 深度評估裁切掉上方 `40%` 及下方 `5px` 的天空/引擎蓋區域。MonST3R 通常關閉此選項（`--no_eigen_crop`）。

---

### SCARED（手術場景深度，mm 精度）

**指標**：雙向 Chamfer 距離（mm）、precision / recall（閾值 1mm、2mm、5mm）

**GT 格式**：Wavefront `.obj` 點雲

```bash
cd vggt-dyn
python eval.py scared \
  --output_dir outputs/scared_eval/my_case \
  --gt_cloud   ../data/SCARED/dataset_8/keyframe_0/point_cloud.obj \
  --align_scale_mode median
```

**輸出 JSON**：`eval_scared_<mode>.json`

| 參數 | 預設 | 說明 |
|---|---|---|
| `--gt_cloud` | （必填）| GT 點雲 `.obj` 路徑 |
| `--gt_calib` | `None` | 內窺鏡標定 YAML（選用） |

> **注意**：SCARED 使用 `pts3d.npy` 作為預測點雲（非深度圖），需要 `run.py` 輸出完整的 `pts3d.npy`。

---

### DTU（多視角重建，Chamfer 距離）

**指標**：準確率（Accuracy）、完整度（Completeness）、整體 Chamfer（Overall）

**GT 格式**：DTU STL 點雲（`Points/stl/stlXXX_total.ply`）

```bash
cd vggt-dyn
# 使用 Sim(3) 對齊（透過 GT 相機中心）
python eval.py dtu \
  --output_dir outputs/dtu_eval/scan24 \
  --stl_dir    ../data/DTU/Points/stl \
  --scan_id    24 \
  --calib_dir  "../data/DTU/SampleSet/MVS Data/Calibration/cal18" \
  --images     "../data/DTU/Rectified/scan24/rect_*_3_r5000.png" \
  --align_scale_mode sim3
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `--stl_dir` | （必填）| DTU `Points/stl/` 目錄 |
| `--scan_id` | （必填）| DTU scan ID，例如 `24` |
| `--calib_dir` | `None` | `pos_XXX.txt` 所在目錄（`sim3` 模式必填） |
| `--images` | `None` | 輸入影像 glob（`sim3` 模式必填，用於找對應相機） |
| `--dataset_dir` | `None` | DTU SampleSet/MVS Data 目錄（選用） |

> **相依**：DTU 評估呼叫外部 `DTUeval-python/eval.py`，需自行安裝 [DTUeval-python](https://github.com/jzhangbs/DTUeval-python)。

---

## 對齊模式與 MonST3R 協議對照

| 評估場景 | 建議模式 | 說明 |
|---|---|---|
| KITTI（MonST3R 對標）| `scale_and_shift` + `--no_eigen_crop` + `--max_depth 0` | MonST3R notebook 設定 |
| KITTI（MonST3R single-frame）| `single_frame` + `--no_eigen_crop` + `--max_depth 0` | 每幀各自對齊 |
| KITTI（標準 Eigen）| `scale_only` + `--max_depth 80` | 傳統協議 |
| Bonn（MonST3R 對標）| `scale_and_shift` + `--max_depth 70` | |
| Sintel（MonST3R 對標）| `scale_and_shift` + `--max_depth 70` + `--post_clip_max 70` | |
| Sintel Pose | 固定 Sim(3) 內部對齊 | ATE/RPE |
| SCARED | `median` 或 `none` | 點雲 Chamfer |
| DTU | `sim3` | 透過 GT 相機中心對齊 |

---

## 批次評估

搭配 `scripts/batch.py` 可一次對多序列執行 run + eval。

### 批次深度評估（Bonn / Sintel / KITTI）

#### KITTI（run + eval）

```bash
cd vggt-dyn
python scripts/batch.py depth \
  --dataset kitti \
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

#### Bonn（MonST3R 子集，run + eval）

```bash
cd vggt-dyn
python scripts/batch.py depth \
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

#### Sintel（MonST3R 子集，run + eval）

```bash
cd vggt-dyn
python scripts/batch.py depth \
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

#### Sintel（只重跑評估）

```bash
cd vggt-dyn
python scripts/batch.py depth \
  --dataset sintel \
  --stage eval \
  --output_root outputs/sintel_batch_dyn \
  --align_scale_mode scale_only \
  --max_depth 70 \
  --post_clip_max 70
```

`depth` 子命令常用選項：
- `--stage run|eval|all`
- `--full_seq`：全序列（預設 MonST3R 子集）
- `--sequences a,b,c`：只跑指定序列
- `--loss_version mon|dyn`
- `--align_scale_mode scale_only|scale_and_shift|single_frame|median`
- `--skip_existing`、`--continue_on_error`、`--dry_run`
- `--use_eigen_crop`：KITTI 開啟 Eigen crop（預設關閉）

### 原生 VGGT Baseline 批次評估（scripts/vggt_baseline.py）

無 TTO 最佳化，純 VGGT feed-forward baseline：

```bash
cd vggt-dyn
python scripts/batch.py single_frame \
  --dataset bonn \
  --stage all \
  --output_root outputs/vggt_single_frame_bonn \
  --checkpoint ../vggt/checkpoints/VGGT-1B.pt \
  --max_depth 70 \
  --device cuda

python scripts/batch.py single_frame \
  --dataset kitti \
  --stage all \
  --image_dir ../data/kitti/depth_selection/val_selection_cropped/image \
  --gt_dir ../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth \
  --output_root outputs/vggt_single_frame_kitti \
  --checkpoint ../vggt/checkpoints/VGGT-1B.pt \
  --max_depth 0 \
  --device cuda
```

### Sintel Pose 批次評估

指標對齊 MonST3R（ATE / RPE trans / RPE rot）：

```bash
cd vggt-dyn
# MonST3R 14-序列子集，run + eval
python scripts/batch.py pose \
  --stage all \
  --image_dir ../data/sintel/training/final \
  --gt_dir ../data/sintel/training/camdata_left \
  --output_root outputs/sintel_pose_batch_dyn \
  --ckpt ../vggt/checkpoints/VGGT-1B.pt \
  --raft ../Endo3R/checkpoints/raft-things.pth \
  --preprocess center_crop \
  --niter 40 \
  --loss_version dyn \
  --device cuda

# 只重跑評估（重用既有 run 輸出）
python scripts/batch.py pose \
  --stage eval \
  --gt_dir ../data/sintel/training/camdata_left \
  --output_root outputs/sintel_pose_batch_dyn
```

---

## metrics.py 函式參考

| 函式 | 說明 |
|---|---|
| `depth_metrics(pred, gt)` | AbsRel / SqRel / RMSE / RMSElog / δ1/2/3 |
| `median_scale_align(pred, gt)` | per-frame 中位數 scale 對齊 |
| `scale_only_fit(pred, gt)` | 全序列單一 scale，魯棒迭代 |
| `scale_and_shift_fit(pred, gt)` | 全序列 scale + shift 最小二乘 |
| `chamfer_metrics(pred_pts, gt_pts)` | 雙向 Chamfer（mm）+ threshold recall |
| `print_metrics(d)` | 格式化列印 metric dict |
