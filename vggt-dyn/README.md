# vggt-dyn

**VGGT-Dyn** 是一個針對動態場景的幾何重建後處理 pipeline。
以 VGGT 的 feed-forward 輸出作為高品質初始值，取代 MonST3R global optimization 的 MST 初始化，並引入 optical flow consistency loss 對靜態區域施加物理約束，從而在不重新訓練任何模型的條件下強化動態影像的深度與相機姿態品質。

> **注意**：本 README 已隨 2026-06 重構更新。舊版批次腳本（`depth_batch.py`、`kitti_batch.py`、`sintel_pose_batch.py`、`vggt_single_frame_batch.py`）已整合為 `scripts/batch.py`，舊版 `sintel_pose_eval.py` / `inspect_components.py` 亦已移至對應子目錄。

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

## 專案結構

```
vggt-dyn/
├── README.md
│
├── run.py                        # TTO 優化 CLI 入口（thin wrapper → vggt_dyn/pipeline.py）
├── eval.py                       # 評估調度器：dispatches to evaluators/
├── run_vggt_single_frame.py      # （已移至 scripts/vggt_baseline.py）
│
├── vggt_dyn/                     # 核心 TTO 套件（inference-time only）
│   ├── __init__.py
│   ├── pipeline.py               # 主流程：VGGT → init → optimize → output
│   ├── initializer.py            # VGGT 輸出 → optimizer 初始狀態
│   ├── optimizer.py              # TTO 優化器（anchor + flow + depth_reg loss）
│   ├── dynamic_mask.py           # ego-flow 殘差 → dynamic_mask，SAM2 refine
│   └── utils/
│       ├── flow_utils.py         # RAFT 相鄰幀光流計算
│       └── pose_utils.py         # pose_enc 解碼 / 編碼 / 合成
│
├── evaluators/                   # 各資料集評估器（見 evaluators/README.md）
│   ├── base.py, metrics.py
│   ├── bonn.py                   # Bonn RGB-D (AbsRel/RMSE/δ)
│   ├── sintel.py                 # MPI-Sintel depth (.dpt)
│   ├── sintel_pose.py            # Sintel camera pose (ATE/RPE)
│   ├── kitti.py                  # KITTI Eigen split
│   ├── scared.py                 # SCARED surgical (Chamfer)
│   └── dtu.py                    # DTU multi-view (Chamfer + Sim(3))
│
├── finetune/                     # MonST3R-style freeze 微調工具（與 TTO 核心分離）
│   ├── datasets/
│   │   ├── __init__.py           # dataset registry / builder
│   │   ├── common.py             # 共用影像/相機/視窗工具
│   │   ├── point_odyssey.py      # PointOdyssey clip dataset
│   │   ├── tartanair.py          # TartanAir clip dataset
│   │   ├── spring.py             # Spring clip dataset
│   │   └── waymo.py              # Waymo clip dataset
│   └── train.py                  # 訓練主流程 CLI（單資料集 / 混合比例）
│
├── scripts/                      # 批次執行 & 工具腳本
│   ├── batch.py                  # 統一批次執行器（depth / pose / single_frame）
│   ├── vggt_baseline.py          # 原生 VGGT 無 TTO 推理（baseline）
│   ├── inspect_components.py     # 元件診斷視覺化
│   ├── visualize.py              # 診斷 GIF（RGB/Depth/Mask/Conf/RAFT-flow/Ego-flow）
│   └── collect_eval_sum.py       # 匯整評估 JSON 至 eval_sum/
│
└── third_party/
    ├── goem_opt.py               # DepthBasedWarping, OccMask 等幾何工具
    ├── raft.py                   # RAFT 載入封裝
    ├── RAFT/
    └── sam2/
```

### 各模組職責

#### `vggt_dyn/pipeline.py`

整合 VGGT → RAFT → Initializer → Optimizer 的完整流程，可程式化呼叫：

```python
from vggt_dyn.pipeline import run_pipeline
run_pipeline(args)   # args 同 run.py parse_args() 的回傳物件
```

#### `vggt_dyn/initializer.py` — VGGTInitializer

接收 VGGT 的 raw 輸出，轉換為 optimizer 所需初始狀態：

- `im_poses[i]`：從 `pose_enc` 解碼 extrinsics，取 inverse 得 cam-to-world
- `im_focals[i]`：從 `fov_w` 計算 $f_x = \frac{W/2}{\tan(\text{fov\_w}/2)}$
- `im_depthmaps[i]`：`log(depth)` 對應 log-depth 格式
- `conf_weights[i]`：直接使用 VGGT 的 `depth_conf`

#### `vggt_dyn/optimizer.py` — VGGTDynOptimizer

Test-time 優化器，持有可訓練殘差 `delta_depth` / `delta_rotvec` / `delta_t`：

- flow loss 從第 0 步啟用（初始值夠好，無需 warm-up）
- `depth = vggt_depth * exp(delta_depth)`，`pose = vggt_pose ⊕ delta_pose`

#### `vggt_dyn/dynamic_mask.py`

兩階段動態 mask 生成：

1. **Flow residual mask**：`|ego_flow - raft_flow| > threshold`
2. **SAM2 refine**（選用）：以 Stage 1 mask 為 prompt 精化語意邊界

#### `finetune/`

MonST3R-style freeze 微調工具，與核心 TTO 套件完全分離：

- `datasets/`：依資料集模組化管理（`point_odyssey`、`tartanair`、`spring`、`waymo`）
- `train.py`：凍結大部分 backbone，只訓練末端 blocks 與 heads；支援單資料集與混合比例抽樣

#### `scripts/batch.py`

統一批次執行器，取代舊版四支 batch 腳本，以子命令區分：

```bash
python scripts/batch.py depth        --dataset bonn|sintel|kitti ...
python scripts/batch.py pose         ...   # Sintel pose eval
python scripts/batch.py single_frame --dataset bonn|sintel|kitti ...
```

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

建議流程：**重建 → 評估 → 診斷**。

### 1) 重建（run.py）

```bash
cd vggt-dyn
python run.py \
  --images "path/to/frames/*.png" \
  --ckpt ../vggt/checkpoints/VGGT-1B.pt \
  --raft ../Endo3R/checkpoints/raft-things.pth \
  --output outputs/my_scene \
  --preprocess long_edge \
  --niter 50 \
  --device cuda
```

輸出：`depth/`、`depth_orig_res/`、`dynamic_mask/`、`extrinsics.npy`、`intrinsics.npy`、`pts3d.npy`、`metrics.json`

前處理選項：`letterbox`（補邊正方形）/ `center_crop`（中心裁切）/ `long_edge`（長邊縮放 518px，KITTI 常用）

### 2) 評估（eval.py）

```bash
python eval.py <dataset> --output_dir <dir> --align_scale_mode <mode> [dataset 專屬參數]
```

支援資料集：`bonn`、`sintel`、`sintel_pose`、`kitti`、`scared`、`dtu`

各資料集完整指令、參數說明與 `--align_scale_mode` 對照表見 **[evaluators/README.md](evaluators/README.md)**。

### 2.1) 批次評估（scripts/batch.py）

取代舊版四支 batch 腳本，以子命令區分：

```bash
python scripts/batch.py depth        --dataset bonn|sintel|kitti --stage run|eval|all ...
python scripts/batch.py pose         --stage run|eval|all ...          # Sintel 姿態
python scripts/batch.py single_frame --dataset bonn|sintel|kitti ...  # VGGT baseline
```

完整批次指令見 **[evaluators/README.md#批次評估](evaluators/README.md#批次評估)**。

常用選項：`--full_seq`、`--sequences a,b,c`、`--loss_version mon|dyn`、`--skip_existing`、`--dry_run`

### 3) 視覺化診斷

```bash
# 元件診斷：flow / residual / mask 視覺化
python scripts/inspect_components.py \
  --images "path/to/frames/*.png" \
  --from_run outputs/my_scene \
  --raft ../Endo3R/checkpoints/raft-things.pth \
  --save_dir outputs/inspect_my_scene --gif

# 基本：RGB / Depth / Dynamic Mask / Conf（4-panel GIF + stats.json）
# --images 可省略，會自動讀取 metrics.json 內的 image_paths
python scripts/visualize.py --output_dir outputs/my_scene

# 含 RAFT flow / ego-flow（6-panel，並重算 dynamic mask 做一致性比對）
python scripts/visualize.py \
  --output_dir outputs/my_scene \
  --raft ../Endo3R/checkpoints/raft-things.pth
```

輸出：`<output_dir>/viz.gif`、`<output_dir>/stats.json`（含每幀 dyn%、conf 範圍，給 `--raft` 時還有 raft/ego flow 量值與 residual 統計）。

### 4) 建議的最小實驗流程

1. `--max_frames 20` 小規模重建，確認不 OOM
2. `scripts/visualize.py --raft ...` 確認 flow / residual / mask / conf 合理
3. 完整序列 + `eval.py` / `scripts/batch.py` 產生最終指標
4. `scripts/collect_eval_sum.py` 匯整多次實驗 JSON 至 `outputs/eval_sum/`

---

## 微調（MonST3R-style freeze，選用）

```bash
cd vggt-dyn

# 單一資料集（PointOdyssey）
python finetune/train.py \
  --dataset point_odyssey \
  --root ../data/point_odyssey \
  --split train \
  --ckpt ./finetune/checkpoints/VGGT-1B.pt \
  --output finetune_outputs/po_freeze \
  --epochs 3 --train_last_n_blocks 8 --amp

# 混合比例抽樣（MonST3R 比例範例）
OPENCV_IO_ENABLE_OPENEXR=1 python finetune/train.py \
  --mix \
  --mix_datasets point_odyssey,tartanair,spring,waymo \
  --mix_weights 10000,5000,1000,4000 \
  --mix_roots point_odyssey=../data/point_odyssey,tartanair=../data/tartanair,spring=../data/spring,waymo=../data/waymo_processed \
  --mix_samples_per_epoch 20000 \
  --ckpt ./finetune/checkpoints/VGGT-1B.pt \
  --output finetune_outputs/mix_freeze \
  --epochs 10 --train_last_n_blocks 0 --clip_len 2 --amp
```

---

## 評估器詳細說明

詳見 [evaluators/README.md](evaluators/README.md).
