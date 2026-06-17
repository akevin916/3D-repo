# 3D-repo
## VGGT-Dyn
這是 **VGGT-Dyn**，一個針對動態場景的幾何重建後處理 pipeline，核心想法是：

**目標**：用 VGGT (frozen) 的 feed-forward 輸出取代 MonST3R 的 MST 初始化，再根據物理限制設計loss對靜態區域做物理約束，以增強動態影片的深度/相機姿態品質——不需重新訓練模型。

**流程**：
1. **VGGT (frozen)** 輸出 `pose_enc`、`depth`、`depth_conf`
2. **RAFT (frozen)** 預先計算光流
3. **VGGTInitializer** 把 VGGT 輸出轉成 MonST3R optimizer 的初始狀態（跳過 MST，迭代數從 300 降到 ~40）
4. **VGGTDynOptimizer** 用 anchor loss + flow  loss + depth regularization + smooth loss 微調 `delta_depth`/`delta_pose`，並更新 dynamic mask（flow residual + 可選 SAM2 refine）

**目錄結構**：
- [run.py](vggt-dyn/run.py) — 多幀推論 + TTO 入口（VGGT-Dyn）
- [run_single_frame.py](vggt-dyn/run_single_frame.py) — 單幀推論入口，無 TTO（原始 VGGT-1B）
- [eval.py](vggt-dyn/eval.py) — 評估調度器
- [visualize.py](vggt-dyn/visualize.py) — 診斷視覺化（GIF / 6-panel inspect）
- [vggt_dyn/](vggt-dyn/vggt_dyn/) — 核心 TTO 套件（pipeline、initializer、optimizer、dynamic_mask）
- [evaluators/](vggt-dyn/evaluators/) — 各資料集評估（bonn/sintel/kitti/scared/dtu）
- [finetune/](vggt-dyn/finetune/) — freeze 微調
- [scripts/batch.py](vggt-dyn/scripts/batch.py) — 統一批次執行器

---

## 執行方式

### 單一 sequence

```bash
cd vggt-dyn

# 多幀推論 + TTO
python run.py \
  --images "data/sintel/training/final/alley_2/*.png" \
  --ckpt checkpoints/vggt_dyn.pt \
  --raft path/to/raft-things.pth \
  --output outputs/alley_2

# 單幀推論（無 TTO，原始 VGGT-1B baseline）
python run_single_frame.py \
  --images "data/sintel/training/final/alley_2/*.png" \
  --checkpoint vggt/checkpoints/VGGT-1B.pt \
  --output outputs/alley_2_single

# 視覺化
python visualize.py gif --output_dir outputs/alley_2 --raft path/to/raft-things.pth
```

### 批次執行（scripts/batch.py）

`batch.py` 以子命令區分評估目標：

| 子命令 | 說明 | Metrics |
|--------|------|---------|
| `depth --mode multi` | 多幀 TTO，Bonn / Sintel / KITTI | AbsRel / RMSE / δ |
| `depth --mode single` | 單幀無 TTO，Bonn / Sintel / KITTI | AbsRel / RMSE / δ |
| `pose` | 相機姿態，僅 Sintel | ATE / RPE |
| `mask` | Dynamic mask 品質，僅 Sintel | IoU / precision / recall |
| `viz` | 批次產生 GIF，對已有的 run 輸出 | — |

```bash
# 典型完整流程
python scripts/batch.py depth --dataset sintel \
  --ckpt checkpoints/vggt_dyn.pt --raft path/to/raft-things.pth

python scripts/batch.py pose \
  --ckpt checkpoints/vggt_dyn.pt --raft path/to/raft-things.pth

python scripts/batch.py mask --target vggt_dyn \
  --pred_root outputs/sintel_pose_batch

python scripts/batch.py viz \
  --output_root outputs/sintel_batch --raft path/to/raft-things.pth
```

常用選項：`--stage run|eval|all`、`--sequences a,b`、`--full_seq`、`--skip_existing`、`--dry_run`

詳細說明見 [vggt-dyn/README.md](vggt-dyn/README.md) 與 [vggt-dyn/evaluators/README.md](vggt-dyn/evaluators/README.md)。

