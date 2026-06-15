# Open Issues

當前待解決的問題清單，依優先級排序。每項記錄問題、相關實驗、目前狀態。

---

## 1. VGGT 初始 pose translation/depth scale 在高難場景偏差約2倍

**狀態**: Exp-003 假設未獲驗證，方向修正中

**相關實驗**: [Exp-003](experiment.md), [Exp-005](experiment.md)

**問題（原始）**: cave_2 (高動態/低紋理場景) 在 iter_0000 (優化前) 的 ego_flow 幅度
就比 RAFT 真實flow大約2倍，且帶有額外垂直分量。flow loss 基於此偏大的
ego_flow 算梯度，等於用錯誤訊號修正pose，導致 flow_weight=0 時 cave_2/
temple_3 反而ATE/RPE_r更好。

**Exp-005 更新**: 對 4 場景全部 49 個 adjacent pair 重新計算 static
pixel 上的 ego/raft scale ratio，中位數全部落在 0.95~1.03，**未觀察到
全域 2x 偏差**。cave_2/temple_3 的偏差來自少數 outlier pair（RAFT flow
本身在這些 pair 劇烈跳動），而非系統性 pose-scale 問題。

**目前方向**:
- ~~全域 pose-scale pre-calibration~~ （Exp-005 顯示中位數已 ≈1，此方向擱置）
- 改為調查 outlier pair（temple_3 pair 13/19/21等）的 RAFT flow 品質，
  評估 per-pair flow loss 穩健化（RAFT confidence weighting / outlier
  pair 降權）
- 檢查 ambush_4/6 是否有類似 outlier pair 模式

---

## 2. mix_freeze_v2 finetune：Sintel pose validation持續惡化

**狀態**: 觀察中

**相關實驗**: [Exp-004](experiment.md)

**問題**: epoch1→4，val_ate 0.0262→0.0383 (+46%)，val_rpe_rot 2.03→6.11
(+3倍)，雖然 val_abs_rel (depth) 持續改善 0.0468→0.0367。

**可能方向**:
- 觀察 epoch5+ 是否回穩
- 提高 camera_weight (目前0.5)
- 調整訓練方式

---

## 3. (待評估) VGGT原生track是否可取代RAFT flow

**狀態**: 構想階段，未開始

**問題**: VGGT TrackHead 為 query-based track，需額外實作 dense grid
query + 推論流程才能產生 dense flow，工作量中等。

**可能方向**: 在問題1/3確認瓶頸不是RAFT本身的情況下，優先度較低。
