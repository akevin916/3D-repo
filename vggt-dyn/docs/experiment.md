# Exp-001

Date: 2026-06-11

## Goal

修正 ego_flow / raft_flow 方向相反問題，驗證是否改善 dynamic_mask 與相機姿態。

## Modification

dynamic_mask.py / optimizer.py

R, T (cam-from-world) 直接傳入 DepthBasedWarping
↓

轉換為 cam-to-world 再傳入

R_c2w = R^T
t_c2w = -R^T @ T

## Result

Baseline (alley_2, 5 frames, niter=5)

ATE = 0.00903 / RPE_t = 0.01952 / RPE_r = 0.11528
mask agreement = 96.8%

Experiment (alley_2, 5 frames, niter=5)

ATE = 0.00800 / RPE_t = 0.01398 / RPE_r = 0.08480
mask agreement = 96.9%

Experiment (Sintel MonST3R 14 scenes, niter=50)

mask agreement = 100% (全部14場景)

| 分組 | 場景數 | ATE | RPE_t | RPE_r |
|---|---|---|---|---|
| 好 (ATE<0.05) | 7 | 0.0210 | 0.0089 | 0.126 |
| 中 (0.05<ATE<0.3) | 5 | 0.2025 | 0.0948 | 1.239 |
| 差 (ATE>0.3) | 2 | 0.9212 | 0.3054 | 4.621 |

## Observation

- alley_2 三項姿態誤差全面下降
- 全量 mask agreement 由 ~96-97% 提升至 100%
- 多數場景 (alley_2, sleeping_1/2, shaman_3, market_2/6, temple_2) ATE < 0.05
- 高動態場景 (cave_2, ambush_4/6, temple_3) RPE_r 仍偏高 (1.9~7.0)，scale 估計偏離 (cave_2=28, temple_2=15.5)

## Hypothesis

convention 修正使 flow loss 提供正確的 pose 梯度方向，
因此改善 dynamic_mask 一致性與整體相機姿態精度。

高動態場景誤差大可能來自初始化本身不穩定，而非 flow convention。

## Next Step

檢查高動態場景 (cave_2/ambush_4/6/temple_3) 的 VGGT 初始化品質與 dynamic_mask threshold。

---

# Exp-002

Date: 2026-06-11

## Goal

Sweep flow_weight / depth_reg_weight / mon_smooth_weight ∈ {0, 0.1, 0.5, 1}，
驗證各 loss 權重對不同難度場景 pose 的影響。

## Result

baseline: flow=1, depth_reg=0.1, smooth=0.1
flow_weight 對 ATE@300 影響 (vs baseline)

| 場景 | 難度 | flow_0 | flow_0.1 |
|---|---|---|---|
| alley_2 | 好 | 0.0047→0.0153 (變差3x) | ≈baseline |
| sleeping_2 | 好 | 0.0044→0.0040 (略好) | ≈baseline |
| ambush_5 | 中 | 0.1454→0.1240 (變好) | ≈baseline |
| market_5 | 中 | 0.2330→0.3790 (明顯變差) | 0.2330→0.2309 (略好) |
| cave_2 | 差 | 1.1492→1.0792 (變好) | ≈baseline |
| temple_3 | 差 | RPE_r 3.49→3.01 (變好) | ATE 0.7422→0.7239 (略好) |

depth_reg_weight (0/0.5/1.0) 與 mon_smooth_weight (0/0.5/1.0)：

全部6場景 ATE@300 幾乎無變化 (差異 < 0.001)

## Observation

- depth_reg_weight、mon_smooth_weight 在測試範圍內對 pose 無顯著作用
- flow_weight 是唯一有效參數，但效果方向因場景而異：
  - 好場景：flow loss 必要 (關閉後明顯變差)
  - 中場景：market_5 必要，ambush_5 反而變好
  - 差場景 (cave_2, temple_3)：flow_0 反而改善 ATE/RPE_r

## Hypothesis

高動態/差場景的 flow loss 可能因 dynamic_mask 誤判（將動態區當靜態），
對 pose 施加錯誤約束，導致關閉 flow loss 反而表現更好。

瓶頸可能在 dynamic_mask 品質，而非 loss 權重設定。

## Next Step

檢查 cave_2/temple_3/ambush_4/6 的 dynamic_mask 品質與 threshold 設定；
depth_reg_weight、mon_smooth_weight 維持預設值即可。

---

# Exp-003

Date: 2026-06-15

## Goal

驗證高難場景下 flow constraints 的失效跟 mask 覆蓋率的關係

## Result

mask 覆蓋率 (iter_0300)

| 場景 | 難度 | mask覆蓋率 |
|---|---|---|
| alley_2 | 好 | 4.2% |
| sleeping_2 | 好 | 18.4% |
| temple_3 | 差 | 21.2% |
| cave_2 | 差 | 36.4% |

cave_2 frame0→1 數值診斷 (iter_0000 vs iter_0050 vs RAFT)

| | ego_flow幅度 | fx | fy |
|---|---|---|---|
| iter_0000 (優化前) | 15.32px | -10.29 | +10.58 |
| iter_0050 | 15.26px | -13.47 | +6.54 |
| RAFT (真實) | 7.66px | -6.95 | +0.93 |

相機相對位姿：rot≈0.5°、純平移為主

公式驗證：f·\|T_rel\|/depth_mean ≈ 13.48px，與 ego_flow(15.26) 吻合 (1.13x)，
與 RAFT(7.66) 不符 (0.57x) → ego_flow 計算本身內部一致，非程式碼scale bug

## Observation

- ego_flow 與 RAFT 方向一致 (fx皆為負，符合相機右移→背景左移的觀察)
- 但 ego_flow 幅度比 RAFT 大約 2 倍，且帶有 RAFT 沒有的垂直分量 (fy)
- iter_0000 (優化前) 就已存在此 2 倍偏差，優化50 iters 未修正
- mask覆蓋率與此偏差程度相關 (cave_2 36.4% 最高)

## Hypothesis

VGGT 對 cave_2 (高動態/低紋理) 場景的初始 pose translation
(相對於 depth scale) 估計偏大約 2 倍，從第一次前向輸出就存在，
非 optimizer 或 ego_flow convention 造成。

flow loss 基於這個偏大的 ego_flow 計算梯度，等於用錯誤訊號修正pose，
解釋了 Exp-003 中 flow_weight=0 對 cave_2/temple_3 反而更好的現象。

## Next Step

- 檢查是否可在優化前加入 pose-scale pre-calibration (用RAFT flow統計校正
  初始 translation scale)，再進入主優化迴圈
- 檢查 temple_3/ambush_4/6 是否有相同的scale偏差模式

---

# Exp-004

Date: 2026-06-15

## Goal

確認進行中的 VGGT finetune (mix_freeze_v2: point_odyssey/tartanair/spring/
waymo, camera_weight=0.5) 是否能改善 Exp-003 發現的 pose scale 偏差問題。

## Result

| Epoch | val_abs_rel (depth) | val_ate | val_rpe_rot |
|---|---|---|---|
| 1 | 0.0468 | 0.0262 | 2.03 |
| 2 | 0.0436 | 0.0311 | 4.49 |
| 3 | 0.0361 | 0.0362 | 3.96 |
| 4 | 0.0367 | 0.0383 | 6.11 |

## Observation

- depth (abs_rel) 持續改善：0.047 → 0.037
- val_loss_camera 略降：0.186 → 0.139
- 但 Sintel val/ate (+46%) 與 val/rpe_rot (+3倍) 持續惡化

## Hypothesis

mix dataset 的 pose 分佈/scale 與 Sintel 差異大，finetune 過程
偏向改善 depth/point loss，camera_weight=0.5 不足以維持 Sintel pose 精度，
導致 pose 在 Sintel 上漂移。

## Next Step

持續觀察 epoch 5+ 是否回穩；若 val_ate/rpe_rot 持續惡化，
考慮提高 camera_weight 或調整 mix 資料分佈。

---

# Exp-005

Date: 2026-06-15

## Goal

驗證 Exp-003 的假設：VGGT 初始 pose translation/depth scale 在高動態場景
（cave_2, temple_3）是否存在系統性 ~2x 偏差。寫獨立診斷腳本
[scripts/scale_diag.py](../scripts/scale_diag.py)，對 4 個場景 (alley_2,
sleeping_2, cave_2, temple_3) 全部 49 個 adjacent pair 重新計算：

- ego_flow（用 iter_0000 raw VGGT extrinsics/intrinsics/depth）
- RAFT flow（重新計算）
- 在 iter_0000 的 "static" 像素（valid_fwd & 不被閾值判定為動態）上，
  取 median(|ego_flow|) / median(|raft_flow|) 作為 scale ratio

## Result

| 場景 | ratio_median (全49 pair) | ratio_mean | mask_cov@iter_0300 |
|---|---|---|---|
| alley_2 | 0.997 | 1.009 | 4.24% |
| sleeping_2 | 1.007 | 1.008 | 18.39% |
| cave_2 | 0.989 | 1.267 | 36.36% |
| temple_3 | 0.953 | 2.010 | 21.20% |

cave_2 pair 0（Exp-003 檢查的同一對）：本次 ego_med=16.17, raft_med=14.95,
ratio=1.08 — 與 Exp-003 報告的 2x (15.32 vs 7.66) 不一致。

temple_3 個別 pair 出現極端 ratio（例如 pair13: ratio=11.86, pair19:
ratio=12.33, pair21: ratio=7.29），但這些 pair 的 raft_med 本身在相鄰 pair
間跳動劇大（pair19 raft=18.6 → pair20 raft=80.6），且這些 pair 的
mask_cov@iter_0300 也偏高 (37-68%)。

## Observation

- 全 4 場景的中位數 ratio 都落在 0.95~1.03，**沒有觀察到 Exp-003 所稱的
  全域 2x scale 偏差**
- cave_2 / temple_3 的 ratio_mean 偏高（1.27 / 2.01）完全由少數
  outlier pair 拉高，非系統性
- 這些 outlier pair 同時是 RAFT flow 自身數值劇烈跳動、且 mask 覆蓋率
  偏高的 pair —— 較可能是 RAFT 在快速運動/低紋理下失效，
  而非 VGGT pose-scale 系統性誤差

## Hypothesis

Exp-003 的 2x 結論可能來自單一像素/單一 pair 的取樣，未能代表整個序列。
"全域 pose-scale pre-calibration" 並非正確修復方向 —— 中位數已經 ≈1。

真正驅動 cave_2/temple_3 表現不佳的，較可能是：
(a) 少數 pair 上 RAFT flow 本身不可靠，污染這些 pair 的 dynamic_mask
    與 flow_loss；
(b) 高 mask 覆蓋率讓可用的 static pixel 過少，flow_loss 統計不穩定。

## Next Step

- 針對 outlier pair（如 temple_3 pair 13/19/21），檢查 RAFT flow 本身
  的品質（與相鄰 pair 比較、可視化）
- 評估是否需要 per-pair（而非全域）的 flow loss 穩健化機制
  （例如 RAFT confidence-based pixel weighting、outlier pair 偵測後降權）
- open_issues.md #1 的 "全域 pose-scale pre-calibration" 方向建議擱置，
  改為調查個別 outlier pair 的 RAFT flow 品質
