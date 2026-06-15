# VGGT 原生訓練 vs vggt-dyn Finetune：逐項合理性分析

> 延續 [project_map.md](project_map.md) 的比較結果，本文針對每個簡化/修改點評估「合理之處」「風險與疑慮」「建議」。

---

## 1. 新增模組

| 模組 | 合理之處 | 風險/疑慮 | 建議 |
|---|---|---|---|
| `finetune/datasets/*`（point_odyssey/tartanair/waymo/spring/sintel） | 原生只支援 CO3D/VKitti，無法支援 MonST3R 風格的動態場景資料；新增是任務必需 | 各資料集的 `extrinsics` 都是各自世界座標系下的 world-to-cam（見下方第 7 節），彼此座標系定義不一致 | 無 |
| `finetune/data_loader.py`（mixed-ratio 採樣） | 對齊 MonST3R 的多資料集混合策略，合理 | 採樣權重（10000:5000:1000:4000）為經驗值，未見自動調整機制 | 可記錄/監控各資料集 loss，必要時動態調整權重 |
| `finetune/validation.py` + `pose_metrics.py`（abs_rel/rmse/d1/ATE/RPE） | **明顯優於原生**：原生驗證只回報 loss，本專案新增任務相關幾何指標，更能反映 finetune 是否真的改善幾何精度 | 無 | 維持 |
| `finetune/model_utils.py` | 把凍結策略、LR schedule、checkpoint I/O 抽出成獨立小工具，符合單機腳本需求 | 無 | 維持 |
| `finetune/losses.py` | 任務導向的單一聚合 loss，必要 | 詳見第 7 節 | 詳見第 7 節 |

**整體評價：新增部分基本合理，是支撐「動態場景 finetune」這個新任務目標所必需的擴充，沒有明顯過度設計。**

---

## 2. 刪除模組

| 被刪模組 | 合理之處 | 風險/疑慮 | 建議 |
|---|---|---|---|
| `trainer.py`（DDP Trainer）、`launch.py`（Hydra）、`train_utils/distributed.py`、`train_utils/checkpoint.py` | 單卡小規模 finetune 不需要多機多卡與 Hydra 設定樹，移除大幅降低複雜度，**合理** | 若未來要 scale up 到多卡，需要重新引入這套機制 | 目前規模下無需處理 |
| `vggt/training/data/*`（composed_dataset, dynamic_dataloader, co3d, vkitti...） | 與任務資料集不符，移除合理 | 無 | 無 |
| `train_utils/optimizer.py`（fvcore CompositeParamScheduler 系統） | 對單一 param group + 單一 cosine 排程而言，這套基於 glob pattern 的多 param-group scheduler 系統確實是過度設計，移除合理 | **但連同這套系統一起消失的「5% linear warmup」是個有意義的功能，不是這套系統本身的累贅**（見第 6 節） | 建議單獨把 warmup 邏輯帶回 `cosine_lr`，不需整套 scheduler 系統 |
| `train_utils/gradient_clip.py`（per-module GradientClipper） | 原生對 aggregator/depth/camera 分別做 grad-norm clip 並各自記錄；finetune 只訓練少數模組（last-N blocks + 3 個 head），用單一 `clip_grad_norm_` 整體 clip，複雜度下降，**大致合理** | 失去「分模組監控 grad norm」的可觀測性，若某個 head（例如 camera_head）grad 爆炸，會被其他模組的正常 grad 稀釋掉，不易第一時間發現 | 可在 TensorBoard 額外記錄 `depth_head`/`camera_head`/`point_head`/`aggregator(last-N)` 各自的 grad norm（不一定要分別 clip，但至少分別監控） |
| `train_utils/normalization.py`（`normalize_camera_extrinsics_and_points_batch`） | 移除使流程變簡單 | **這是本次比較中最值得關注的一項**：原生在計算 loss 前，會把 extrinsics/world_points/depth 轉換到「以第一幀相機為原點、平均距離=1」的座標系——這正是 VGGT 預訓練時定義的輸出座標系（`pose_enc` 的 T、world_points 都是 cam0-relative + 尺度正規化）。finetune 的資料集回傳的是各自資料集**原始世界座標系**下的 extrinsics（見第 7 節），loss 裡只做了「除以 scene_scale」的尺度正規化，**沒有做 cam0-relative 的座標系轉換**。換句話說：predictions 仍假設輸出在「以 cam0 為原點」的框架下，但 GT 的 frame 0 並不一定是單位矩陣 — 兩者參考座標系不一致 | 這不一定是「bug」（finetune 透過梯度可以重新學一套對應關係），但等於**部分抹除了預訓練時學到的 cam0-relative 慣例**，finetune 需要額外的梯度步數去適應新的座標系定義，可能拖慢收斂/降低 sample efficiency。建議：在 `monst3r_style_loss` 中，對 `gt_extrinsics`／`gt_pts` 也做一次「相對於第一幀」的座標轉換（旋轉+平移），再做尺度正規化，使其與預訓練慣例一致 |
| `train_utils/tb_writer.py` / `logging.py`（visuals 記錄） | 簡化為標準 `SummaryWriter`，對 debug 影響不大，**合理** | 失去訓練過程的影像/點雲視覺化，較難肉眼檢查 finetune 是否在學到合理的幾何 | 之後若要 debug 動態物體區域的表現，建議補一個輕量的 visual logging（例如每 N steps 存一張 pred depth vs gt depth 的圖） |
| `vggt/training/loss.py`（MultitaskLoss：多 stage 加權、gradient/normal loss、quantile 過濾、`check_and_fix_inf_nan`） | 簡化為單一 stage、單一聚合 loss，降低複雜度 | 詳見第 7 節，這裡同時失去了「數值穩定性保護」與「outlier 過濾」，而這兩者在**動態場景**資料（GT depth 常有遮擋/穿模噪聲，且動態物體本身對靜態模型來說就是 outlier）中可能更重要，不是單純的「複雜度」問題 | 詳見第 7 節 |

**整體評價：絕大多數刪除（DDP/Hydra/dataset 基礎設施）都是「規模不對等」下合理的簡化。但 `normalization.py` 與 `MultitaskLoss` 的數值保護/outlier 過濾這兩塊，刪除的理由更多是「圖方便」而非「不需要」，是後續最值得補強的兩個點。**

---

## 3. 修改模組

### 3.1 凍結策略：`freeze_modules(["*aggregator*"])` → `apply_monst3r_style_freeze`

- **原生**：凍結整個 aggregator，只訓練三個 head（camera/depth/point）。
- **finetune**：凍結全部，再解凍 aggregator 最後 N 個 frame/global blocks + 三個 head。

**合理之處**：
- 原生的「只訓練 head」策略適用於 head 任務遷移（例如新增一個輸出頭），但動態場景與靜態場景的差異主要來自**特徵層的幾何推理方式**（如何處理移動物體的多視角不一致），單靠 head 很難學到這種行為調整。MonST3R 的做法（部分解凍 backbone 後段）更貼近這個任務需求，**改動方向是對的**。

**風險/疑慮**：
- 可訓練參數量大幅增加（last-N blocks 通常遠大於 3 個 head 的參數量），在：
  1. 沒有 warmup（第 6 節）
  2. 沒有分模組 grad clip（第 2 節）
  3. loss 沒有 outlier 過濾（第 7 節）
  
  三者同時缺席的情況下，「解凍更多參數」會放大訓練不穩定的風險——這三項原本在原生 pipeline 裡是配套出現的「安全機制」，但 finetune 在加大可訓練範圍的同時把這些安全機制都拿掉了，組合起來風險被放大。

**建議**：`train_last_n_blocks` 建議搭配 warmup 一起調，並觀察 last-N blocks 的 grad norm 是否異常偏大。

### 3.2 Checkpoint 格式

- 由 `DDPCheckpointSaver` 改為單一 `torch.save(dict)`，符合單卡腳本需求，**合理，無疑慮**。

---

## 4. Training Pipeline 差異

| 差異點 | 合理之處 | 風險/疑慮 |
|---|---|---|
| 單卡、無 `accum_steps` | clip_len=8 的小 batch 在單卡下通常吃得下，移除 accum 簡化邏輯，**合理** | 若未來要加大 `max_img_per_gpu` 或 batch size，可能需要重新引入 |
| 無 `_apply_batch_repetition`（flip 串接 2x batch） | 屬於資料增強，移除是「少一種增強」而非「壞掉」 | 對動態場景而言，左右翻轉可能改變物體運動方向的語義（例如時間軸不翻），是否適用本身存疑，**移除反而可能是對的**，不一定要加回來 |
| 無 `_process_batch` 座標正規化 | 同第 2 節 `normalization.py` | 同上，是本次分析中最大的疑慮 |
| AMP dtype 未指定（預設 float16）vs 原生可選 bfloat16 | float16 在 finetune（小 lr、少數 step）下通常夠用 | bfloat16 對梯度/loss 數值範圍更穩定；若觀察到 `non-finite loss` 警告頻繁出現，可優先改成 bfloat16 而非調小 lr |
| 非 finite loss 處理：原生整 batch return vs finetune 跳過該 step 繼續 | finetune 的做法（跳過、繼續訓練）對單一不穩定 batch 更寬容，**合理，甚至比原生更穩健** | 但若「跳過」頻繁發生卻沒有上層告警/統計（目前只有 log warning），可能掩蓋系統性問題 | 建議統計 `skip_count / total_steps`，若比例過高應視為訓練不穩定的訊號 |
| 新增 depth/pose 幾何驗證指標 | 對「finetune 是否真的提升動態場景幾何精度」這個目標來說，比原生單看 loss 更直接、更可信，**明顯合理且是改進** | 無 |

---

## 5. Optimizer 差異

| 差異點 | 合理之處 | 風險/疑慮 |
|---|---|---|
| `betas=(0.9, 0.95)` vs 原生預設 `(0.9, 0.999)` | 較小的 β2 讓二階動量估計更快適應近期梯度變化，是 LLM/finetune 圈常見作法，對「少量 step 的 finetune」**合理** | 與原生預訓練時的 optimizer 動態不同，若直接沿用原生 checkpoint 的 optimizer state（`--resume`）做 finetune，beta 不一致可能讓動量估計需要重新「熱機」幾步 |
| 只把 `trainable_params` 放進 optimizer（而非全部參數） | 更乾淨、省記憶體，且結果與原生等價（原生凍結參數 grad 為 None，optimizer 不會更新它們） | 無 |
| 單一 param group（無法針對 last-N blocks vs heads 設不同 lr/wd） | 對目前的凍結策略而言簡單夠用 | 若觀察到 heads 與 last-N blocks 的最佳 lr 差異很大（heads 通常可以用更大 lr，因為是從頭適配新 loss 形式），單一 lr 可能是次優解 |

**建議**：若要微調，下一步優先嘗試「heads 用較大 lr、last-N blocks 用較小 lr」的雙 param group，而不是重建整套 fvcore scheduler 系統。

---

## 6. Scheduler 差異

| 差異點 | 合理之處 | 風險/疑慮 |
|---|---|---|
| 移除 fvcore `CompositeParamScheduler` 整套系統 | 對單一 param group 而言，自訂 `cosine_lr()` 已足夠，**移除系統本身合理** | — |
| **但連同系統一起消失的「前 5% linear warmup」** | — | 原生 warmup（1e-8 → 5e-5，前 5% 訓練）存在的原因是：訓練一開始模型/optimizer state 尚未穩定，大 lr 容易造成梯度尖峰。finetune 在 **解凍參數量更多**（last-N blocks）且**沒有分模組 grad clip、沒有 outlier 過濾**的情況下，反而**完全沒有 warmup**，是三個「安全機制同時消失」中最直接可被驗證/修補的一個 |
| Weight decay 維持常數 | 原生雖然用 scheduler 包裝，但值本身也是 `ConstantParamScheduler(0.05)`，**功能上等價**，無實質差異 | — |

**建議（優先序最高的一項具體修改）**：在 `model_utils.cosine_lr` 加入前 N% step 的 linear warmup（例如 5%，從 `min_lr` 或更小值線性升到 `lr`），不需要引入 fvcore，幾行程式碼即可：

```python
def cosine_lr(base_lr, min_lr, step, total_steps, warmup_ratio=0.05):
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        return min_lr + (base_lr - min_lr) * (step / max(1, warmup_steps))
    ...
```

---

## 7. Loss 差異

### 7.1 GT world points：dataset 提供 vs depth 反投影

- **合理之處**：用 GT depth + extrinsics + intrinsics 反投影出 `gt_pts`，只需要維護「depth + 相機參數」一份 ground truth，避免額外維護 `world_points` 欄位，**減少資料集實作負擔，合理**。

### 7.2 座標系與尺度正規化（與第 2 節 `normalization.py` 同一議題）

- finetune 用「除以 `scene_scale`（GT points 到原點距離的平均值）」做尺度正規化，這部分**方向正確**（VGGT 預訓練時也是除以 avg distance）。
- 但**缺少「轉換到以 cam0 為原點」這一步**。原生先把 world_points/extrinsics 轉到 cam0 坐標系，再除以 avg distance；finetune 是在資料集的原始世界座標系下直接除以 avg distance。
  - 結果：`scene_scale` 數值本身可能類似（都是「平均距離」的量級），但**座標系原點/朝向不同**——原生保證 frame 0 的 extrinsics 在正規化後恆為單位矩陣（R=I, t=0），finetune 的 frame 0 GT extrinsics 正規化後仍是資料集原始世界座標系下的任意 pose。
- **這是本次比較中唯一一個「方向部分對、但不完整」的修改**：尺度對齊有做，座標系對齊沒做。

**建議**：在 `monst3r_style_loss` 開頭，對 `gt_extrinsics`／`gt_pts` 額外做一次「相對第一幀」的剛體變換（用 `extrinsics[0]` 的逆，作用在所有幀上），再做尺度正規化。這樣可以同時：
1. 與預訓練慣例一致，加速收斂；
2. 讓 `gt_pose_enc` 的 frame 0 恆為 (T=0, R=I)，camera loss 的監督訊號更乾淨。

### 7.3 Camera loss：多 stage gamma 加權 + T/R/FL 分權重 vs 單一平均

- **原生**：每個 stage 用 `gamma^(n-i-1)` 加權（越後面的 stage 權重越高），且 T/R/FL 各有獨立權重。
- **finetune**：所有 stage 直接平均，T/R/FL 沒有分開權重（全部用同一個 `(pred-gt).abs().mean()`）。

**合理之處**：簡化、少一組超參數要調。

**風險/疑慮**：
- 早期 stage 的 pose_enc 通常精度較低、且 last-N blocks 主要影響的是**後段** stage，給早期 stage 與後期 stage 同樣權重，等於把監督訊號「稀釋」到對 last-N blocks 較不敏感的早期 stage 上，可能降低訓練訊號的針對性。
- T/R/FL 量級本身差異大（平移 vs 四元數 vs FoV），同一個 `.abs().mean()` 直接相加，等於隱含假設三者尺度相近——配合 7.2 的尺度正規化後（T 已除以 scene_scale），量級差距會縮小，**部分緩解**了這個問題，但 FoV（內參）的尺度仍與 T/R 不同。

**建議**：若觀察到 `loss_camera` 中某個分量（例如 FL）長期主導梯度，可考慮把 T/R/FL 拆開記錄（即使不分開加權，至少分開 log），這比重建整套 gamma 加權系統成本低很多。

### 7.4 Depth/Point loss：L1/L2 + conf-weight，但無 gradient/normal loss、無 quantile 過濾

- **合理之處**：少了 `gradient_loss_fn` 與 `valid_range` 兩組超參數，loss 公式更直觀，調參空間更小，對「先把 pipeline 跑起來」階段是合理的簡化。

- **風險/疑慮（本節是除 7.2 外第二個值得關注的點）**：
  - **沒有 quantile 過濾**：原生 `valid_range=0.98` 的設計動機是過濾「長尾大誤差」對 loss 的主導。在**動態場景** finetune 的語境下，「動態物體區域」對一個原本只見過靜態場景的模型來說，**天然就是誤差最大的區域**——也就是 outlier 過濾原本要濾掉的那批點，恰好可能是這次 finetune **最想學的訊號**。
    - 這意味著：是否要做 quantile 過濾，其實是一個**目標選擇問題**，不是單純的「數值穩定性」問題：
      - 不過濾 → 動態物體的大誤差會主導梯度，訓練方向偏向「優先修正動態物體」，符合 finetune 動機，但也可能因為早期梯度過大造成不穩定（尤其疊加 7.2 的座標系問題、第 6 節缺 warmup）。
      - 過濾 → 訓練更穩定，但等於把「動態物體」當 outlier 濾掉，**違背 finetune 的初衷**。
    - 目前 finetune 選擇「不過濾」，**方向上與任務目標一致**，但需要靠 7.2（座標系對齊）與第 6 節（warmup）來補足穩定性，否則「該學的訊號」可能先以「訓練爆炸」的形式表現出來。
  - **沒有 gradient/normal loss**：會降低深度圖邊緣的幾何一致性監督。對動態物體邊界（運動模糊、遮擋邊緣）的學習可能不夠精細，但這是「錦上添花」而非「正確性」問題，可以等基本 pipeline 收斂後再加。
  - **缺少 `check_and_fix_inf_nan`**：真實資料集（尤其 Waymo/Sintel 的稀疏/遠距深度）較容易出現數值極端值，原生在每個 loss 分量都做了 inf/nan 防護。finetune 目前僅靠 `valid = gt_depth > 0` 與最外層 `torch.isfinite(total_loss)` 兜底，**防護層級較薄**。

**建議（優先序）**：
1. 先做 7.2（座標系對齊）——這會同時改善 camera loss 與 point/depth loss 的目標一致性，是「地基」問題。
2. 補 6 節的 warmup，降低「解凍範圍變大 + 不做 outlier 過濾」帶來的早期不穩定風險。
3. 視訓練穩定性再決定是否加回 `check_and_fix_inf_nan`（成本低，可以先加，不影響語義）。
4. gradient/normal loss 與 quantile 過濾留到 baseline 穩定後再實驗性加入（目前專案已有 `scripts/sweep_loss_weights.sh`，可以作為後續加入這些 term 時的調參基礎設施）。

### 7.5 Camera weight：5.0 → 0.5

- **合理之處**：原生 `point` loss 預設關閉（`point: null`），只有 camera(5.0) + depth(1.0) 兩項，camera 權重相對高；finetune 同時啟用 point + depth + camera 三項都是「always-on」，整體 loss 量級提高，camera 權重相應下調是合理的重新平衡。
- **疑慮**：0.5 是否為最佳值取決於 7.2/7.3/7.4 的修正是否到位——目前的權重很可能是在「座標系未對齊」「無 outlier 過濾」的現狀下調出來的經驗值，若後續修正了 7.2，建議重新跑一次 `sweep_loss_weights.sh`。

---

## 總結：問題優先序

1. **(最高)** 7.2 / 第2節 — GT 座標系未轉換到 cam0-relative，與預訓練慣例不一致，影響 camera loss 與 point/depth loss 的目標一致性。
2. **(高)** 第6節 — 缺少 LR warmup，在「解凍範圍變大 + 無分模組 grad clip + 無 outlier 過濾」的組合下放大不穩定風險。
3. **(中)** 第2節 — 分模組 grad norm 監控缺失（不一定要分別 clip，但至少要能看到）。
4. **(中)** 7.4 — outlier 過濾與 `check_and_fix_inf_nan` 的取捨，建議先以監控為主，視訓練穩定性決定是否加回。
5. **(低)** 7.3 / gradient loss / 雙 param group lr — 屬於「調優」層級，待 1~3 穩定後再迭代。

以上 1~3 項彼此有交互作用（座標系不一致 → loss 數值偏移 → 沒有 warmup/grad clip 緩衝 → 更容易在訓練初期出現不穩定），建議按優先序逐項驗證，而不要同時改動多項，否則難以歸因。
