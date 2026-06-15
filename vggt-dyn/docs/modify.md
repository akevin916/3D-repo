# Finetune Pipeline 修改計畫

延續 [diff.md](diff.md) 的差異分析，整理出待實作的修改項目、合理性評估、優先級與相互依賴關係。

---

## 執行順序總覽

```
[bf16]（獨立、零風險，最先做）
   |
[座標轉換 cam0-relative] + [尺度正規化統一 avg_dis]（核心修正，合併做）
   |
[infinite loss: outlier過濾+NaN/Inf防護] + [warmup]（吸收上面改動帶來的早期不穩定）
   |
   +-- 以下為穩定後的精修，無嚴格順序 --
   |
[camera loss細項拆分+gamma+權重] + [valid_frame_mask] + [point_weight/conf_alpha分離]
[per-module gradient clip]（建議在加 gradient loss 之後一起決定分組）
[loss加gradient loss]
[雙 param group lr]

[_apply_batch_repetition] / [accum_steps] / [EMA checkpoint]（隨時可加，視資源）
```

---

## 🔴 高優先（基礎修正層）

### 1. 座標轉換（cam0-relative）

**內容**：在算 loss 前，把 GT 的 `extrinsics`/`world_points` 轉換到「以第 0 幀相機為原點」的座標系（對應原生 `normalize_camera_extrinsics_and_points_batch` 的第一步）。

**為什麼是最高優先**：目前 finetune 的 GT 目標座標系（資料集原始世界座標系）跟 pretrained 模型輸出所代表的座標系（cam0-relative）不一致——這不是「需求/寫法差異」，而是 **loss 的監督目標定義本身跟模型語義不對齊**。模型要先花步數「重新學座標系定義」，等於部分對抗預訓練先驗。是唯一一項「方向部分正確但不完整」的既有修改。

**依賴關係**：第 2、6、7（camera 精修）、9（尺度正規化）都建立在這項之上，**必須先做**。

---

### 2. 尺度正規化統一（avg_dis）

**內容**：跟第 1 項合併實作——轉換到 cam0 座標系後，統一算一個 `avg_scale =`（轉換後 valid world points 到原點的平均距離），套用到 `world_points`/`extrinsics` 平移量/`depths`/`cam_points`，取代現有分離的 `scene_scale`、`depth_scale`。

**合理性**：
- 「用平均距離做跨資料集尺度統一」本身是 DUSt3R/MonST3R 系列的標準作法（avg_dis），合理且已有大量先例驗證。
- 風險：mean 對 depth outlier（遠景/雜訊）較敏感，對「近景動態物體 + 遠景靜態背景」混合場景，`avg_scale` 可能被遠景拉高，壓縮前景訊號——這是方法本身的已知限制，原生跟 finetune 都有，不是這次改動引入的新問題。
- **務必跟第 1 項合併做**：分開做會留下「座標系已轉換但尺度仍分離計算」的中間態，數值意義不乾淨。

---

### 3. infinite loss：從「跳過 step」改成「outlier 過濾 + NaN/Inf 防護」

**內容**：
- 在 `regression_loss` 等位置加入 `filter_by_quantile`（`valid_range≈0.98`，丟棄誤差最大的前 2% 像素，不參與梯度）。
- 加入 `check_and_fix_inf_nan`（把 NaN/Inf 數值替換為 0 並 clamp 到 `[-100,100]`），取代目前「整個 step `continue`」的粗暴跳過。

**為什麼跟第 1/2 項同批**：座標轉換改動會讓訓練初期（模型適應新 GT 分佈期間）更容易出現極端值，現有「整個 step 跳過」策略會浪費大量 step；先建好防護網，能讓第 1/2 項的過渡更平滑，也方便事後判斷異常是「座標轉換的暫態」還是「真的數值問題」。

**對於動態場景的特別考量**：outlier 過濾移除的「outlier」在靜態場景模型眼中是噪聲，但在動態場景 finetune 的目標裡，**動態物體本身可能就是「outlier」**——是否要過濾，本質是「目標選擇問題」。建議：先加上但配合 `min_elements`/`hard_max` 等保守參數，觀察訓練曲線後再決定過濾強度，而非完全照搬原生 `0.98`。

---

### 4. bfloat16

**內容**：`torch.cuda.amp.autocast(enabled=args.amp, dtype=torch.bfloat16)`，取代目前未指定 dtype（預設 float16）+ `GradScaler`。

**合理性**：
- bfloat16 指數位跟 float32 相同，數值範圍大，不易 overflow/underflow，不需要 loss scaling；float16 範圍小，容易在梯度/loss 出現 `inf`/`nan`。
- RTX 5090（Blackwell）原生支援 bf16 tensor core，幾乎零額外成本。
- 留著 `GradScaler` 不會報錯（對 bf16 基本是 no-op），但建議順手關閉（`enabled=False`）保持乾淨。

**為什麼最先做**：一行改動、無依賴、能降低後續改動（1/2/3）引入數值不穩定的風險，本身就是「安全帶」的一部分。

---

### 5. Warmup

**內容**：`cosine_lr` 加上前 `warmup_ratio`（建議 5%）步數的線性 warmup：

```python
def cosine_lr(base_lr, min_lr, step, total_steps, warmup_ratio=0.05):
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        return min_lr + (base_lr - min_lr) * (step / max(1, warmup_steps))
    ...
```

**合理性**：原生 `CompositeParamScheduler` 前 5% 是 linear warmup，finetune 的 `cosine_lr` 完全沒有 warmup（從 step 0 就是 `base_lr`）。低成本、原生已驗證過的設計，補回來風險低。

**為什麼跟 3 同批**：跟 3 一樣是「吸收第 1/2 項早期不穩定」的緩衝機制。

---

## 🟡 中優先（依賴高優先層完成後才有意義的精修）

### 6. Camera loss 細項拆分 + gamma 加權 + 權重調整

**內容**：
- 把 9 維 pose encoding 拆成 T/R/FoV 三部分，各自給 `weight_trans`/`weight_rot`/`weight_focal`（原生預設 1.0/1.0/0.5）。
- 多 stage 輸出加上 `gamma^(n_stages-stage_idx-1)` 時間衰減加權（原生 `gamma=0.6`），而非現在的簡單平均。
- 重新評估 `camera_weight`（目前 0.5）。

**前提**：**必須等第 1 項完成**。T/R/FoV 各自權重、gamma 衰減的「合理數值」是建立在「GT 跟 pred 同座標系、量級一致」的假設上；座標系沒修正前調這些權重，調出來的數字很可能是在「補償座標系錯位」，第 1 項做完後大多需要重調。

**跟現有 open issue 的關聯**：[open_issues.md #2](open_issues.md) 提到 mix_freeze_v2 finetune 的 Sintel pose validation 隨 epoch 惡化，當時提出的方向之一是「提高 camera_weight」——這項改動（連同第 1 項）應該優先於單純調高 `camera_weight` 這個數值本身，因為 #2 觀察到的退化可能部分來自座標系不一致導致 camera loss 訊號本身有偏差，單調權重未必能解決根因。

---

### 7. Camera loss 加 `valid_frame_mask`

**內容**：仿照原生 `valid_frame_mask = point_masks[:,0].sum(dim=[-1,-2]) > 100`，只用「有足夠有效深度點」的 frame 計算 camera loss，避免幾乎無深度資訊的 frame 污染 camera pose 監督訊號。

**合理性**：跟第 1 項**無依賴**，是單純加一個 mask 篩選，不涉及座標系數值計算，**現在就可以獨立加**。建議跟第 6 項一起做（兩者都是 camera loss 的精修），但不必等第 1 項。

---

### 8. Loss 加上 gradient loss（"grad"）

**內容**：仿照原生 `gradient_loss`（對 pred-gt 誤差圖算 x/y 方向一階差分的 L1 損失，鼓勵預測的深度/點圖在局部結構上跟 GT 一致），加進 depth（及可選 point）loss。

**合理性**：補強空間結構一致性，原生已驗證過的輔助項。複雜度中等（需實作 multi-scale wrapper）。

**建議時機**：可與第 1/2 項並行開發（不同程式碼區塊），但**上線順序排在 1/2 之後**——gradient loss 是對「誤差圖空間分佈」做正則，若誤差圖本身因座標系不一致而系統性偏移，這項學到的「平滑性」意義會跑掉。

---

### 9. Point loss 給明確權重 + `conf_alpha` 分離

**內容**：
- 加 `point_weight` 參數（目前隱含為 1），讓 point/depth/camera 三項權重都可獨立調整。
- `conf_alpha` 改為 point 跟 depth 各自一個（目前共用 0.2）。

**合理性**：
- Point loss 目前存在是因為 `VGGT()` 未設 `enable_point=False`（推測非刻意設計，原生預設是關閉的）。既然保留，應該明確給權重，方便後續做「point loss 對動態場景是否有幫助」的 A/B（直接調權重到 0 即可，不用改程式結構）。
- Point（3D 距離）跟 depth（純量深度）的誤差量級本就不同（這也是為何要分別做 `scene_scale`/`depth_scale`），共用一個 `conf_alpha` 隱含「兩者該有多保守」程度一樣的假設，不一定成立。
- 低成本（多幾個 CLI 參數），建議跟第 6 項一起做（都是 loss 權重的整體調整）。

---

### 10. Per-module gradient clip

**內容**：仿照原生 `GradientClipper`，把可訓練參數分組（例如 last-N blocks / depth_head+point_head / camera_head），各組獨立算 grad norm 並 clip，取代目前的全域 `clip_grad_norm_(trainable_params, ...)`。

**合理性**：目前 loss 是「point + depth + camera_weight×camera」多任務合成，不同任務梯度量級差異大。**全域 clip** 在某一任務梯度突刺時，會把 `coef`（縮放係數）拉小，連帶壓低其他任務本來正常的梯度——這是「同一套多任務梯度隔離問題，原生有解、finetune 拿掉了一半」，不只是少了監控。

**建議時機**：**放在第 8 項（加 gradient loss）之後**——loss 組成確定後（point/depth/camera + gradient loss），一次性決定好分組，避免做完 clip 分組後又因新增 loss 項要重新分組。

---

### 11. 雙 param group LR

**內容**：last-N blocks（backbone 末端）跟 3 個 output heads 使用不同的 lr（heads 通常可以用較大 lr 加速適應，backbone 末端用較小 lr 維持穩定）。

**合理性**：finetuning 常見做法，合理但屬於**收斂效率調優，不是修正功能缺陷**。

**建議時機**：放在第 1/2 項完成、訓練穩定後做——座標系改變後各部分的 loss landscape 會變，現在調好的 lr 比例未必適用，過早調整容易做白工。

---

## 🟢 低優先 / 視資源決定

### 12. `_apply_batch_repetition`

**內容**：把每個序列的 frame 順序反轉（`torch.flip(dims=[1])`，反轉的是 sequence 維度，**不是**左右翻轉影像），跟原序列一起 concat 成 2x batch。每個樣本內容不變，效果是「讓 cam0-relative 正規化的參考幀（永遠是 frame 0）多一種選擇」。

**合理性**：低風險、無依賴、隨時可加。但效益邊際（只是多一種參考幀的增強），且會讓**有效 batch size 變成 2 倍**——若 GPU memory 吃緊需把 batch size 減半，等於沒有真的增加多樣性。

**建議**：等高/中優先項做完、訓練穩定且有餘裕時再加。

---

### 13. Gradient accumulation（`accum_steps`）

**內容**：目前 finetune 完全沒有 grad accumulation（原生 default `accum_steps=2`）。

**合理性**：只在記憶體吃緊時才需要——例如第 12 項讓 batch size 減半後，用 accumulation 補回有效 batch size，不犧牲第 12 項效果。否則不必加。

---

### 14. EMA checkpoint

**內容**：維護一份模型權重的指數移動平均，`best.pt`/`final.pt` 存 EMA 版本。

**合理性**：原生程式碼裡沒有此功能，純粹是額外建議。「只訓練 last-N blocks + heads」的 finetuning，單一 checkpoint 對某 step 的梯度噪聲較敏感（尤其第 1/2 項剛上線、訓練初期還在適應新座標系時）。實作成本低（幾行），能讓最終評估結果更穩定。**錦上添花，排最後**。

---

## 不建議做的

- **Track loss / track_head**：兩邊都關閉，VGGT TrackHead 需額外的 query_points 輸入跟對應 GT，跟現有資料管線不相容，引入成本遠大於收益（也呼應 [open_issues.md #3](open_issues.md) 的構想——若要做 dense track，工作量本身就是獨立題目，不屬於本次 loss/optimizer 修改範疇）。
- **恢復原生整套 Hydra/`OptimizerWrapper`/`ComposedDataset`/`DynamicDistributedSampler` 架構**：規模對等的取捨，finetune 現有的單一 param group（+第 11 項雙 param group）、`ConcatDataset`+`WeightedRandomSampler` 已足夠，整套搬回來的成本遠大於收益。
