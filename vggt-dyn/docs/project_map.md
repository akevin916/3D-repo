# VGGT Finetune Pipeline — Project Map

## 1. 入口檔案

[finetune/train.py](../finetune/train.py)

執行方式：`python -m finetune.train --output <dir> [--ckpt ... | --resume ...] ...`

## 2. 訓練流程圖

```
train(args)
 ├─ set_seed(args.seed)                                   [model_utils.py]
 ├─ 寫出 config.json / 建立 TensorBoard SummaryWriter
 ├─ _build_model(args, device)                            -> model, trainable_params, resume_state
 │    ├─ VGGT() + load_state_dict (--ckpt 或 --resume)
 │    └─ apply_monst3r_style_freeze(...)                  [model_utils.py]
 ├─ build_loader(args)                                    -> train loader   [data_loader.py]
 ├─ build_val_loader(args)                                -> val loader     [validation.py]
 ├─ _setup_optimizer(...)                                  -> optimizer, scaler, start_epoch, global_step, best_epoch_loss, end_epoch
 ├─ _load_history(args)                                    -> history list (resume 用)
 │
 └─ for epoch in range(start_epoch, end_epoch):
       ├─ _run_epoch(...)            ─ 逐 batch:
       │      images/depths/extrinsics/intrinsics -> device
       │      (可選) ColorJitter
       │      cosine_lr() 更新 lr
       │      model(images) -> preds
       │      monst3r_style_loss(preds, gt...) -> total_loss, info
       │      AMP backward + grad clip + optimizer.step()
       │      寫入 TensorBoard / history
       │
       ├─ _run_validation(...)        每 val_every epoch 呼叫 validate()
       │
       └─ _save_epoch_checkpoints(...) 存 epoch_xxx.pt，若刷新最佳則另存 best.pt

訓練結束後：存 final.pt + 寫出 train_history.json
```

## 3. Dataset 建立位置

- 統一入口：`build_dataset(args)` — [finetune/datasets/__init__.py](../finetune/datasets/__init__.py)
- 各資料集實作：
  - [finetune/datasets/point_odyssey.py](../finetune/datasets/point_odyssey.py)
  - [finetune/datasets/tartanair.py](../finetune/datasets/tartanair.py)
  - [finetune/datasets/waymo.py](../finetune/datasets/waymo.py)
  - [finetune/datasets/spring.py](../finetune/datasets/spring.py)
  - [finetune/datasets/sintel.py](../finetune/datasets/sintel.py)（僅驗證用）
  - 共用工具：[finetune/datasets/common.py](../finetune/datasets/common.py)
- DataLoader 組裝（單一資料集 / mixed-ratio 多資料集採樣）：[finetune/data_loader.py](../finetune/data_loader.py) `build_loader()`
- 驗證用 DataLoader 組裝：[finetune/validation.py](../finetune/validation.py) `build_val_loader()`

## 4. Model 建立位置

[finetune/train.py:57-80](../finetune/train.py) `_build_model()`
- 建立 `VGGT()`（[vggt/models/vggt.py](../vggt/models/vggt.py)）
- 載入權重（`--ckpt` 或 `--resume`）
- 呼叫 `apply_monst3r_style_freeze()`（[finetune/model_utils.py:22-67](../finetune/model_utils.py)）凍結 backbone，僅訓練最後 N 個 frame/global blocks + 三個 head（depth/point/camera）

## 5. Loss 計算位置

[finetune/losses.py](../finetune/losses.py) `monst3r_style_loss()`
- 由 GT depth + extrinsics + intrinsics 反投影出 GT world points
- 計算 conf-weighted point loss、depth loss、camera pose-encoding loss
- 呼叫位置：訓練於 [finetune/train.py:157](../finetune/train.py)（`_run_epoch`），驗證於 [finetune/validation.py:99](../finetune/validation.py)（`validate`）

## 6. Optimizer 建立位置

[finetune/train.py:83-113](../finetune/train.py) `_setup_optimizer()`
- `torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)`
- 同時建立 `torch.cuda.amp.GradScaler`，並處理 `--resume` 時 optimizer/scaler 狀態還原與 epoch/step 接續
- 學習率排程：`cosine_lr()`（[finetune/model_utils.py:70-72](../finetune/model_utils.py)），每個 step 在 `_run_epoch` 中更新

## 7. Checkpoint 保存位置

[finetune/model_utils.py:75-101](../finetune/model_utils.py) `save_checkpoint()`
- 內容：model state_dict、args、optimizer/scaler state、epoch、global_step、best_epoch_loss
- 呼叫位置：
  - 每個 epoch 結束：`epoch_{NNN}.pt`（[finetune/train.py:227-233](../finetune/train.py)，`_save_epoch_checkpoints`）
  - 若刷新最佳（val loss 或 train loss）：`best.pt`（[finetune/train.py:244-250](../finetune/train.py)）
  - 訓練全部結束：`final.pt`（[finetune/train.py:302-308](../finetune/train.py)）
- 另外每個 epoch 結束會把訓練歷史（loss/lr 等）寫入 `train_history.json`

## 8. Evaluation 流程

[finetune/validation.py](../finetune/validation.py)
- `build_val_loader()`：組成驗證集 = PointOdyssey(`--val_split`) + Sintel(`final`，需 `--sintel_root`)，任一不存在則跳過
- `validate()`（每 `--val_every` 個 epoch 由 `_run_validation` 呼叫一次）：
  - 對每個 batch 跑 model forward + `monst3r_style_loss` 取得 loss 系列指標
  - Depth 指標：median-scale 對齊後計算 `abs_rel`、`rmse`、`d1`
  - Pose 指標（需 ≥2 frame）：以 [finetune/pose_metrics.py](../finetune/pose_metrics.py) 的 `pose_metrics()` 計算 Sim(3) 對齊後的 `ate`、`rpe_rot`
  - 回傳各指標平均值，寫入 TensorBoard（`val/*`），並作為 `best.pt` 的比較依據
