# 实验记录

规矩：每次运行都必须新增一行，哪怕结果很差。不许事后补记。

| 日期 | 编号 | 模型 | 类别 | shot | seed | image AUROC | pixel AUROC | 配置改动 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-10 | patchcore-bottle-inf-42 | PatchCore | bottle | inf | 42 | 1.0000 | 0.9856 | batch=8, workers=2 | 可用，与原论文一致（100.0 / 98.6） |

## 字段说明

- **编号**：`<模型>-<类别>-<shot>-<seed>`，例如 `patchcore-bottle-4-42`
- **shot**：训练用的正常样本张数，全量填 `inf`
- **配置改动**：只写和默认配置的差异，没改就写 `-`
- **结论**：一句话，写“可用 / 异常 / 待查”，不要写感受

## 踩坑记录

报错原文 + 解决办法，按时间倒序追加。同一个坑不踩第二次。

### 2026-11-10 `RuntimeError: bad allocation`

现象：训练第一个 batch 时崩在 `conv2d`，报 `bad allocation`。
原因：不是显存，是主机内存告急（16G 内存只剩 2.4G 可用），默认 `batch=32` + 4 个 dataloader 进程把它撑爆了。
解决：`batch` 降到 8、`workers` 降到 2，跑之前关掉浏览器腾内存。

### 2026-11-10 MVTec AD 下载 404

现象：`urllib.error.HTTPError: HTTP Error 404`，anomalib 内置的 mydrive.ch 链接已失效。
解决：用 `prepare_data.py` 从 HuggingFace 镜像（`TheoM55/mvtec_anomaly_detection`）拉取并重排成 anomalib 的目录结构。
