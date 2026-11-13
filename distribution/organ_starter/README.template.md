# {{ORGAN_NAME}}虚拟 T 细胞持续学习研究版

**器官：`{{ORGAN}}`。默认示例全部为 SYNTHETIC / 合成软件测试。**

这个压缩包独立包含源码、固定依赖、该器官的示例数据、启动脚本和测试。
不需要另一个器官的压缩包、GitHub 账号、API key、GPU 或外部数据库。
当前为可运行研究核心；尚未发布真实{{ORGAN_NAME}}数据训练完成的基础模型，也没有证明 AGI。

## 1. 最快启动

先将整个 ZIP 解压至可写目录。请使用 **64 位 CPython 3.12**。
首次安装需要联网下载公开 Python 依赖；计算和数据处理在本机完成。

### Windows

安装 Python 3.12 后，双击 `run_demo.bat`，或者在终端运行：

```powershell
py -3.12 run.py
```

### macOS / Linux

安装 Python 3.12 后，在解压目录运行：

```bash
bash run_demo.sh
```

也可以运行 `python3.12 run.py`。macOS 可使用随包的 `run_demo.command`；
若系统限制双击启动，使用上述终端命令即可。

启动脚本自动创建本包自己的 `.venv`，安装固定依赖和完整源码，再执行该器官的流程。
不会借用系统中其他项目的 NumPy、pandas 或已安装的 Virtual-T-Cell。
首次安装时间取决于网络；安装完成后的演示使用普通 CPU。

## 2. 在哪里查看结果

运行成功后打开：

**`output/demo/report.html`**

同时生成：

| 文件 | 内容 |
|---|---|
| `organ_report.json` | 完整状态、模型更新、测试和规划记录 |
| `test_predictions.csv` | 独立留出集的预测、真实合成值与误差 |
| `candidate_rankings.csv` | 候选实验的信息价值与排序 |
| `atlas_belief.json` | 该组织的供者层面描述性状态统计 |
| `active_model.npz` | 本次合成示例实际拟合的模型；不是预训练生物学模型 |
| `active_model.npz.belief.json` | 与该模型版本匹配的信念状态 |
| `learning/` | 候选模型、升级门槛、历史版本与证据记录 |
| `runtime.json` | 实际解释器、安装路径和依赖版本 |

默认使用两个批次共 16 个合成细胞，以及独立拆分的合成响应记录。
演示的数值机制已知且为线性，因此使用 Bayesian linear regression 来检查软件流程。
完整引擎也支持随机非线性特征，但这些合成结果不能作为真实器官预测准确率。

再次运行不会重复增加同一批细胞，也不会把已升级模型重置为初始模型。
首次安装后可完全跳过安装与网络访问：

```bash
python3.12 run.py --no-install
```

Windows 对应 `py -3.12 run.py --no-install`。
规划结果为 `stop` 表示继续测量的预期决策收益不足，是允许的正常输出。

## 3. 接入自己的{{ORGAN_NAME}}数据

推荐先使用原始计数 CSV 与细胞元数据 CSV：

```bash
python3.12 run.py ingest --counts /path/counts.csv --metadata /path/metadata.csv --batch-id batch_001 --source my_stable_study
```

Windows 使用 `py -3.12` 并替换为实际文件路径。路径含空格时加引号。
计数宽表第一列是 `cell_id`，其余是基因的非负整数 raw counts。
metadata 需要 `cell_id/sample_id/donor_id/organ/cell_type/condition`；
建议明确 `species`，有真实实验时间时提供 `biological_time_hours`。
**本包要求所有 metadata 的 `organ` 均为 `{{ORGAN}}`。**
不同供者、样本、处理和时间不能通过改名混为独立新证据。

真实输入默认写入 `output/ingest/real_atlas.sqlite`，与合成演示的数据库分开。
观察图谱更新状态统计；训练干预响应需要另外提供有来源、匹配对照和单位的响应记录。
宽松的默认 QC 是结构检查，不能代替研究方案中的双细胞、背景 RNA 或注释质控。

### H5AD（可选）

先准备可选依赖：

```bash
python3.12 run.py --h5ad check
```

再导入，必须明确原始计数层：

```bash
python3.12 run.py ingest --h5ad /path/batch.h5ad --counts-layer counts --batch-id batch_002 --source my_stable_study
```

`obs` 的元数据要求与 CSV 相同。本工具不默认认为 `.X` 就是 raw counts。
不同 gene panel 分开统计；CP10K 分母是输入 panel 的总 counts。

## 4. 训练、持续更新与完整命令

随包源码提供全部学习命令，可用以下入口调用：

```bash
python3.12 run.py cli -- --help
python3.12 run.py cli -- train --records responses.jsonl --out-model output/real_model.npz
```

更完整的数据 schema、独立验证、回放和 `sync-inbox` 配置见 `docs/learning/README.md`。
响应训练必须使用来源清楚的实验或实验汇总；普通图谱、模型预测和重叠状态快照不会成为干预标签。
新模型只有通过整体、组织及输出退化检查和物种验证覆盖检查后才会升级。

示例输入还以只读参考的形式放在 `examples/`，可用于理解格式。
`output/demo/inputs/` 是每次演示使用的固定合成 fixture，请不要在里面放真实数据。

## 5. 自检与复现

```bash
python3.12 run.py check
```

`runtime.json` 中的 `package_file` 应位于本包自己的 `.venv` 下。
固定依赖见 `requirements.txt`，H5AD 依赖见 `requirements-h5ad.txt`。
`PACKAGE_MANIFEST.json` 记录源码与随包文件校验值。

包内源码测试（准备环境后）：

- macOS/Linux：`.venv/bin/python -m unittest discover -s tests`
- Windows：`.venv\Scripts\python.exe -m unittest discover -s tests`

## 6. 常见问题

- **提示 Python 版本不符：** 本包锁定 3.12。不要把最新固定依赖装入 Python 3.10 或 3.11。
- **首次安装网络失败：** 在能够连接公开 Python 包索引的网络下重试，已有输出不会被删除。
- **离线实验室：** 可预先准备对应系统和 Python 3.12 的 wheels，使用 `--offline-wheels /path/wheels`；ZIP 本身未内置各操作系统的第三方二进制依赖。
- **没有写权限：** 将整个包解压到自己可写的目录，不要直接在 ZIP 预览窗口运行。
- **更换电脑：** 复制原 ZIP 重新解压；不要复制另一台电脑生成的 `.venv`。
- **数据冲突：** 已有证据身份对应的内容发生改变会被拒绝；用清楚的版本和来源重新整理，而非通过换 ID 重复计数。

## 7. 项目与边界

源码项目：https://github.com/zhu-algorithm/Virtual-T-Cell

当前开发 PR：https://github.com/zhu-algorithm/Virtual-T-Cell/pull/4

同一个学习核心由本包配置为 `{{ORGAN}}`，默认运行不读取其他器官包。
没有将血液 CD4 参考权重改名为{{ORGAN_NAME}}模型；本包不附带那些历史大模型资产。
真实细胞注释、批次校正、组织微环境、动态轨迹和前瞻性生物验证仍需继续研究。
部署到服务器、接通测序设备或开展长期后台运行，需要实际配置，启动演示不会自动完成这些动作。
