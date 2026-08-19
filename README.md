# Virtual T Cell

第三个独立算法平台：基于公开 T 细胞 Perturb-seq 数据预测单基因敲低/敲除或多靶点抑制后的转录组与信号通路变化。

本仓库与“天然产物筛选”和“PD-L1 环肽”平台完全独立，拥有自己的源码、模型、依赖、测试和 CI。

## Developer

**Di Zhu**  
[Google Scholar](https://scholar.google.com/citations?user=HSui3U8AAAAJ&hl=en)

## 当前能力

- 单基因 KO 转录响应预测；
- stimulated / unstimulated 两种 TCR 条件；
- 多靶点及不同抑制强度的药物近似；
- TCR、NFAT、NF-κB、AP-1/MAPK、JAK-STAT、激活、细胞毒、耗竭、增殖和凋亡通路评分；
- 基因级不确定性；
- 对少量未观测靶点使用显式标记的网络邻居外推。
- TCRα/β CDR3、V/J、HLA、表位和抗原证据匹配；
- 10x Cell Ranger VDJ `filtered_contig_annotations.csv` 的克隆型、αβ 配对率和多样性分析。

## TCR 测序预测

仓库内置 VDJdb 2026-06-03 的人类 TCR 紧凑证据库，包含 180,895 条记录、163,598 个唯一 CDR3、60,404 条 α 链、120,491 条 β 链和 1,946 个表位。查询配对 α/β 链：

```powershell
virtual-t-cell predict-tcr `
  --database models\vdjdb_2026_06_tcr_evidence.npz `
  --cdr3-alpha CAAAAAGNEKLTF `
  --cdr3-beta CASSEGTGELFF `
  --max-distance 1 `
  --out run_output\tcr_specificity.csv
```

`edit_distance=0` 表示数据库精确证据；距离大于 0 的结果仅表示序列近邻假设。α、β 链都支持同一表位时会显示更高的配对链证据等级。不能把序列近似等同于确定的抗原结合。

分析自己的 10x VDJ 文件：

```powershell
virtual-t-cell analyze-tcr `
  --contigs filtered_contig_annotations.csv `
  --out-dir run_output\repertoire
```

候选数据源及整合状态见 `data_sources/tcr_sources.json`。第三方数据许可与引用要求见 `THIRD_PARTY_DATA.md`。

## v0.2 原代 CD4⁺ T 细胞数据升级

新增 [GSE314342 / Primary Human CD4+ T Cell Perturb-seq](https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq) 构建管线。公开数据包含约 2,200 万个原代人 CD4⁺ T 细胞、4 位供体、3 个刺激状态，以及全基因组 CRISPRi 扰动。差异表达对象包含 33,983 个“扰动 × 状态”结果和 10,282 个表达读出基因。

平台发布的轻量模型默认保留：

- 512 个跨 Rest、Stim8hr、Stim48hr 都有结果的高置信扰动基因（高于最低 200 个要求）；
- 2,048 个高变异响应基因，并强制纳入 TCR、NFAT、NF-κB、AP-1 和耗竭标志基因；
- log2 fold-change 效应、标准误、每个扰动的细胞数和三种状态基线；
- 严格排除低表达、单 guide、邻近基因共同敲低和远端脱靶结果。

仓库不直接存放数 GB 的单细胞原始矩阵；构建工作流从官方公开 S3 数据源下载。已经通过下载、构建和真实预测冒烟测试的紧凑模型存放在 `models/gse314342_primary_cd4_virtual_t_cell.npz`，同时也作为 GitHub Actions artifact 发布。

## v0.4 主数据集：GSE278572

当前主模型改为 [GSE278572](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278572) 原代人 CD4⁺ Treg/Teff Perturb-CITE-seq。模型优先使用作者在 [Zenodo 13924126](https://zenodo.org/records/13924126) 发布的 S8 激活评分、S9 伪批量差异表达和 S14 细胞元数据；以 [Zenodo 5784651](https://zenodo.org/records/5784651) 的 CRISPRa/CRISPRi IL2、IFNG 全基因组筛选作为独立表型校准；对于主数据未覆盖的靶点，回退到 GSE314342 全基因组 CD4⁺ T 细胞模型。

四个状态为 `Teff_Resting`、`Teff_Stimulated`、`Treg_Resting`、`Treg_Stimulated`。预测除基因与通路文件外，还输出 `phenotype_predictions.csv`，记录 Zenodo 5784651 支持的 IL2/IFNG CRISPRa/i 表型及 FDR。

v0.5 同时整合两个 GEO 数据集：GSE278572 作为原代 Treg/Teff 主效应层，GSE92872 作为 Jurkat TCR 刺激的独立跨数据验证层。对于两套数据共有的干预靶点，预测会额外生成 `cross_dataset_validation.csv`，报告所有共有响应基因和 Top-200 响应基因的 Pearson 方向一致性。由于细胞来源不同，两者不会被直接平均。

原始数据可通过 GitHub Actions 的 **Download and validate GEO sources** 工作流复现下载。工作流下载 GSE92872 的两个表达矩阵，以及 GSE278572 的 barcodes、features、3.6 GB Matrix Market 矩阵和 protospacer calls；随后生成包含文件大小、SHA-256、矩阵维度和行数的 `geo_download_manifest.json`。大型原始数据不纳入 Git 历史。

## 快速运行

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
virtual-t-cell predict `
  --model models\gse278572_primary_context_virtual_t_cell.npz `
  --condition Teff_Stimulated `
  --perturb LCK `
  --out-dir run_output\LCK_KO
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install -e .
virtual-t-cell predict \
  --model models/gse278572_primary_context_virtual_t_cell.npz \
  --condition Teff_Stimulated \
  --perturb LCK \
  --out-dir run_output/LCK_KO
```

输出：

- `gene_predictions.csv`：基线表达、预测表达、变化量和标准误；
- `pathway_predictions.csv`：10 类 T 细胞功能/信号通路变化；
- `prediction_metadata.json`：条件、靶点、强度和外推模式。

## 多靶点药物近似

例如 LCK 抑制 80%，PTPN11 抑制 30%：

```powershell
virtual-t-cell predict `
  --model models\gse92872_virtual_t_cell.npz `
  --condition stimulated `
  --perturb LCK:0.8 PTPN11:0.3 `
  --out-dir run_output\drug_like
```

当前采用加性扰动近似，尚未学习真实剂量、时间和非线性交互。

## 从 GSE92872 重新训练

从 [NCBI GEO GSE92872](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92872) 下载：

```text
GSE92872_CROP-seq_Jurkat_TCR.digital_expression.csv.gz
```

然后运行：

```powershell
virtual-t-cell prepare --expression data\GSE92872_CROP-seq_Jurkat_TCR.digital_expression.csv.gz --out artifacts\gse92872_prepared.npz --genes 2000
virtual-t-cell train --prepared artifacts\gse92872_prepared.npz --out models\gse92872_virtual_t_cell.npz
```

GSE137554 的注释和 10x HDF5 可从 [NCBI GEO GSE137554](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137554) 下载。解析旧版 HDF5 需要：

```bash
python -m pip install -e ".[hdf5]"
```

## 从 GSE314342 重新构建

下载官方 `GWCD4i.DE_stats.h5ad` 后运行：

```powershell
python -m pip install -e ".[atlas]"
virtual-t-cell prepare-gse314342 `
  --de-h5ad data\GWCD4i.DE_stats.h5ad `
  --out models\gse314342_primary_cd4_virtual_t_cell.npz `
  --targets 512 `
  --genes 2048
```

也可以在 GitHub 的 **Actions → Build primary CD4 T-cell atlas model → Run workflow** 复现下载、构建和冒烟测试。

## 验证结果与边界

旧版捆绑模型使用 GSE92872 的 5,905 个 Jurkat 细胞训练，覆盖 32 个非对照靶点和两个刺激条件。独立 sgRNA 配对验证的全基因中位 Pearson 约为 `0.040`，说明旧数据噪声较强。v0.2 构建管线改用原代 CD4⁺ T 细胞，并保留论文提供的 guide/供体一致性、脱靶和标准误质控字段。

因此：

- 当前版本是机制探索和工程原型；
- 不能作为临床疗效预测器；
- Jurkat 不能替代原代 CD4、CD8 或 Treg；
- 药物模拟是靶点效应加性近似；
- 结果必须经过 Perturb-seq、流式、细胞因子和功能实验验证。

完整验证指标位于 `models/gse92872_validation_metrics.json`。

## 测试

```bash
python -m unittest discover -s tests -v
```

GitHub Actions 会在 Python 3.10、3.11 和 3.12 上执行安装、语法检查和端到端预测测试。

## 许可证

MIT。公开数据仍受其原始数据库和论文的数据使用条款约束。
