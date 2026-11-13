"""Installed-package entry for one organ; no dependency on the source repository."""
from __future__ import annotations

import argparse
import csv
import html
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parent


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def runtime_report(organ: str) -> dict:
    import virtual_t_cell
    from virtual_t_cell.learning.world_model import BayesianWorldModel
    return {"organ": organ, "python": platform.python_version(), "platform": platform.platform(),
            "python_executable": sys.executable, "package_file": str(Path(virtual_t_cell.__file__).resolve()),
            "dependencies": {name: version(name) for name in ("numpy", "pandas")},
            "engine_available": BayesianWorldModel.__name__, "gpu_required": False, "api_key_required": False}


def write_html(output: Path, organ: str, report: dict) -> None:
    name = "肝脏" if organ == "liver" else "脾脏"
    encoded = html.escape(json.dumps(report, ensure_ascii=False, indent=2))
    metrics = report["test_evaluation"]["overall"]
    first = html.escape(report["first_learning_cycle"]["status"])
    repeated = html.escape(report["repeat_learning_cycle"]["status"])
    decision = html.escape(report["plan"]["decision"])
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{name}虚拟 T 细胞 · 运行结果</title>
<style>body{{margin:0;background:#f5f7f9;color:#173042;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1000px;margin:40px auto;padding:0 24px}}h1{{line-height:1.2}}.notice{{padding:18px 22px;border-left:5px solid #b46b00;background:#fff4dd;border-radius:8px}}.card{{background:white;border:1px solid #dfe7eb;padding:24px;border-radius:12px;margin:24px 0}}a{{color:#006b75}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}small{{color:#576b78}}</style>
<main><small>VIRTUAL T CELL · {organ.upper()} · RESEARCH STARTER</small>
<h1>{name}版已完成一次运行</h1>
<div class="notice"><strong>SYNTHETIC / 合成演示数据</strong><br>此结果用于验证数据、训练、评估、规划与更新流程。它不证明真实{name}预测准确性、临床有效性或 AGI。</div>
<section class="card"><h2>本次流程检查</h2><p>组织：<strong>{organ}</strong>　合成细胞：<strong>{report['atlas']['n_cells']}</strong>　批次：<strong>{report['atlas']['n_batches']}</strong></p><p>首次模型更新：<strong>{first}</strong>　重复循环：<strong>{repeated}</strong>　实验规划：<strong>{decision}</strong></p><p>合成数值留出测试 RMSE：<strong>{metrics['rmse']:.6f}</strong>；训练均值基线：{metrics['mean_baseline_rmse']:.6f}。这些数值仅用于验证已知的合成规律。</p></section>
<section class="card"><h2>结果文件</h2><p><a href="organ_report.json">完整运行报告</a> · <a href="test_predictions.csv">留出预测表</a> · <a href="candidate_rankings.csv">实验候选排序</a> · <a href="atlas_belief.json">组织状态统计</a> · <a href="runtime.json">运行环境</a></p><p>首次运行和重复运行都保留证据身份；模型更新由独立验证门槛决定。规划返回 stop 也属于正常结果。</p></section>
<section class="card"><details><summary>查看完整机器可读输出</summary><pre>{encoded}</pre></details></section></main></html>'''
    (output / "report.html").write_text(body, encoding="utf-8")


def check_metadata(path: Path, organ: str) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "organ" not in (reader.fieldnames or []):
            raise ValueError("metadata needs an organ column")
        labels = {str(row.get("organ", "")).strip() for row in reader}
    if labels != {organ}:
        raise ValueError(f"This package expects only organ={organ}; metadata contains {sorted(labels)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--organ", choices=("liver", "spleen"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("command", choices=("demo", "check", "ingest", "cli"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = runtime_report(args.organ)
    write_json(output / "runtime.json", runtime)
    if args.command == "check":
        print(json.dumps(runtime, ensure_ascii=False, indent=2))
        return 0
    if args.command == "demo":
        from virtual_t_cell.learning.organ_demo import run_organ_demo
        report = run_organ_demo(output, args.organ)
        write_html(output, args.organ, report)
        print(json.dumps({"status": "completed", "organ": args.organ, "evidence_type": "synthetic",
                          "report": str(output / "report.html"), "json": str(output / "organ_report.json")}, indent=2))
        return 0
    if args.command == "cli":
        from virtual_t_cell.learning.cli import main as learning_main
        return learning_main(args.arguments)
    ingest = argparse.ArgumentParser(description="Ingest real observational data for this package's organ.")
    group = ingest.add_mutually_exclusive_group(required=True)
    group.add_argument("--counts")
    group.add_argument("--h5ad")
    ingest.add_argument("--metadata")
    ingest.add_argument("--counts-layer")
    ingest.add_argument("--batch-id", required=True)
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--evidence-type", choices=("observational", "synthetic"), default="observational")
    data = ingest.parse_args(args.arguments)
    if data.counts:
        if not data.metadata:
            ingest.error("CSV counts require --metadata")
        check_metadata(Path(data.metadata), args.organ)
    else:
        if not data.counts_layer:
            ingest.error("H5AD requires --counts-layer and the launcher --h5ad dependency option")
        import anndata
        adata = anndata.read_h5ad(data.h5ad, backed="r")
        try:
            if "organ" not in adata.obs or set(adata.obs["organ"].astype(str).str.strip()) != {args.organ}:
                raise ValueError(f"H5AD obs must contain only organ={args.organ}")
        finally:
            adata.file.close()
    from virtual_t_cell.learning.atlas import AtlasStore
    from virtual_t_cell.learning.schema import write_jsonl
    from virtual_t_cell.learning.state import summarize_atlas_states
    with AtlasStore(output / "real_atlas.sqlite") as atlas:
        existing_organs = set(atlas.summary()["organs"])
        if existing_organs - {args.organ}:
            raise ValueError("Output atlas already contains a different organ; choose a separate output directory")
        keywords = {"batch_id": data.batch_id, "source": data.source, "evidence_type": data.evidence_type}
        if data.counts:
            report = atlas.ingest_counts(data.counts, data.metadata, **keywords)
        else:
            report = atlas.ingest_h5ad(data.h5ad, counts_layer=data.counts_layer, **keywords)
        states = atlas.pseudobulk_records()
        write_jsonl(output / "atlas_states.jsonl", states)
        write_json(output / "atlas_belief.json", summarize_atlas_states(states))
        report["atlas"] = atlas.summary()
    write_json(output / "ingestion_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, ImportError) as error:
        print(f"Run failed: {error}", file=sys.stderr)
        raise SystemExit(2)
