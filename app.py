"""Interactive visual interface for the Virtual T Cell research model."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from virtual_t_cell.cli import predict


ROOT = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    ROOT / "models" / "tcell_expanded_subtypes.npz",
    ROOT / "work" / "expanded_subtypes" / "tcell_expanded_subtypes.npz",
    ROOT / "models" / "tcell_subtype_multiomics.npz",
]


@st.cache_resource
def load_model():
    path = next((item for item in MODEL_CANDIDATES if item.exists()), None)
    if path is None:
        raise FileNotFoundError("No subtype model was found. Run the expanded-model workflow first.")
    return path, np.load(path, allow_pickle=False)


def signed_chart(frame: pd.DataFrame, label: str, value: str, title: str):
    shown = frame.copy()
    shown["direction"] = np.where(shown[value] >= 0, "上调", "下调")
    return alt.Chart(shown).mark_bar().encode(
        x=alt.X(f"{value}:Q", title="预测变化"),
        y=alt.Y(f"{label}:N", sort="-x", title=None),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["上调", "下调"], range=["#d95f4b", "#3977b6"]), title=None),
        tooltip=[label, alt.Tooltip(value, format=".4f"), "direction"],
    ).properties(title=title, height=max(260, min(620, len(shown) * 25)))


def virtual_cell_svg(pathways: pd.DataFrame, target: str, subtype: str) -> str:
    values = dict(zip(pathways.pathway, pathways.delta_score.fillna(0)))
    positions = {
        "TCR_SIGNALING": (400, 70), "JAK_STAT": (670, 165),
        "NFAT": (210, 220), "NFKB": (400, 210), "AP1_MAPK": (575, 260),
        "ACTIVATION": (305, 385), "PROLIFERATION": (490, 390),
        "CYTOTOXICITY": (165, 455), "EXHAUSTION": (640, 455), "APOPTOSIS": (400, 500),
    }
    edges = [("TCR_SIGNALING", "NFAT"), ("TCR_SIGNALING", "NFKB"),
             ("TCR_SIGNALING", "AP1_MAPK"), ("JAK_STAT", "ACTIVATION"),
             ("NFAT", "ACTIVATION"), ("NFKB", "PROLIFERATION"),
             ("AP1_MAPK", "PROLIFERATION"), ("ACTIVATION", "CYTOTOXICITY"),
             ("PROLIFERATION", "EXHAUSTION"), ("EXHAUSTION", "APOPTOSIS")]
    finite = [abs(float(v)) for v in values.values() if np.isfinite(v)]
    scale = max(finite, default=1.0) or 1.0
    def color(value):
        intensity = min(abs(float(value)) / scale, 1.0)
        if value >= 0:
            return f"rgba(210,67,55,{0.25 + 0.70 * intensity:.2f})"
        return f"rgba(45,108,176,{0.25 + 0.70 * intensity:.2f})"
    lines = "".join(
        f'<line x1="{positions[a][0]}" y1="{positions[a][1]}" x2="{positions[b][0]}" y2="{positions[b][1]}" />'
        for a, b in edges
    )
    nodes = "".join(
        f'<g><circle cx="{x}" cy="{y}" r="54" fill="{color(values.get(name, 0))}" />'
        f'<text x="{x}" y="{y-4}" text-anchor="middle">{name.replace("_", " ")}</text>'
        f'<text class="score" x="{x}" y="{y+18}" text-anchor="middle">{values.get(name, 0):+.4f}</text></g>'
        for name, (x, y) in positions.items()
    )
    return f"""
    <div class="vtc-cell-wrap">
      <svg viewBox="0 0 800 580" role="img" aria-label="{subtype} virtual cell pathway response after {target} knockout">
        <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#718096"/></marker></defs>
        <ellipse cx="400" cy="300" rx="365" ry="260" fill="#f7fafc" stroke="#8da2b5" stroke-width="3"/>
        <ellipse cx="400" cy="405" rx="165" ry="110" fill="#edf2f7" stroke="#9aa9b8" stroke-width="2"/>
        <text class="region" x="400" y="558" text-anchor="middle">{subtype} · {target} knockout</text>
        <text class="region" x="400" y="330" text-anchor="middle">NUCLEUS</text>
        <g class="links">{lines}</g>{nodes}
      </svg>
      <div class="vtc-legend"><span><i class="up"></i>上调/增强</span><span><i class="down"></i>下调/抑制</span><span>圆内数值：通路 Δ score</span></div>
    </div>
    <style>
      .vtc-cell-wrap{{font-family:Arial,sans-serif;color:#263746}}
      .vtc-cell-wrap svg{{width:100%;height:auto;max-height:610px}}
      .vtc-cell-wrap .links line{{stroke:#718096;stroke-width:2;opacity:.55;marker-end:url(#arrow)}}
      .vtc-cell-wrap text{{font-size:13px;font-weight:600;fill:#1f2937}}
      .vtc-cell-wrap text.score{{font-size:12px;font-weight:400}}
      .vtc-cell-wrap text.region{{font-size:14px;letter-spacing:1px;fill:#52606d}}
      .vtc-legend{{display:flex;gap:22px;justify-content:center;flex-wrap:wrap;font-size:13px}}
      .vtc-legend i{{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px}}
      .vtc-legend .up{{background:#d24337}} .vtc-legend .down{{background:#2d6cb0}}
    </style>"""


st.set_page_config(page_title="Virtual T Cell", page_icon="🧬", layout="wide")
st.title("Virtual T Cell · 虚拟 T 细胞")
st.caption("选择 T 细胞亚型和扰动基因，查看转录组、信号通路及多组学证据变化。仅用于科研假设生成。")

try:
    model_path, model = load_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

conditions = model["conditions"].astype(str).tolist()
subtypes = model["subtypes"].astype(str).tolist()
targets = model["targets"].astype(str).tolist()

with st.sidebar:
    st.header("预测设置")
    subtype = st.selectbox("T 细胞亚型", subtypes, index=subtypes.index("Th17") if "Th17" in subtypes else 0)
    condition = st.selectbox("实验状态", conditions, index=conditions.index("Teff_Stimulated") if "Teff_Stimulated" in conditions else 0)
    target = st.selectbox("敲除/抑制基因", targets, index=targets.index("ZAP70") if "ZAP70" in targets else 0)
    strength = st.slider("扰动强度", 0.05, 1.00, 1.00, 0.05,
                         help="1.0 表示完整模型效应；较小数值用于近似部分药物抑制。")
    top_n = st.slider("显示前 N 项", 10, 50, 20, 5)
    run = st.button("运行预测", type="primary", width="stretch")
    st.caption(f"模型：{model_path.name} · {len(subtypes)} 亚型 · {len(model['genes']):,} 响应基因")

if run or "result" not in st.session_state:
    with st.spinner("正在计算虚拟细胞响应……"):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            predict(model_path, condition, [(target, strength)], output, subtype)
            st.session_state.result = {
                "genes": pd.read_csv(output / "gene_predictions.csv"),
                "pathways": pd.read_csv(output / "pathway_predictions.csv"),
                "multiomics": pd.read_csv(output / "multiomic_predictions.csv") if (output / "multiomic_predictions.csv").exists() else pd.DataFrame(),
                "metadata": json.loads((output / "prediction_metadata.json").read_text(encoding="utf-8")),
                "config": {"subtype": subtype, "condition": condition, "target": target, "strength": strength},
            }

result = st.session_state.result
genes = result["genes"]
pathways = result["pathways"]
multiomics = result["multiomics"]
config = result["config"]

col1, col2, col3 = st.columns(3)
col1.metric("当前亚型", config["subtype"])
col2.metric("扰动", config["target"], f"强度 {config['strength']:.2f}")
col3.metric("模型覆盖", f"{len(genes):,} 基因", f"{len(pathways)} 条通路")

tab0, tab1, tab2, tab3, tab4 = st.tabs(["虚拟细胞", "单基因敲除结果", "通路变化", "多组学证据", "结果下载"])
with tab0:
    components.html(virtual_cell_svg(pathways, config["target"], config["subtype"]), height=650, scrolling=False)
with tab1:
    up = genes.nlargest(top_n, "delta")
    down = genes.nsmallest(top_n, "delta")
    target_row = genes[genes.gene.astype(str).str.upper().eq(config["target"].upper())]
    if not target_row.empty:
        row = target_row.iloc[0]
        st.info(f"目标基因 {config['target']}：预测 Δ = {row.delta:+.4f}，预测表达 = {row.predicted_log1p_cp10k:.4f}，不确定性 SE = {row.uncertainty_se:.4f}")
    left, right = st.columns(2)
    with left:
        st.altair_chart(signed_chart(up.sort_values("delta"), "gene", "delta", "最强上调基因"), width="stretch")
    with right:
        st.altair_chart(signed_chart(down.sort_values("delta"), "gene", "delta", "最强下调基因"), width="stretch")
    st.subheader("完整基因预测")
    st.dataframe(genes, width="stretch", hide_index=True)
with tab2:
    pathway_view = pathways.dropna(subset=["delta_score"]).sort_values("delta_score")
    st.altair_chart(signed_chart(pathway_view, "pathway", "delta_score", "信号通路预测变化"), width="stretch")
    st.dataframe(pathway_view, width="stretch", hide_index=True)
with tab3:
    if multiomics.empty:
        st.info("当前模型不包含多组学证据层。")
    else:
        evidence = multiomics.assign(abs_delta=multiomics.integrated_multiomic_delta.abs()).nlargest(top_n, "abs_delta")
        st.altair_chart(signed_chart(evidence.sort_values("integrated_multiomic_delta"), "gene", "integrated_multiomic_delta", "多组学整合变化"), width="stretch")
        st.dataframe(evidence.drop(columns="abs_delta"), width="stretch", hide_index=True)
with tab4:
    st.download_button("下载基因预测 CSV", genes.to_csv(index=False).encode("utf-8-sig"), "gene_predictions.csv", "text/csv")
    st.download_button("下载通路预测 CSV", pathways.to_csv(index=False).encode("utf-8-sig"), "pathway_predictions.csv", "text/csv")
    if not multiomics.empty:
        st.download_button("下载多组学预测 CSV", multiomics.to_csv(index=False).encode("utf-8-sig"), "multiomic_predictions.csv", "text/csv")
    st.json(result["metadata"])

st.warning("预测代表跨队列计算假设，不是临床疗效结论；亚型基线来自分选细胞数据，扰动效应并非每个亚型的直接实验测量。")
