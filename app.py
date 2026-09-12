"""Interactive visual interface for the Virtual T Cell research model."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

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
            }

result = st.session_state.result
genes = result["genes"]
pathways = result["pathways"]
multiomics = result["multiomics"]

col1, col2, col3 = st.columns(3)
col1.metric("当前亚型", subtype)
col2.metric("扰动", target, f"强度 {strength:.2f}")
col3.metric("模型覆盖", f"{len(genes):,} 基因", f"{len(pathways)} 条通路")

tab1, tab2, tab3, tab4 = st.tabs(["基因变化", "通路变化", "多组学证据", "结果下载"])
with tab1:
    ranked = genes.assign(abs_delta=genes.delta.abs()).nlargest(top_n, "abs_delta").sort_values("delta")
    st.altair_chart(signed_chart(ranked, "gene", "delta", f"{target} 扰动后变化最大的基因"), width="stretch")
    st.dataframe(ranked.drop(columns="abs_delta"), width="stretch", hide_index=True)
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
