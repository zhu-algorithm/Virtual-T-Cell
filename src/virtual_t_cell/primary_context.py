"""Build the GSE278572-led multimodal primary T-cell model."""
from __future__ import annotations
import json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

CONDITIONS = ["Teff_Resting", "Teff_Stimulated", "Treg_Resting", "Treg_Stimulated"]

def _member(archive, suffix):
    matches = [n for n in archive.namelist() if n.endswith(suffix) and "/._" not in n]
    if len(matches) != 1: raise ValueError(f"Expected one {suffix}, found {matches}")
    return matches[0]

def build_primary_context_model(data_tables_zip: Path, screens_zip: Path, fallback_model: Path, out: Path) -> dict:
    with zipfile.ZipFile(data_tables_zip) as z:
        with z.open(_member(z, "S9_pseudobulk_rnaseq_diff_expressed_regulators.xlsx")) as f: de = pd.read_excel(f)
        with z.open(_member(z, "S8_perturb_seq_activation_scoring_summary_table.xlsx")) as f: activation = pd.read_excel(f)
    required = {"gene_name","baseMean","log2FoldChange","lfcSE","padj","KO","cell_type","stimulation"}
    if not required.issubset(de): raise ValueError(f"Missing {sorted(required-set(de))}")
    de = de.dropna(subset=["gene_name","KO","cell_type","stimulation"]).copy()
    de["condition"] = de.cell_type.astype(str)+"_"+de.stimulation.astype(str)
    de = de[de.condition.isin(CONDITIONS) & (pd.to_numeric(de.padj,errors="coerce")<=0.1)]
    base = np.load(fallback_model,allow_pickle=False)
    bg=base["genes"].astype(str).tolist(); bt=base["targets"].astype(str).tolist()
    pg=sorted(set(de.gene_name.astype(str))); pt=sorted(set(de.KO.astype(str)))
    genes=np.asarray(bg+[g for g in pg if g not in set(bg)],dtype=str)
    targets=np.asarray(bt+[t for t in pt if t not in set(bt)],dtype=str)
    gp={g:i for i,g in enumerate(genes)}; tp={g:i for i,g in enumerate(targets)}
    shape=(4,len(targets),len(genes)); effects=np.zeros(shape,np.float32); unc=np.zeros(shape,np.float32)
    baseline=np.zeros((4,len(genes)),np.float32); counts=np.zeros((4,len(targets)),np.int32)
    effect_source=np.full((4,len(targets)),"GSE314342_fallback",dtype="<U20")
    bc=base["conditions"].astype(str).tolist()
    for ci,c in enumerate(CONDITIONS):
        bi=bc.index("Rest" if c.endswith("Resting") else "Stim8hr")
        baseline[ci,:len(bg)]=base["baseline"][bi]; effects[ci,:len(bt),:len(bg)]=base["effects"][bi]
        unc[ci,:len(bt),:len(bg)]=base["uncertainty"][bi]; counts[ci,:len(bt)]=base["counts"][bi]
    grouped=de.groupby(["condition","KO","gene_name"],as_index=False).agg(log2FoldChange=("log2FoldChange","mean"),lfcSE=("lfcSE","mean"),baseMean=("baseMean","mean"))
    for r in grouped.itertuples(index=False):
        ci,ti,gi=CONDITIONS.index(r.condition),tp[str(r.KO)],gp[str(r.gene_name)]
        effects[ci,ti,gi]=float(r.log2FoldChange); unc[ci,ti,gi]=float(r.lfcSE)
        baseline[ci,gi]=np.log2(1+max(float(r.baseMean),0)); effect_source[ci,ti]="GSE278572_primary"
    activation=activation.dropna(subset=["sg_target","HTO_maxID"])
    at=np.asarray(sorted(set(activation.sg_target.astype(str))),dtype=str); ac=np.asarray(sorted(set(activation.HTO_maxID.astype(str))),dtype=str)
    av=np.full((len(ac),len(at)),np.nan,np.float32); ati={g:i for i,g in enumerate(at)}; aci={g:i for i,g in enumerate(ac)}
    for _,r in activation.iterrows(): av[aci[str(r["HTO_maxID"])],ati[str(r["sg_target"])]]=float(r["mean.activation.score"])
    names=[]; frames=[]
    with zipfile.ZipFile(screens_zip) as z:
        for mode in ("CRISPRa","CRISPRi"):
            for cytokine in ("IL2","IFNG"):
                with z.open(_member(z,f"mageck.test.{mode}.{cytokine}.gene_summary.txt")) as f: frame=pd.read_csv(f,sep="\t")
                frame["id"]=frame.id.astype(str); frames.append(frame.set_index("id")); names.append(f"{mode}_{cytokine}")
    st=np.asarray(sorted(set().union(*(set(x.index) for x in frames))),dtype=str); sp={g:i for i,g in enumerate(st)}
    sl=np.zeros((4,len(st)),np.float32); sf=np.ones_like(sl)
    for pi,frame in enumerate(frames):
        for gene,r in frame.iterrows(): sl[pi,sp[gene]]=float(r["pos|lfc"]); sf[pi,sp[gene]]=float(min(r["pos|fdr"],r["neg|fdr"]))
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,genes=genes,conditions=np.asarray(CONDITIONS),targets=targets,baseline=baseline,effects=effects,uncertainty=unc,counts=counts,effect_source=effect_source,source=np.asarray("GSE278572_primary"),effect_unit=np.asarray("log2_fold_change"),activation_targets=at,activation_contexts=ac,activation_score=av,screen_targets=st,screen_phenotypes=np.asarray(names),screen_lfc=sl,screen_fdr=sf)
    summary={"primary_source":"GSE278572 / Zenodo 13924126","auxiliary_source":"Zenodo 5784651","fallback_source":"GSE314342","conditions":CONDITIONS,"targets":len(targets),"response_genes":len(genes),"gse278572_targets":len(pt),"gse278572_significant_effects":len(grouped),"activation_targets":len(at),"screen_targets":len(st),"clinical_use":False}
    out.with_suffix(".metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); return summary
