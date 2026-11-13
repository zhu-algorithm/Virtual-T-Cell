"""Command line entry point for evidence-grounded virtual-cell research learning."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from .schema import canonical_json, read_jsonl, write_jsonl


def write_report(path: str | Path, report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Virtual T Cell research learning core (not demonstrated AGI).")
    commands = root.add_subparsers(dest="command", required=True)

    command = commands.add_parser("demo", help="Run an explicitly synthetic liver/spleen integration example.")
    command.add_argument("--out-dir", required=True)

    command = commands.add_parser("import-reference", help="Convert a source-labelled NPZ into experimental-summary JSONL.")
    command.add_argument("--model", required=True)
    command.add_argument("--organ", required=True)
    command.add_argument("--cell-type", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--outcomes", nargs="+")
    command.add_argument("--effect-unit-override")

    command = commands.add_parser("train", help="Fit response evidence; observational atlases are not response labels.")
    command.add_argument("--records", required=True)
    command.add_argument("--out-model", required=True)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--random-features", type=int, default=16)
    command.add_argument("--prior-precision", type=float, default=1.0)
    command.add_argument("--allow-synthetic", action="store_true")

    command = commands.add_parser("evaluate", help="Evaluate on independent groups and report training-mean baseline.")
    command.add_argument("--model", required=True)
    command.add_argument("--records", required=True)
    command.add_argument("--out", required=True)

    command = commands.add_parser("plan", help="Rank potential measurements; does not execute any experiment.")
    command.add_argument("--model", required=True)
    command.add_argument("--candidates", required=True)
    command.add_argument("--outcome", required=True)
    command.add_argument("--budget", type=float, required=True)
    command.add_argument("--direction", choices=("maximize", "minimize"), default="maximize")
    command.add_argument("--min-information-value", type=float, default=1e-6)
    command.add_argument("--allow-extrapolation", action="store_true")
    command.add_argument("--out", required=True)

    command = commands.add_parser("update", help="Train a candidate with replay; promote only when the validation gate accepts.")
    command.add_argument("--model", required=True)
    command.add_argument("--replay", required=True)
    command.add_argument("--new", required=True)
    command.add_argument("--validation", required=True)
    command.add_argument("--out-dir", required=True)
    command.add_argument("--max-relative-regression", type=float, default=0.05)
    command.add_argument("--allow-synthetic", action="store_true")
    command.add_argument("--promote-if-accepted", action="store_true")

    command = commands.add_parser("ingest-counts", help="QC and store a raw-count single-cell batch as observational states.")
    command.add_argument("--counts", required=True)
    command.add_argument("--metadata", required=True)
    _atlas_arguments(command)

    command = commands.add_parser("ingest-h5ad", help="Optional H5AD import; an explicit raw-count layer is required.")
    command.add_argument("--h5ad", required=True)
    command.add_argument("--counts-layer", required=True)
    _atlas_arguments(command)

    command = commands.add_parser("atlas-status", help="Summarize tissue/context coverage and optionally export states.")
    command.add_argument("--atlas", required=True)
    command.add_argument("--out-states")
    command.add_argument("--out-belief", help="Export donor-weighted descriptive state estimates.")

    command = commands.add_parser("evidence-add", help="Append empirical or explicitly synthetic records without training.")
    command.add_argument("--database", required=True)
    command.add_argument("--records", required=True)

    command = commands.add_parser("sync-inbox", help="Ingest immutable batch manifests; optional local polling.")
    command.add_argument("--inbox", required=True)
    command.add_argument("--atlas", required=True)
    command.add_argument("--evidence", required=True)
    command.add_argument("--out")
    command.add_argument("--state-belief", help="Refresh descriptive atlas belief after each ingestion pass.")
    command.add_argument("--learning-config", help="Optional JSON config enabling gated response-model updates.")
    command.add_argument("--watch", action="store_true")
    command.add_argument("--interval", type=float, default=30.0)
    return root


def _atlas_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--atlas", required=True)
    command.add_argument("--batch-id", required=True)
    command.add_argument("--source", required=True)
    command.add_argument("--evidence-type", choices=("observational", "synthetic"), default="observational")
    command.add_argument("--min-counts", type=float, default=1)
    command.add_argument("--min-genes", type=int, default=1)
    command.add_argument("--max-mito-fraction", type=float, default=1.0)


def _run(args: argparse.Namespace) -> dict:
    if args.command == "import-reference":
        from .adapters import import_reference_npz
        records, report = import_reference_npz(args.model, organ=args.organ, cell_type=args.cell_type,
                                               outcome_genes=args.outcomes, effect_unit_override=args.effect_unit_override)
        write_jsonl(args.out, records)
        write_report(str(args.out) + ".provenance.json", report)
        return {"output": args.out, "records": len(records), "provenance": report}
    if args.command == "train":
        from .world_model import BayesianWorldModel
        from .belief import BeliefState
        model = BayesianWorldModel.fit(read_jsonl(args.records), seed=args.seed, random_features=args.random_features,
                                       prior_precision=args.prior_precision, allow_synthetic=args.allow_synthetic)
        Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.out_model)
        belief = BeliefState.from_model(model).to_dict()
        write_report(str(args.out_model) + ".belief.json", belief)
        return {"model": args.out_model, "model_id": model.metadata.get("model_id"), "belief": belief}
    if args.command in {"evaluate", "plan", "update"}:
        from .world_model import BayesianWorldModel
        if args.command == "update":
            from .runner import _file_hash
            expected_model_sha = _file_hash(Path(args.model))
        model = BayesianWorldModel.load(args.model)
        if args.command == "update" and _file_hash(Path(args.model)) != expected_model_sha:
            raise ValueError("current model changed while it was loaded")
        if args.command == "evaluate":
            from .continual import evaluate_model
            report = evaluate_model(model, read_jsonl(args.records))
            write_report(args.out, report)
            return report
        if args.command == "plan":
            from .planner import plan_experiment
            report = plan_experiment(model, read_jsonl(args.candidates), outcome=args.outcome, budget=args.budget,
                                     direction=args.direction, min_information_value=args.min_information_value,
                                     allow_extrapolation=args.allow_extrapolation)
            write_report(args.out, report)
            return report
        from .continual import propose_update
        current_path = Path(args.model).resolve()
        output = Path(args.out_dir)
        output.mkdir(parents=True, exist_ok=True)
        candidate_path = output / "candidate_model.npz"
        if current_path == candidate_path.resolve():
            raise ValueError("candidate output must not overwrite the current model")
        candidate, report = propose_update(model, read_jsonl(args.replay), read_jsonl(args.new),
                                            read_jsonl(args.validation), max_relative_regression=args.max_relative_regression,
                                            allow_synthetic=args.allow_synthetic)
        candidate.save(candidate_path)
        report["candidate_path"] = str(candidate_path)
        report["promoted"] = False
        write_report(output / "gate_report.json", report)
        if args.promote_if_accepted and report["accepted"]:
            from .runner import _model_lock, _promote_candidate
            try:
                with _model_lock(current_path):
                    saved = _promote_candidate(candidate, model_path=current_path, candidate_path=candidate_path,
                                               expected_sha256=expected_model_sha,
                                               current_model_id=model.metadata["model_id"], output_dir=output)
                report.update(saved)
                report["promoted"] = True
            except (OSError, ValueError, ImportError) as error:
                report["promotion_error"] = str(error)
                write_report(output / "gate_report.json", report)
                raise
        write_report(output / "gate_report.json", report)
        return report
    if args.command in {"ingest-counts", "ingest-h5ad", "atlas-status"}:
        from .atlas import AtlasStore
        atlas = AtlasStore(args.atlas)
        try:
            if args.command == "atlas-status":
                if args.out_states:
                    write_jsonl(args.out_states, atlas.pseudobulk_records())
                if args.out_belief:
                    from .state import summarize_atlas_states
                    write_report(args.out_belief, summarize_atlas_states(atlas.pseudobulk_records()))
                return atlas.summary()
            kwargs = {"batch_id": args.batch_id, "source": args.source, "evidence_type": args.evidence_type,
                      "min_counts": args.min_counts, "min_genes": args.min_genes, "max_mito_fraction": args.max_mito_fraction}
            if args.command == "ingest-counts":
                return atlas.ingest_counts(args.counts, args.metadata, **kwargs)
            return atlas.ingest_h5ad(args.h5ad, counts_layer=args.counts_layer, **kwargs)
        finally:
            atlas.close()
    if args.command == "evidence-add":
        from .evidence import EvidenceStore
        with EvidenceStore(args.database) as store:
            return store.append(read_jsonl(args.records))
    if args.command == "sync-inbox":
        from .pipeline import sync_inbox
        if not 1 <= args.interval <= 3600:
            raise ValueError("interval must be between 1 and 3600 seconds")
        while True:
            report = sync_inbox(args.inbox, atlas_path=args.atlas, evidence_path=args.evidence)
            if args.state_belief:
                from .atlas import AtlasStore
                from .state import summarize_atlas_states
                with AtlasStore(args.atlas) as atlas:
                    write_report(args.state_belief, summarize_atlas_states(atlas.pseudobulk_records()))
            if args.learning_config:
                if report["errors"]:
                    report["continual_learning"] = {"status": "ingestion_errors", "promoted": False}
                else:
                    try:
                        from .runner import run_learning_cycle
                        config_path = Path(args.learning_config).resolve()
                        config = json.loads(config_path.read_text(encoding="utf-8"))
                        if not isinstance(config, dict):
                            raise ValueError("learning config must be a JSON object")
                        permitted = {"model_path", "validation_path", "output_dir", "replay_path",
                                     "allow_synthetic", "max_relative_regression", "promote"}
                        if set(config) - permitted:
                            raise ValueError("unrecognized learning config fields: " + ", ".join(sorted(set(config) - permitted)))
                        for field in ("model_path", "validation_path", "output_dir", "replay_path"):
                            if field in config and config[field] is not None:
                                if not isinstance(config[field], str) or not config[field]:
                                    raise ValueError(f"{field} must be a nonempty path string")
                                value = Path(config[field])
                                config[field] = str(value if value.is_absolute() else config_path.parent / value)
                        if any(field not in config for field in ("model_path", "validation_path", "output_dir")):
                            raise ValueError("learning config needs model_path, validation_path, output_dir")
                        report["continual_learning"] = run_learning_cycle(evidence_path=args.evidence, **config)
                        report["response_model_automatically_promoted"] = report["continual_learning"].get("promoted", False)
                    except (ValueError, KeyError, OSError, ImportError) as error:
                        report["errors"].append({"stage": "continual_learning", "error": str(error)})
            if args.out:
                write_report(args.out, report)
            if not args.watch:
                return report
            print(canonical_json(report), flush=True)
            time.sleep(args.interval)
    if args.command == "demo":
        return run_demo(args.out_dir)
    raise ValueError("unknown command")


def run_demo(output: str | Path) -> dict:
    from .atlas import AtlasStore
    from .belief import BeliefState
    from .continual import evaluate_model, propose_update
    from .demo import write_demo_inputs
    from .pipeline import sync_inbox
    from .planner import plan_experiment
    from .world_model import BayesianWorldModel
    from .evidence import EvidenceStore
    from .runner import run_learning_cycle

    output = Path(output)
    manifest = write_demo_inputs(output / "inputs")
    initial = read_jsonl(output / "inputs/train.jsonl")
    new = read_jsonl(output / "inputs/new.jsonl")
    validation = read_jsonl(output / "inputs/validation.jsonl")
    test = read_jsonl(output / "inputs/test.jsonl")
    current = BayesianWorldModel.fit(initial, seed=0, random_features=8, allow_synthetic=True)
    current.save(output / "initial_model.npz")
    candidate, gate = propose_update(current, initial, new, validation, allow_synthetic=True)
    candidate.save(output / "candidate_model.npz")
    active = candidate if gate["accepted"] else current
    active.save(output / "active_model.npz")
    belief = BeliefState.from_model(active).to_dict()
    evaluation = evaluate_model(active, test)
    plan = plan_experiment(active, read_jsonl(output / "inputs/candidates.jsonl"), outcome="response_a", budget=3)
    ingestion = sync_inbox(manifest["inbox"], atlas_path=output / "atlas.sqlite", evidence_path=output / "evidence.sqlite")
    repeated = sync_inbox(manifest["inbox"], atlas_path=output / "atlas.sqlite", evidence_path=output / "evidence.sqlite")
    atlas = AtlasStore(output / "atlas.sqlite")
    try:
        atlas_summary = atlas.summary()
        states = atlas.pseudobulk_records()
        write_jsonl(output / "atlas_states.jsonl", states)
        from .state import summarize_atlas_states
        write_report(output / "atlas_belief.json", summarize_atlas_states(states))
    finally:
        atlas.close()
    stream_model = output / "stream_current.npz"
    if not stream_model.exists():
        current.save(stream_model)
    with EvidenceStore(output / "response_stream.sqlite") as journal:
        journal.append(new)
    cycle_arguments = {"model_path": stream_model, "evidence_path": output / "response_stream.sqlite",
                       "replay_path": output / "inputs/train.jsonl",
                       "validation_path": output / "inputs/validation.jsonl",
                       "output_dir": output / "stream_updates", "allow_synthetic": True}
    cycle = run_learning_cycle(**cycle_arguments)
    repeated_cycle = run_learning_cycle(**cycle_arguments)
    report = {"evidence_type": "synthetic", "biological_validation": False, "inputs": manifest,
              "belief": belief, "update_gate": gate, "test_evaluation": evaluation, "plan": plan,
              "ingestion": ingestion, "repeat_ingestion": repeated, "atlas": atlas_summary,
              "continual_cycle": cycle, "repeated_continual_cycle": repeated_cycle,
              "limitations": ["All liver/spleen example cells and response values are synthetic fixtures.",
                              "No sequencing, wet-lab execution, causal identification, or demonstrated AGI.",
                              "Probabilistic uncertainty is approximate; held-out coverage is not a guarantee."]}
    write_report(output / "demo_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = _run(args)
        print(canonical_json(result))
        return 1 if result.get("errors") else 0
    except KeyboardInterrupt:
        print("Stopped local batch watcher.", file=sys.stderr)
        return 130
    except (ValueError, KeyError, OSError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
