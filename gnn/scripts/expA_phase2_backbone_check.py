import csv
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent
PHASE2_CSV = ROOT / "phase2_elliptic_results.csv"
BACKBONE_CSV = ROOT / "phase3_elliptic_results.csv"


def load_phase2(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "Method": r["Method"],
                    "PR-AUC": float(r["PR-AUC"]),
                    "std": float(r["std"]),
                    "Phase": r.get("Phase", ""),
                }
            )
    return rows


def load_backbone(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "backbone": r["backbone"],
                    "seed": int(r["seed"]),
                    "PR-AUC": float(r["PR-AUC"]),
                }
            )
    return rows


def summarize_backbone(rows):
    by_model = {}
    for r in rows:
        by_model.setdefault(r["backbone"], []).append(r["PR-AUC"])
    stats = {}
    for name, vals in by_model.items():
        stats[name] = {
            "mean": mean(vals),
            "std": pstdev(vals),
            "n": len(vals),
        }
    return stats


def main():
    if not PHASE2_CSV.exists() or not BACKBONE_CSV.exists():
        missing = [str(p.name) for p in [PHASE2_CSV, BACKBONE_CSV] if not p.exists()]
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)}")

    phase2 = load_phase2(PHASE2_CSV)
    p2 = {r["Method"]: r for r in phase2}

    required = [
        "AISO(Smart)->AISO",
        "AISO(Smart)->AISO(Rand M)",
        "AISO(Rand)->AISO(Rand M)",
        "Random",
    ]
    for k in required:
        if k not in p2:
            raise KeyError(f"Required method not found in phase2 csv: {k}")

    smart_both = p2["AISO(Smart)->AISO"]["PR-AUC"]
    smart_rand2 = p2["AISO(Smart)->AISO(Rand M)"]["PR-AUC"]
    rand_both = p2["AISO(Rand)->AISO(Rand M)"]["PR-AUC"]
    random_sel = p2["Random"]["PR-AUC"]

    # Phase-2 proxy: directional structure contributes at stage-2 and stage-1.
    delta_stage2 = smart_both - smart_rand2
    delta_stage1 = smart_rand2 - rand_both
    delta_vs_random = smart_both - random_sel

    backbone_rows = load_backbone(BACKBONE_CSV)
    bstats = summarize_backbone(backbone_rows)

    for model in ["GCN", "GraphSAGE", "GAT", "GIN", "SGC"]:
        if model not in bstats:
            raise KeyError(f"Required backbone not found in phase3 csv: {model}")

    nonlinear_best = max([bstats["GCN"]["mean"], bstats["GraphSAGE"]["mean"], bstats["GAT"]["mean"]])
    sgc_mean = bstats["SGC"]["mean"]
    linear_gap = nonlinear_best - sgc_mean

    print("=== Exp A Proxy Check (from Phase2 + Backbone) ===")
    print("[Phase2: Smart M directional contribution]")
    print(f"AISO(Smart)->AISO             : {smart_both:.4f}")
    print(f"AISO(Smart)->AISO(Rand M)     : {smart_rand2:.4f}")
    print(f"AISO(Rand)->AISO(Rand M)      : {rand_both:.4f}")
    print(f"Random                        : {random_sel:.4f}")
    print(f"Delta stage-2 (Smart2-Rand2)  : {delta_stage2:+.4f}")
    print(f"Delta stage-1 (Smart1-Rand1)  : {delta_stage1:+.4f}")
    print(f"Delta vs Random               : {delta_vs_random:+.4f}")

    print("\n[Backbone on fixed AISO subgraph]")
    for m in ["GCN", "GraphSAGE", "GAT", "GIN", "SGC"]:
        print(f"{m:<10} mean={bstats[m]['mean']:.4f} std={bstats[m]['std']:.4f} n={bstats[m]['n']}")
    print(f"Nonlinear-best - SGC gap      : {linear_gap:+.4f}")

    # Conservative pass/fail checks for the proxy narrative.
    pass_stage2 = delta_stage2 > 0
    pass_stage1 = delta_stage1 > 0
    pass_linear = linear_gap > 0.25

    print("\n[Proxy Verdict]")
    print(f"Stage-2 directional gain > 0  : {'PASS' if pass_stage2 else 'FAIL'}")
    print(f"Stage-1 directional gain > 0  : {'PASS' if pass_stage1 else 'FAIL'}")
    print(f"Nonlinear >> linear (SGC)     : {'PASS' if pass_linear else 'FAIL'}")

    if pass_stage2 and pass_stage1 and pass_linear:
        print("Conclusion: Existing Phase2/Backbone evidence supports Exp A asymmetry claim as a proxy.")
    else:
        print("Conclusion: Proxy evidence is incomplete; run full symmetric-M ablation for Exp A.")


if __name__ == "__main__":
    main()
