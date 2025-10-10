#!/usr/bin/env python3
"""
Dashboard Montecarlo π (Ray workers)

Versión con visualización de precisión (%) acotada automáticamente.
"""

import argparse
import math
import re
from pathlib import Path
from typing import List, Optional
import pandas as pd
import plotly.express as px


def infer_workers_from_name(path: Path) -> Optional[int]:
    name = path.stem.lower()
    m = re.search(r"(?:^|[_\-])w(?:orkers?)?_?(\d+)(?:[_\-]|$)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"work(?:er|ers)?_?(\d+)$", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[_\-])(\d+)w(?:[_\-]|$)", name)
    if m:
        return int(m.group(1))
    return None


def load_inputs(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df.columns = [c.lower() for c in df.columns]
        expected = {"n", "pi", "time", "vel"}
        if not expected.issubset(df.columns):
            raise ValueError(f"'{p}' no contiene las columnas requeridas {expected}.")
        df["source_file"] = p.name
        df["workers"] = infer_workers_from_name(p)
        frames.append(df)

    big = pd.concat(frames, ignore_index=True)
    for c in ["n", "pi", "time", "vel"]:
        big[c] = pd.to_numeric(big[c], errors="coerce")
    big = big.sort_values(["workers", "n", "time"], na_position="last").reset_index(drop=True)
    big["run_id"] = big.groupby(["workers", "n"], dropna=False).cumcount() + 1
    return big


def compute_aggregates(df: pd.DataFrame, baseline_workers: Optional[int] = 1) -> pd.DataFrame:
    df = df.copy()
    df["error_abs"] = (df["pi"] - math.pi).abs()
    df["error_rel"] = df["error_abs"] / math.pi
    df["precision"] = 100 * (1 - df["error_rel"])  # precisión en %

    agg = (
        df.groupby(["n", "workers"], dropna=False)
          .agg(time_mean=("time", "mean"),
               time_std=("time", "std"),
               vel_mean=("vel", "mean"),
               vel_std=("vel", "std"),
               pi_mean=("pi", "mean"),
               precision_mean=("precision", "mean"),
               runs=("pi", "size"))
          .reset_index()
    )

    if baseline_workers is not None and baseline_workers in agg["workers"].dropna().unique():
        base = (agg[agg["workers"] == baseline_workers][["n", "time_mean"]]
                .rename(columns={"time_mean": "time_baseline"}))
        agg = agg.merge(base, on="n", how="left")
    else:
        base = (agg.dropna(subset=["workers"])
                  .sort_values(["n", "workers"])
                  .groupby("n", as_index=False)
                  .first()[["n", "time_mean"]]
                  .rename(columns={"time_mean": "time_baseline"}))
        agg = agg.merge(base, on="n", how="left")

    agg["speedup"] = agg["time_baseline"] / agg["time_mean"]
    agg["efficiency"] = agg["speedup"] / agg["workers"]
    return agg


def make_dashboard(df: pd.DataFrame, agg: pd.DataFrame, out_html: Path, out_csv: Path) -> None:
    workers_order = sorted([w for w in agg["workers"].dropna().unique()])
    n_order = sorted(agg["n"].unique())

    # --- Fig 1: Tiempo medio ---
    fig_time = px.line(
        agg.sort_values(["workers", "n"]),
        x="n", y="time_mean", color="workers", error_y="time_std",
        markers=True,
        title="Tiempo medio por tamaño de muestra (n) y número de workers",
        labels={"n": "n (puntos)", "time_mean": "Tiempo medio (s)", "workers": "Workers"}
    )
    fig_time.update_xaxes(type="log")

    # --- Fig 2: Velocidad ---
    fig_vel = px.line(
        agg.sort_values(["workers", "n"]),
        x="n", y="vel_mean", color="workers", error_y="vel_std",
        markers=True,
        title="Velocidad media (puntos/segundo) por n y workers",
        labels={"n": "n (puntos)", "vel_mean": "Velocidad media (pts/s)", "workers": "Workers"}
    )
    fig_vel.update_xaxes(type="log")

    # --- Fig 3: Speedup ---
    fig_speedup = px.bar(
        agg.dropna(subset=["workers"]),
        x="workers", y="speedup", facet_col="n", facet_col_wrap=3,
        title="Speedup vs número de workers (por n)",
        labels={"workers": "Workers", "speedup": "Speedup (T_base / T_workers)"}
    )
    fig_speedup.update_yaxes(matches=None)

    # --- Fig 4: Eficiencia ---
    fig_eff = px.line(
        agg.dropna(subset=["workers"]).sort_values(["n", "workers"]),
        x="workers", y="efficiency", facet_col="n", facet_col_wrap=3,
        markers=True,
        title="Eficiencia vs número de workers (por n)",
        labels={"workers": "Workers", "efficiency": "Eficiencia = Speedup / Workers"}
    )
    fig_eff.update_yaxes(matches=None, range=[0, 1.1])

    # --- Fig 5: Precisión (%), rango dinámico ---
    min_prec, max_prec = agg["precision_mean"].min(), agg["precision_mean"].max()
    margin = (max_prec - min_prec) * 0.05 if max_prec > min_prec else 0.05
    y_min, y_max = max(0, min_prec - margin), min(100, max_prec + margin)

    fig_prec = px.line(
        agg.sort_values(["workers", "n"]),
        x="n", y="precision_mean", color="workers",
        markers=True,
        title="Precisión media (%) por n y número de workers",
        labels={"n": "n (puntos)", "precision_mean": "Precisión media (%)", "workers": "Workers"}
    )
    fig_prec.update_xaxes(type="log")
    fig_prec.update_yaxes(range=[y_min, y_max])

    # --- Tabla resumen ---
    display_cols = ["n", "workers", "runs", "time_mean", "time_std",
                    "vel_mean", "vel_std", "pi_mean", "precision_mean",
                    "speedup", "efficiency"]
    table_html = (agg[display_cols]
                  .sort_values(["n", "workers"])
                  .to_html(index=False, float_format=lambda x: f"{x:.6g}", classes="table table-striped"))

    # --- Conclusiones ---
    insights = []
    best = agg.loc[agg["time_mean"].idxmin()]
    insights.append(f"- Configuración más rápida: n={int(best['n'])}, workers={int(best['workers'])} "
                    f"→ {best['time_mean']:.3f}s, velocidad={best['vel_mean']:.3g} pts/s.")
    insights_html = "<ul>" + "".join([f"<li>{line}</li>" for line in insights]) + "</ul>"

    agg.to_csv(out_csv, index=False)

    parts = [
        fig_time.to_html(full_html=False, include_plotlyjs="cdn"),
        fig_vel.to_html(full_html=False, include_plotlyjs=False),
        fig_speedup.to_html(full_html=False, include_plotlyjs=False),
        fig_eff.to_html(full_html=False, include_plotlyjs=False),
        fig_prec.to_html(full_html=False, include_plotlyjs=False),
    ]

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Dashboard Montecarlo π · Ray</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 24px; }}
.card {{ background: #fff; border: 1px solid #ddd; border-radius: 12px; padding: 16px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid #ddd; padding: 6px; text-align: right; }}
th {{ background: #f7f7f7; text-align: center; }}
</style>
</head>
<body>
<h1>Benchmark Montecarlo π (Ray Workers)</h1>
<div class="grid">
<div class="card">{parts[0]}</div>
<div class="card">{parts[1]}</div>
<div class="card">{parts[2]}</div>
<div class="card">{parts[3]}</div>
<div class="card">{parts[4]}</div>
</div>
<h2>Resumen numérico</h2>
<div class="card">{table_html}</div>
<h2>Conclusiones</h2>
<div class="card">{insights_html}</div>
</body>
</html>"""
    out_html.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Genera un dashboard HTML de benchmarks Montecarlo π (Ray).")
    parser.add_argument("--inputs", nargs="*", type=str, help="CSV de entrada.")
    parser.add_argument("--baseline-workers", type=int, default=1, help="Workers de referencia (default=1).")
    parser.add_argument("--out-html", type=str, default="dashboard_montecarlo.html", help="HTML de salida.")
    parser.add_argument("--out-csv", type=str, default="resumen_montecarlo.csv", help="CSV resumen.")
    args = parser.parse_args()

    if args.inputs:
        paths = [Path(p) for p in args.inputs]
    else:
        paths = sorted(Path(".").glob("resultados*.csv"))
    if not paths:
        raise SystemExit("No se han encontrado CSV de entrada.")

    df = load_inputs(paths)
    agg = compute_aggregates(df, baseline_workers=args.baseline_workers)
    make_dashboard(df, agg, Path(args.out_html), Path(args.out_csv))
    print(f"OK - Dashboard: {args.out_html} · Resumen CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
