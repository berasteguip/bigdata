#!/usr/bin/env python3
"""
Dashboard Montecarlo π (Ray workers) — Versión categorías + precisión + export individual

Cambios solicitados:
1) 'n' tratado como CATEGORÍA (espaciado uniforme: 5M, 10M, 50M), no escala log.
2) Mostrar PRECISIÓN = 1 - error_relativo, en vez de error absoluto.
3) Exportar cada visualización por separado (HTML y, si hay kaleido instalado, también PNG).

Entrada: uno o varios CSV con columnas: n, pi, time, vel
Salida:
 - dashboard_montecarlo.html (panel único)
 - resumen_montecarlo.csv     (tabla agregada)
 - figs/tiempos.html, figs/velocidades.html, figs/speedup.html, figs/eficiencia.html, figs/precision.html
   (+ PNG si hay kaleido)
"""
import argparse
import math
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px

# Intento opcional de exportar a PNG
_HAVE_KALEIDO = False
try:
    import kaleido  # noqa: F401
    _HAVE_KALEIDO = True
except Exception:
    _HAVE_KALEIDO = False


def infer_workers_from_name(path: Path) -> Optional[int]:
    name = path.stem.lower()
    m = re.search(r"(?:^|[_\-])w(?:orkers?)?_?(\d+)(?:[_\-]|$)", name, flags=re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"work(?:er|ers)?_?(\d+)$", name, flags=re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[_\-])(\d+)w(?:[_\-]|$)", name, flags=re.I)
    if m:
        return int(m.group(1))
    return None


def human_n(n: float) -> str:
    """Convierte n en etiqueta humana (5M, 10M, 50M, etc.)."""
    n = float(n)
    if n >= 1e6 and abs(n - round(n / 1e6) * 1e6) < 1e-6:
        return f"{int(round(n/1e6))}M"
    if n >= 1e3 and abs(n - round(n / 1e3) * 1e3) < 1e-6:
        return f"{int(round(n/1e3))}k"
    return str(int(n))

def load_inputs(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df.columns = [c.lower() for c in df.columns]
        expected = {"n", "pi", "time", "vel"}
        if not expected.issubset(df.columns):
            raise ValueError(f"'{p}': faltan columnas {expected}. Columnas detectadas: {list(df.columns)}")
        df["source_file"] = p.name
        df["workers"] = infer_workers_from_name(p)
        frames.append(df)

    big = pd.concat(frames, ignore_index=True)
    for c in ["n", "pi", "time", "vel"]:
        big[c] = pd.to_numeric(big[c], errors="coerce")
    big["error_abs"] = (big["pi"] - math.pi).abs()
    big["error_rel"] = big["error_abs"] / math.pi
    big["precision"] = 1.0 - big["error_rel"]
    big["n_label"] = big["n"].map(human_n)
    return big

def compute_aggregates(df: pd.DataFrame, baseline_workers: Optional[int] = 1) -> pd.DataFrame:
    agg = (
        df.groupby(["n", "n_label", "workers"], dropna=False)
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
        base = (agg.dropna(subset=["workers"]
                  ).sort_values(["n", "workers"]
                  ).groupby("n", as_index=False).first()[["n", "time_mean"]]
                  .rename(columns={"time_mean": "time_baseline"}))
        agg = agg.merge(base, on="n", how="left")

    agg["speedup"] = agg["time_baseline"] / agg["time_mean"]
    agg["efficiency"] = agg["speedup"] / agg["workers"]

    n_order = sorted(agg["n"].unique())
    n_label_order = [human_n(x) for x in n_order]
    agg["n_label"] = pd.Categorical(agg["n_label"], categories=n_label_order, ordered=True)
    return agg

def _save_fig(fig, out_html: Path, out_png: Optional[Path] = None):
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    if _HAVE_KALEIDO and out_png is not None:
        try:
            fig.write_image(out_png, scale=2, width=1200, height=700)
        except Exception:
            pass

def make_dashboard(agg: pd.DataFrame, out_html: Path, out_csv: Path) -> None:
    workers_order = sorted([w for w in agg["workers"].dropna().unique()])
    n_label_order = list(agg["n_label"].cat.categories)

    fig_time = px.line(
        agg.sort_values(["workers", "n"]),
        x="n_label", y="time_mean", color="workers", error_y="time_std",
        markers=True, category_orders={"n_label": n_label_order, "workers": workers_order},
        title="Tiempo medio por tamaño de muestra (n categórico) y número de workers",
        labels={"n_label": "n (categoría)", "time_mean": "Tiempo medio (s)", "workers": "Workers"}
    )

    fig_vel = px.line(
        agg.sort_values(["workers", "n"]),
        x="n_label", y="vel_mean", color="workers", error_y="vel_std",
        markers=True, category_orders={"n_label": n_label_order, "workers": workers_order},
        title="Velocidad media (puntos/segundo) por n (categórico) y workers",
        labels={"n_label": "n (categoría)", "vel_mean": "Velocidad media (pts/s)", "workers": "Workers"}
    )

    fig_speedup = px.bar(
        agg.dropna(subset=["workers"]),
        x="workers", y="speedup", facet_col="n_label", facet_col_wrap=3,
        category_orders={"n_label": n_label_order, "workers": workers_order},
        title="Speedup vs número de workers (por n)",
        labels={"workers": "Workers", "speedup": "Speedup (T_base / T_workers)", "n_label": "n"}
    )
    fig_speedup.update_yaxes(matches=None)

    fig_eff = px.line(
        agg.dropna(subset=["workers"]).sort_values(["n", "workers"]),
        x="workers", y="efficiency", facet_col="n_label", facet_col_wrap=3,
        markers=True,
        category_orders={"n_label": n_label_order, "workers": workers_order},
        title="Eficiencia vs número de workers (por n)",
        labels={"workers": "Workers", "efficiency": "Eficiencia = Speedup / Workers", "n_label": "n"}
    )
    fig_eff.update_yaxes(matches=None, range=[0, 1.1])

    fig_prec = px.line(
        agg.sort_values(["workers", "n"]),
        x="n_label", y="precision_mean", color="workers",
        markers=True, category_orders={"n_label": n_label_order, "workers": workers_order},
        title="Precisión media de la estimación de π por n (categórico) y workers",
        labels={"n_label": "n (categoría)", "precision_mean": "Precisión media (1 - |π̂-π|/π)", "workers": "Workers"}
    )
    fig_prec.update_yaxes(range=[0, 1])

    display_cols = ["n_label", "workers", "runs", "time_mean", "time_std",
                    "vel_mean", "vel_std", "pi_mean", "precision_mean", "speedup", "efficiency"]
    table_html = (agg[display_cols]
                  .sort_values(["n", "workers"])
                  .rename(columns={"n_label": "n"})
                  .to_html(index=False, float_format=lambda x: f"{x:.6g}", classes="table table-striped"))

    insights = []
    best = agg.loc[agg["time_mean"].idxmin()]
    insights.append(f"- Configuración global más rápida: n={best['n_label']}, workers={int(best['workers'])} → tiempo medio={best['time_mean']:.3f}s, velocidad media={best['vel_mean']:.3g} pts/s.")
    for nlab in n_label_order:
        sub = agg[agg["n_label"] == nlab].dropna(subset=["workers"]).sort_values("time_mean")
        if len(sub) == 0:
            continue
        r0 = sub.iloc[0]
        eff_pct = 100 * r0["efficiency"] if pd.notna(r0["efficiency"]) else float("nan")
        insights.append(f"- Para n={nlab}: mejor workers={int(r0['workers'])} → speedup={r0['speedup']:.2f}x, eficiencia={eff_pct:.0f}%.")
    bad = agg.dropna(subset=["efficiency"])
    bad = bad[(bad["workers"] > 1) & (bad["efficiency"] < 0.7)]
    if not bad.empty:
        ns = ", ".join([str(x) for x in agg.loc[bad.index, "n_label"].unique()])
        insights.append(f"- Rendimientos decrecientes detectados (eficiencia <70%) en n ∈ {{{ns}}}.")
    else:
        insights.append("- Buena escalabilidad: no hay eficiencias <70% en los datos agregados.")

    insights_html = "<ul>" + "".join([f"<li>{line}</li>" for line in insights]) + "</ul>"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)

    figs_dir = Path("figs")
    _save_fig(fig_time, figs_dir / "tiempos.html", figs_dir / "tiempos.png")
    _save_fig(fig_vel, figs_dir / "velocidades.html", figs_dir / "velocidades.png")
    _save_fig(fig_speedup, figs_dir / "speedup.html", figs_dir / "speedup.png")
    _save_fig(fig_eff, figs_dir / "eficiencia.html", figs_dir / "eficiencia.png")
    _save_fig(fig_prec, figs_dir / "precision.html", figs_dir / "precision.png")

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
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dashboard Montecarlo π · Ray</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, 'Helvetica Neue', Arial, sans-serif; margin: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 24px; }}
    .card {{ background: #fff; border: 1px solid #e6e6e6; border-radius: 12px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
    .muted {{ color: #666; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
    th {{ background: #f7f7f7; text-align: center; }}
    td:first-child, th:first-child {{ text-align: center; }}
  </style>
</head>
<body>
  <h1>Benchmark Montecarlo π (Ray Workers)</h1>
  <p class="muted">Panel comparativo por <b>n (categórico)</b> y <b>workers</b>. Muestra <i>speedup</i>, <i>eficiencia</i> y <i>precisión</i>.</p>
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
  <p class="muted">Figuras individuales exportadas en <code>./figs/</code>. {'(Incluye PNG)' if _HAVE_KALEIDO else '(Para PNG: pip install kaleido)'}.</p>
</body>
</html>"""
    out_html.write_text(html, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Genera dashboard HTML y figuras individuales de benchmarks Montecarlo π (Ray)." )
    parser.add_argument("--inputs", nargs="*", type=str,
                        help="CSV de entrada. Si se omite, se usan 'resultados*.csv' del directorio actual.")
    parser.add_argument("--baseline-workers", type=int, default=1,
                        help="Workers de referencia para speedup (por defecto 1).")
    parser.add_argument("--out-html", type=str, default="dashboard_montecarlo.html",
                        help="Ruta del HTML de salida.")
    parser.add_argument("--out-csv", type=str, default="resumen_montecarlo.csv",
                        help="Ruta del CSV resumen de salida.")
    args = parser.parse_args()

    if args.inputs:
        paths = [Path(p) for p in args.inputs]
    else:
        paths = sorted(Path(".").glob("resultados*.csv"))
    if not paths:
        raise SystemExit("No se han encontrado CSV de entrada. Pasa --inputs o coloca ficheros 'resultados*.csv'." )

    df = load_inputs(paths)
    agg = compute_aggregates(df, baseline_workers=args.baseline_workers)
    make_dashboard(agg, out_html=Path(args.out_html), out_csv=Path(args.out_csv))
    print(f"OK - Dashboard: {args.out_html}  ·  Resumen CSV: {args.out_csv}  ·  Figuras: ./figs/")

if __name__ == "__main__":
    main()
