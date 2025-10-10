#!/usr/bin/env python3
"""
Dashboard Montecarlo π (Ray workers)

Crea un ÚNICO dashboard (HTML) a partir de uno o varios CSV con columnas:
    n, pi, time, vel
(salida de pi_montecarlo.py + ejecucion.sh).

Características:
- Detecta automáticamente el nº de workers a partir del nombre del fichero (p. ej., resultados_w1.csv, resultados-workers2.csv).
- Agrega múltiples ejecuciones por (n, workers) y calcula medias, desviaciones, speedup y eficiencia.
- Genera un único HTML interactivo con varias gráficas y una tabla-resumen + conclusiones.
- También exporta un CSV resumen con las métricas agregadas.

Uso recomendado:
    python dashboard_montecarlo.py --inputs resultados_w1.csv resultados_w2.csv resultados_w3.csv

Alternativas:
    # Si no pasas --inputs, intentará leer todos los CSV que empiecen por "resultados" en el directorio actual.
    python dashboard_montecarlo.py

Requisitos: Python 3.8+, pandas, plotly
    pip install pandas plotly
"""
import argparse
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def infer_workers_from_name(path: Path) -> Optional[int]:
    """
    Intenta inferir el nº de workers a partir del nombre del fichero.
    Patrones aceptados (insensibles a may/min):
      *_w1.csv, *-w2.csv, *workers3*.csv, *_worker_2.csv, etc.
    """
    name = path.stem.lower()
    # patterns tipo w2 / w_2 / -w3 / _w1
    m = re.search(r"(?:^|[_\-])w(?:orkers?)?_?(\d+)(?:[_\-]|$)", name, flags=re.I)
    if m:
        return int(m.group(1))
    # patterns tipo workers2 / worker3 al final
    m = re.search(r"work(?:er|ers)?_?(\d+)$", name, flags=re.I)
    if m:
        return int(m.group(1))
    # patterns tipo ...-2w / _3w
    m = re.search(r"(?:^|[_\-])(\d+)w(?:[_\-]|$)", name, flags=re.I)
    if m:
        return int(m.group(1))
    return None


def load_inputs(paths: List[Path]) -> pd.DataFrame:
    """
    Lee múltiples CSV y los concatena, añadiendo columnas:
      - source_file
      - workers (inferido si es posible)
      - run_id (ordinal por grupo n,workers)
    Requiere columnas: n, pi, time, vel.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        expected = {"n", "pi", "time", "vel"}
        missing = expected - set(df.columns.str.lower())
        # Si el CSV tiene cabeceras con capitalización exacta del ejemplo, esto funcionará;
        # si vinieran en otro orden/caso, normalizamos a minúsculas:
        df.columns = [c.lower() for c in df.columns]
        if not expected.issubset(df.columns):
            raise ValueError(f"El fichero '{p}' no contiene las columnas requeridas {expected}. Columnas: {list(df.columns)}")
        df["source_file"] = p.name
        df["workers"] = infer_workers_from_name(p)
        frames.append(df)

    big = pd.concat(frames, ignore_index=True)
    # Tipos
    for c in ["n", "pi", "time", "vel"]:
        big[c] = pd.to_numeric(big[c], errors="coerce")
    # Orden lógico
    big = big.sort_values(["workers", "n", "time"], na_position="last").reset_index(drop=True)
    # Asigna run_id por grupo (n, workers) para contar repeticiones
    big["run_id"] = big.groupby(["workers", "n"], dropna=False).cumcount() + 1
    return big


def compute_aggregates(df: pd.DataFrame, baseline_workers: Optional[int] = 1) -> pd.DataFrame:
    """
    Devuelve un dataframe agregado por (n, workers) con medias y métricas derivadas:
      - time_mean, time_std
      - vel_mean, vel_std
      - pi_mean, error_abs_mean, error_rel_mean
      - runs
      - speedup (vs baseline workers para cada n)
      - efficiency = speedup / workers
    Si no hay baseline_workers presente, usa el menor workers disponible como baseline por cada n.
    """
    df = df.copy()
    df["error_abs"] = (df["pi"] - math.pi).abs()
    df["error_rel"] = df["error_abs"] / math.pi

    agg = (
        df.groupby(["n", "workers"], dropna=False)
          .agg(time_mean=("time", "mean"),
               time_std=("time", "std"),
               vel_mean=("vel", "mean"),
               vel_std=("vel", "std"),
               pi_mean=("pi", "mean"),
               error_abs_mean=("error_abs", "mean"),
               error_rel_mean=("error_rel", "mean"),
               runs=("pi", "size"))
          .reset_index()
    )

    # Para baseline: si existe baseline_workers, úsalo; si no, baseline dinámico por n = mínimo workers disponible
    if baseline_workers is not None and baseline_workers in agg["workers"].dropna().unique():
        base = (agg[agg["workers"] == baseline_workers]
                .loc[:, ["n", "time_mean"]]
                .rename(columns={"time_mean": "time_baseline"}))
        agg = agg.merge(base, on="n", how="left")
    else:
        # baseline por n: el workers mínimo
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
    """
    Genera un único HTML con:
      - Gráfica 1: Tiempo medio vs n (log-x), coloreado por workers (+ barras de error)
      - Gráfica 2: Velocidad media vs n (log-x), coloreado por workers
      - Gráfica 3: Speedup vs workers (facet por n)
      - Gráfica 4: Eficiencia vs workers (facet por n)
      - Gráfica 5: Error absoluto medio |pi - π| vs n (log-y), coloreado por workers
      - Tabla resumen (métricas agregadas)
      - Conclusiones automáticas (insights)
    """
    # Ordena categorías
    workers_order = sorted([w for w in agg["workers"].dropna().unique()])
    n_order = sorted(agg["n"].unique())

    # Fig 1: time vs n
    fig_time = px.line(
        agg.sort_values(["workers", "n"]),
        x="n", y="time_mean", color="workers", error_y="time_std",
        markers=True, category_orders={"workers": workers_order, "n": n_order},
        title="Tiempo medio por tamaño de muestra (n) y número de workers",
        labels={"n": "n (puntos)", "time_mean": "Tiempo medio (s)", "workers": "Workers"}
    )
    fig_time.update_xaxes(type="log")

    # Fig 2: vel vs n
    fig_vel = px.line(
        agg.sort_values(["workers", "n"]),
        x="n", y="vel_mean", color="workers", error_y="vel_std",
        markers=True, category_orders={"workers": workers_order, "n": n_order},
        title="Velocidad media (puntos/segundo) por n y workers",
        labels={"n": "n (puntos)", "vel_mean": "Velocidad media (pts/s)", "workers": "Workers"}
    )
    fig_vel.update_xaxes(type="log")

    # Fig 3: speedup vs workers (facet por n)
    fig_speedup = px.bar(
        agg.dropna(subset=["workers"]),
        x="workers", y="speedup", facet_col="n", facet_col_wrap=3,
        category_orders={"workers": workers_order, "n": n_order},
        title="Speedup vs número de workers (por n)",
        labels={"workers": "Workers", "speedup": "Speedup (T_base / T_workers)"}
    )
    fig_speedup.update_yaxes(matches=None)  # ejes independientes por facet
    fig_speedup.update_layout(bargap=0.2)

    # Fig 4: eficiencia vs workers (facet por n)
    fig_eff = px.line(
        agg.dropna(subset=["workers"]).sort_values(["n", "workers"]),
        x="workers", y="efficiency", facet_col="n", facet_col_wrap=3,
        markers=True,
        category_orders={"workers": workers_order, "n": n_order},
        title="Eficiencia vs número de workers (por n)",
        labels={"workers": "Workers", "efficiency": "Eficiencia = Speedup / Workers"}
    )
    fig_eff.update_yaxes(matches=None, range=[0, 1.1])

    # Fig 5: error vs n
    fig_err = px.line(
        agg.sort_values(["workers", "n"]),
        x="n", y="error_abs_mean", color="workers",
        markers=True, category_orders={"workers": workers_order, "n": n_order},
        title="Error absoluto medio |π̂ - π| por n y workers",
        labels={"n": "n (puntos)", "error_abs_mean": "Error absoluto medio", "workers": "Workers"}
    )
    fig_err.update_yaxes(type="log")
    fig_err.update_xaxes(type="log")

    # Tabla resumen (agregados)
    display_cols = ["n", "workers", "runs", "time_mean", "time_std", "vel_mean", "vel_std",
                    "pi_mean", "error_abs_mean", "speedup", "efficiency"]
    table_html = (agg[display_cols]
                  .sort_values(["n", "workers"])
                  .to_html(index=False, float_format=lambda x: f"{x:.6g}", classes="table table-striped"))

    # Conclusiones automáticas sencillas
    insights = []
    # 1) Config global más rápida
    best = agg.loc[agg["time_mean"].idxmin()]
    insights.append(f"- Configuración global más rápida: n={int(best['n'])}, workers={int(best['workers']) if pd.notna(best['workers']) else 'N/A'} "
                    f"→ tiempo medio={best['time_mean']:.3f}s, velocidad media={best['vel_mean']:.3g} pts/s.")
    # 2) Mejor workers por cada n (mínimo tiempo)
    for n in n_order:
        sub = agg[agg["n"] == n].dropna(subset=["workers"]).sort_values("time_mean")
        if len(sub) == 0:
            continue
        r0 = sub.iloc[0]
        eff_pct = 100 * r0["efficiency"] if pd.notna(r0["efficiency"]) else float("nan")
        insights.append(f"- Para n={int(n)}: mejor workers={int(r0['workers'])} → "
                        f"speedup={r0['speedup']:.2f}x, eficiencia={eff_pct:.0f}%.")
    # 3) Detección de rendimientos decrecientes (umbral eficiencia < 70%)
    bad = agg.dropna(subset=["efficiency"])
    bad = bad[(bad["workers"] > 1) & (bad["efficiency"] < 0.7)]
    if not bad.empty:
        ns = ", ".join(sorted({str(int(x)) for x in bad["n"].unique()}))
        insights.append(f"- Ojo: rendimientos decrecientes (eficiencia <70%) detectados en n ∈ {{{ns}}} para ciertos workers.")
    else:
        insights.append("- Buena escalabilidad: no se detectan eficiencias <70% en los datos agregados.")

    insights_html = "<ul>" + "".join([f"<li>{line}</li>" for line in insights]) + "</ul>"

    # Exporta CSV resumen
    agg.to_csv(out_csv, index=False)

    # Compone un único HTML con todas las figuras
    parts = []
    parts.append(fig_time.to_html(full_html=False, include_plotlyjs="cdn"))
    parts.append(fig_vel.to_html(full_html=False, include_plotlyjs=False))
    parts.append(fig_speedup.to_html(full_html=False, include_plotlyjs=False))
    parts.append(fig_eff.to_html(full_html=False, include_plotlyjs=False))
    parts.append(fig_err.to_html(full_html=False, include_plotlyjs=False))

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dashboard Montecarlo π · Ray</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, 'Helvetica Neue', Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-top: 0; }}
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
  <p class="muted">Este panel resume el rendimiento por <b>tamaño de muestra</b> (n) y <b>número de workers</b>. 
     Incluye <i>speedup</i> y <i>eficiencia</i> relativas al baseline por cada n.</p>

  <div class="grid">
    <div class="card">{parts[0]}</div>
    <div class="card">{parts[1]}</div>
    <div class="card">{parts[2]}</div>
    <div class="card">{parts[3]}</div>
    <div class="card">{parts[4]}</div>
  </div>

  <h2>Resumen numérico (agregado por n y workers)</h2>
  <div class="card">
    {table_html}
  </div>

  <h2>Conclusiones automáticas</h2>
  <div class="card">
    {insights_html}
  </div>

  <p class="muted">Generado automáticamente por <code>dashboard_montecarlo.py</code>.</p>
</body>
</html>"""
    out_html.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Genera un dashboard HTML de benchmarks Montecarlo π (Ray).")
    parser.add_argument("--inputs", nargs="*", type=str,
                        help="Ruta(s) de CSV de entrada. Si se omite, se usarán todos los 'resultados*.csv' del directorio actual.")
    parser.add_argument("--baseline-workers", type=int, default=1,
                        help="Workers de referencia para calcular speedup (por defecto 1). Si no existe, se usa el menor workers disponible por n.")
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
        raise SystemExit("No se han encontrado CSV de entrada. Pasa --inputs o coloca ficheros 'resultados*.csv' junto al script.")

    df = load_inputs(paths)
    agg = compute_aggregates(df, baseline_workers=args.baseline_workers)
    make_dashboard(df, agg, out_html=Path(args.out_html), out_csv=Path(args.out_csv))

    print(f"OK - Dashboard: {args.out_html}  ·  Resumen CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
