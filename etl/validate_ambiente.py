import json
from datetime import datetime

import pandas as pd

from etl.utils import (
    STAGING_DIR,
    aviso, erro, ok, resetar_log,
    get_avisos, get_erros,
    check_cobertura, check_nulos, check_outliers,
    check_scores_normalizados, check_percentagens,
    imprimir_resumo_validacao,
)

_PCT_ILIMITADAS = {
    "amb_var_consumo_anual_pct",
    "amb_tcma_consumo_pct",
    "amb_evolucao_valorizacao_pp",
}


def check_valores_positivos(df: pd.DataFrame) -> None:
    for m in [
        "amb_consumo_total_kwh", "amb_consumo_bt_kwh", "amb_consumo_at_kwh",
        "amb_n_cpes_total", "amb_residuos_total_ton",
    ]:
        sub = df[df["metrica_codigo"] == m]
        if sub.empty:
            aviso(f"{m}: sem dados"); continue
        negativos = sub[sub["valor"] < 0]
        if not negativos.empty:
            erro(f"{m}: {len(negativos)} valores negativos")
        else:
            ok(f"{m}: valores positivos ✓")


def check_anos_completos(df: pd.DataFrame) -> None:
    anos_incompletos = {2020, 2025}
    for m in ["amb_consumo_total_kwh", "amb_consumo_bt_kwh", "amb_consumo_at_kwh"]:
        sub = df[df["metrica_codigo"] == m]
        presentes = set(sub["ano"].astype(int).unique())
        incompletos_presentes = presentes & anos_incompletos
        if incompletos_presentes:
            aviso(f"{m}: anos incompletos {sorted(incompletos_presentes)} — TCMA pode estar distorcida")
        else:
            ok(f"{m}: sem anos incompletos")


def main() -> None:
    resetar_log()
    print("\n=== VALIDATE · Cluster 2 — Ambiente ===\n")

    path = STAGING_DIR / "amb_transformed.parquet"
    if not path.exists():
        erro("amb_transformed.parquet não encontrado — corre transform primeiro"); return

    df = pd.read_parquet(path)
    print(f"  Carregados {len(df)} registos\n")

    print("[ Cobertura municipal ]")
    check_cobertura(df, "Ambiente")

    print("\n[ Nulos por métrica ]")
    nulos = check_nulos(df, "Ambiente")

    print("\n[ Outliers (Z-score > 3) ]")
    outliers = check_outliers(df, "Ambiente")

    print("\n[ Scores normalizados ]")
    check_scores_normalizados(df, "Ambiente")

    print("\n[ Valores positivos ]")
    check_valores_positivos(df)

    print("\n[ Percentagens bounded ]")
    check_percentagens(df, ilimitadas=_PCT_ILIMITADAS)

    print("\n[ Anos incompletos em métricas anuais ]")
    check_anos_completos(df)

    print("\n[ Estatísticas por métrica ]")
    stats = df.groupby("metrica_codigo")["valor"].agg(["count", "min", "max", "mean"]).round(2)
    print(stats.to_string())

    report_path = STAGING_DIR / "amb_quality_report.json"
    report = {
        "timestamp":    datetime.now().isoformat(),
        "cluster":      "Ambiente",
        "total_rows":   len(df),
        "n_metricas":   int(df["metrica_codigo"].nunique()),
        "n_municipios": int(df["codigo_ine"].nunique()),
        "avisos":       get_avisos(),
        "erros":        get_erros(),
        "nulos_pct":    nulos,
        "outliers":     outliers,
        "stats":        stats.reset_index().to_dict(orient="records"),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    imprimir_resumo_validacao(report_path)


if __name__ == "__main__":
    main()
