import json
from datetime import datetime

import pandas as pd

from etl.utils import (
    STAGING_DIR, MUNICIPIOS,
    aviso, erro, ok, resetar_log,
    get_avisos, get_erros,
    check_outliers, check_scores_normalizados, check_percentagens,
    imprimir_resumo_validacao,
)

_PCT_ILIMITADAS = {
    "mdv_evolucao_criminalidade_pp",
    "mdv_evolucao_dormidas_pp",
}

_METRICAS_ULS = {
    "mdv_utentes_csp", "mdv_pct_utentes_mdf",
    "mdv_consultas_presenciais", "mdv_consultas_total",
}

# Métricas onde cobertura esperada < 11 municípios (justificada)
_COBERTURA_ESPERADA: dict[str, int] = {
    "mdv_acidentes_vitimas_1000hab": 10,
}


def check_cobertura_mdv(df: pd.DataFrame) -> None:
    """Verifica cobertura municipal por métrica, respeitando excepções conhecidas."""
    for metrica in sorted(df["metrica_codigo"].unique()):
        n_esp    = _COBERTURA_ESPERADA.get(metrica, 11)
        sub      = df[df["metrica_codigo"] == metrica]
        n_mun    = sub["codigo_ine"].nunique()
        ausentes = set(MUNICIPIOS.keys()) - set(sub["codigo_ine"].astype(str).unique())
        if n_mun < n_esp:
            aviso(f"{metrica}: {n_mun}/{n_esp} municípios "
                  f"(ausentes: {[MUNICIPIOS[c] for c in sorted(ausentes)]})")
        elif n_mun == 11:
            ok(f"{metrica}: 11/11 municípios ✓")
        else:
            ok(f"{metrica}: {n_mun}/{n_esp} municípios ✓ (cobertura esperada)")


def check_nulos_mdv(df: pd.DataFrame) -> dict:
    result: dict[str, float] = {}
    for metrica, grupo in df.groupby("metrica_codigo"):
        pct = grupo["valor"].isna().mean() * 100
        result[metrica] = round(pct, 1)
        if pct > 50:
            aviso(f"{metrica}: {pct:.0f}% nulos")
    return result


def check_valores_positivos(df: pd.DataFrame) -> None:
    for m in ["mdv_hab_medico", "mdv_dormidas_100hab",
              "mdv_acidentes_vitimas_1000hab", "mdv_criminalidade_total"]:
        sub = df[df["metrica_codigo"] == m]
        if sub.empty:
            aviso(f"{m}: sem dados"); continue
        neg = sub[sub["valor"] < 0]
        if not neg.empty:
            erro(f"{m}: {len(neg)} valores negativos")
        else:
            ok(f"{m}: valores positivos ✓")


def check_anos_csp(df: pd.DataFrame) -> None:
    """CSP têm dados 2015-2025; interessa 2021-2025 para o dashboard."""
    for m in _METRICAS_ULS:
        sub  = df[df["metrica_codigo"] == m]
        anos = sorted(sub["ano"].unique())
        if not anos:
            aviso(f"{m}: sem dados")
        elif max(anos) < 2024:
            aviso(f"{m}: dados mais recentes em {max(anos)} (esperado ≥ 2024)")
        else:
            ok(f"{m}: anos {anos[0]}–{anos[-1]} ✓")


def main() -> None:
    resetar_log()
    print("\n=== VALIDATE · Cluster 4 — Modos de Vida ===\n")

    path = STAGING_DIR / "mdv_transformed.parquet"
    if not path.exists():
        erro("mdv_transformed.parquet não encontrado — corre transform primeiro"); return

    df = pd.read_parquet(path)
    print(f"  Carregados {len(df)} registos · "
          f"{df['metrica_codigo'].nunique()} métricas · "
          f"{df['codigo_ine'].nunique()} municípios\n")

    print("[ Cobertura municipal por métrica ]")
    check_cobertura_mdv(df)

    print("\n[ Nulos por métrica ]")
    nulos = check_nulos_mdv(df)

    print("\n[ Outliers (Z-score > 3) ]")
    outliers = check_outliers(df, "Modos de Vida")

    print("\n[ Scores normalizados ]")
    check_scores_normalizados(df, "Modos de Vida")

    print("\n[ Percentagens bounded ]")
    check_percentagens(df, ilimitadas=_PCT_ILIMITADAS)

    print("\n[ Valores positivos ]")
    check_valores_positivos(df)

    print("\n[ Séries temporais CSP ]")
    check_anos_csp(df)

    print("\n[ Estatísticas por métrica ]")
    stats = df.groupby("metrica_codigo")["valor"].agg(["count", "min", "max", "mean"]).round(3)
    print(stats.to_string())

    report_path = STAGING_DIR / "mdv_quality_report.json"
    report = {
        "timestamp":    datetime.now().isoformat(),
        "cluster":      "Modos de Vida",
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
