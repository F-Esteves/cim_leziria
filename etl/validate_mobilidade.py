import json
from datetime import datetime

import pandas as pd

from etl.utils import (
    STAGING_DIR, MUNICIPIOS,
    aviso, erro, ok, resetar_log,
    get_avisos, get_erros,
    check_cobertura, check_nulos, check_outliers,
    check_scores_normalizados, check_percentagens,
    imprimir_resumo_validacao,
)

_PCT_ILIMITADAS = {
    "mob_evolucao_veiculos_pp",
    "mob_ve_crescimento_pct",
}


def check_valores_positivos(df: pd.DataFrame) -> None:
    for m in ["mob_registo_total_1000hab", "mob_registo_ligeiros_1000hab", "mob_ve_total"]:
        sub = df[df["metrica_codigo"] == m]
        if sub.empty:
            aviso(f"{m}: sem dados"); continue
        negativos = sub[sub["valor"] < 0]
        if not negativos.empty:
            erro(f"{m}: {len(negativos)} valores negativos")
        else:
            ok(f"{m}: valores positivos ✓")


def check_serie_veiculos(df: pd.DataFrame) -> None:
    """Veículos devem ter 4 anos (2021-2024) por município."""
    for metrica in ["mob_registo_total_1000hab", "mob_registo_ligeiros_1000hab"]:
        sub = df[df["metrica_codigo"] == metrica]
        if sub.empty:
            aviso(f"{metrica}: sem dados"); continue
        anos_por_mun = sub.groupby("codigo_ine")["ano"].nunique()
        incompletos  = anos_por_mun[anos_por_mun < 4].index.tolist()
        if incompletos:
            aviso(f"{metrica}: menos de 4 anos em {[MUNICIPIOS.get(c, c) for c in incompletos]}")
        else:
            ok(f"{metrica}: 4 anos por município")


def check_pontos_ve_consistencia(df: pd.DataFrame) -> None:
    pub  = df[df["metrica_codigo"] == "mob_ve_publicos_pct"]["valor"]
    priv = df[df["metrica_codigo"] == "mob_ve_privados_pct"]["valor"]
    if not pub.empty and not priv.empty:
        fora = ((pub.values + priv.values) < 99.0) | ((pub.values + priv.values) > 101.0)
        if fora.sum() > 0:
            aviso(f"mob_ve_publicos_pct + privados_pct: {fora.sum()} municípios com soma ≠ 100%")
        else:
            ok("mob_ve: público + privado = 100% ✓")
    semi = df[df["metrica_codigo"] == "mob_ve_semirrapidos_pct"]["valor"]
    rap  = df[df["metrica_codigo"] == "mob_ve_rapidos_pct"]["valor"]
    if not semi.empty and not rap.empty:
        ok("mob_ve: métricas de tipo de ponto presentes ✓")


def main() -> None:
    resetar_log()
    print("\n=== VALIDATE · Cluster 3 — Mobilidade ===\n")

    path = STAGING_DIR / "mob_transformed.parquet"
    if not path.exists():
        erro("mob_transformed.parquet não encontrado — corre transform primeiro"); return

    df = pd.read_parquet(path)
    print(f"  Carregados {len(df)} registos\n")

    print("[ Cobertura municipal ]")
    check_cobertura(df, "Mobilidade")

    print("\n[ Nulos por métrica ]")
    nulos = check_nulos(df, "Mobilidade")

    print("\n[ Outliers (Z-score > 3) ]")
    outliers = check_outliers(df, "Mobilidade")

    print("\n[ Scores normalizados ]")
    check_scores_normalizados(df, "Mobilidade")

    print("\n[ Valores positivos ]")
    check_valores_positivos(df)

    print("\n[ Percentagens bounded ]")
    check_percentagens(df, ilimitadas=_PCT_ILIMITADAS)

    print("\n[ Série temporal veículos ]")
    check_serie_veiculos(df)

    print("\n[ Consistência pontos VE ]")
    check_pontos_ve_consistencia(df)

    print("\n[ Estatísticas por métrica ]")
    stats = df.groupby("metrica_codigo")["valor"].agg(["count", "min", "max", "mean"]).round(3)
    print(stats.to_string())

    report_path = STAGING_DIR / "mob_quality_report.json"
    report = {
        "timestamp":    datetime.now().isoformat(),
        "cluster":      "Mobilidade",
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
