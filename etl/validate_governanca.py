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


def check_tv_unidade(df: pd.DataFrame) -> None:
    tem_abs    = "gov_tv_assinantes_abs" in df["metrica_codigo"].values
    tem_100hab = "gov_tv_100hab"         in df["metrica_codigo"].values
    if tem_100hab and tem_abs:
        ok("TV: gov_tv_assinantes_abs (legacy) + gov_tv_100hab presentes")
    elif tem_100hab:
        ok("TV: gov_tv_100hab presente")
    elif tem_abs:
        aviso("TV: apenas gov_tv_assinantes_abs — gov_tv_100hab não calculado "
              "(soc_censos_2021.parquet ausente durante o transform?)")
    else:
        aviso("TV: nenhuma métrica de TV encontrada")


def check_partido_vencedor() -> None:
    path = STAGING_DIR / "gov_partido_vencedor.parquet"
    if not path.exists():
        aviso("gov_partido_vencedor.parquet não encontrado")
        return
    df = pd.read_parquet(path)
    ausentes = set(MUNICIPIOS.keys()) - set(df["codigo_ine"].astype(str).unique())
    if ausentes:
        aviso(f"Partido vencedor: municípios em falta → {[MUNICIPIOS[c] for c in sorted(ausentes)]}")
    else:
        ok("Partido vencedor: todos os 11 municípios presentes")
    print("\n  Resultados autárquicas 2025:")
    for _, r in df.sort_values("nome").iterrows():
        print(f"    {r['nome']:25s} → {str(r['partido']):40s} [{r['categoria']}]")


def check_digital_anos(df: pd.DataFrame) -> None:
    metricas = [
        "gov_banda_larga_100hab", "gov_telefone_100hab",
        "gov_tv_assinantes_abs",  "gov_tv_100hab",
    ]
    for metrica in metricas:
        sub = df[df["metrica_codigo"] == metrica]
        if sub.empty:
            if metrica == "gov_tv_100hab" and "gov_tv_assinantes_abs" in df["metrica_codigo"].values:
                aviso(f"{metrica}: não calculado — soc_censos_2021.parquet estava ausente")
            continue
        anos_por_mun = sub.groupby("codigo_ine")["ano"].nunique()
        incompletos = anos_por_mun[anos_por_mun < 4].index.tolist()
        if incompletos:
            aviso(f"{metrica}: menos de 4 anos em {[MUNICIPIOS.get(c, c) for c in incompletos]}")
        else:
            ok(f"{metrica}: 4 anos por município")


def main() -> None:
    resetar_log()
    print("\n=== VALIDATE · Cluster 1 — Governança ===\n")

    path = STAGING_DIR / "gov_transformed.parquet"
    if not path.exists():
        erro("gov_transformed.parquet não encontrado"); return

    df = pd.read_parquet(path)
    print(f"  Carregados {len(df)} registos\n")

    print("[ Cobertura municipal ]")
    check_cobertura(df, "Governança")

    print("\n[ Nulos por métrica ]")
    nulos = check_nulos(df, "Governança")

    print("\n[ Outliers ]")
    outliers = check_outliers(df, "Governança")

    print("\n[ Scores normalizados ]")
    check_scores_normalizados(df, "Governança")

    print("\n[ Intervalos lógicos — métricas % ]")
    check_percentagens(df)

    print("\n[ Unidade TV ]")
    check_tv_unidade(df)

    print("\n[ Série temporal digital ]")
    check_digital_anos(df)

    print("\n[ Partido vencedor ]")
    check_partido_vencedor()

    print("\n[ Estatísticas por métrica ]")
    df_num = df[df["valor"].notna() & ~df["metrica_codigo"].isin(
        {"gov_partido_vencedor_cm", "gov_tv_assinantes_abs"}
    )]
    stats = df_num.groupby("metrica_codigo")["valor"].agg(
        ["count", "min", "max", "mean", "std"]
    ).round(3)
    print(stats.to_string())

    report_path = STAGING_DIR / "gov_quality_report.json"
    report = {
        "timestamp":    datetime.now().isoformat(),
        "cluster":      "Governança",
        "total_rows":   len(df),
        "n_metricas":   int(df["metrica_codigo"].nunique()),
        "n_municipios": int(df["codigo_ine"].nunique()),
        "avisos":       get_avisos(),
        "erros":        get_erros(),
        "nulos_pct":    nulos,
        "outliers":     outliers,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    imprimir_resumo_validacao(report_path)


if __name__ == "__main__":
    main()
