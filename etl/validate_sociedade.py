import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime

STAGING_DIR = Path("data/staging")

MUNICIPIOS = {
    "1403": "Almeirim",        "1404": "Alpiarça",
    "1103": "Azambuja",        "1405": "Benavente",
    "1406": "Cartaxo",         "1407": "Chamusca",
    "1409": "Coruche",         "1412": "Golegã",
    "1414": "Rio Maior",       "1415": "Salvaterra de Magos",
    "1416": "Santarém",
}

METRICAS_ESPERADAS = {
    "soc_pop_total_cim",
    "soc_tx_natalidade",
    "soc_tx_mortalidade",
    "soc_saldo_natural",
    "soc_pct_pop_estrangeira",
    "soc_densidade_pop",
    "soc_variacao_pop_2011_2021",
    "soc_saldo_natural_acumulado",
}

METRICAS_INVERTER = {
    "soc_tx_mortalidade",
    "soc_variacao_pop_2011_2021",
}

RANGES = {
    "soc_pop_total_cim":           (1_000,   200_000),
    "soc_tx_natalidade":           (0,        30),
    "soc_tx_mortalidade":          (0,        30),
    "soc_saldo_natural":           (-500,     500),
    "soc_pct_pop_estrangeira":     (0,        50),
    "soc_densidade_pop":           (1,        500),
    "soc_variacao_pop_2011_2021":  (-30,      30),
    "soc_saldo_natural_acumulado": (-10_000,  10_000),
}

AVISOS: list[str] = []
ERROS:  list[str] = []


def aviso(msg: str) -> None:
    print(f"  ⚠  {msg}")
    AVISOS.append(msg)


def erro(msg: str) -> None:
    print(f"  ✗  {msg}")
    ERROS.append(msg)


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def check_schema(df: pd.DataFrame) -> None:
    schema_esperado = {"codigo_ine", "nome", "ano", "metrica_codigo",
                       "valor", "valor_normalizado", "valor_texto", "categoria"}
    presentes = set(df.columns)
    em_falta  = schema_esperado - presentes
    extras    = presentes - schema_esperado
    if em_falta:
        erro(f"Colunas em falta no schema: {sorted(em_falta)}")
    else:
        ok("Schema alinhado com Governança (referência)")
    for col in sorted(extras):
        aviso(f"Coluna extra no schema: {col}")


def check_cobertura(df: pd.DataFrame) -> None:
    presentes = set(df["codigo_ine"].astype(str).unique())
    ausentes  = set(MUNICIPIOS.keys()) - presentes
    if ausentes:
        nomes = [MUNICIPIOS[c] for c in sorted(ausentes)]
        aviso(f"Municípios em falta → {nomes}")
    else:
        ok("Todos os 11 municípios presentes")


def check_metricas(df: pd.DataFrame) -> None:
    presentes = set(df["metrica_codigo"].unique())
    em_falta  = METRICAS_ESPERADAS - presentes
    if em_falta:
        erro(f"Métricas esperadas em falta: {sorted(em_falta)}")
    else:
        ok("8 métricas esperadas presentes")
    for m in sorted(presentes - METRICAS_ESPERADAS):
        aviso(f"Métrica extra inesperada: {m}")


def check_nulos(df: pd.DataFrame) -> dict:
    result = {}
    for metrica, grupo in df.groupby("metrica_codigo"):
        pct = grupo["valor"].isna().mean() * 100
        result[metrica] = round(pct, 1)
        if pct > 50:
            aviso(f"{metrica}: {pct:.0f}% nulos")
        elif pct > 0:
            print(f"     ℹ  {metrica}: {pct:.0f}% nulos")
    return result


def check_nulos_ultimo_ano(df: pd.DataFrame) -> None:
    nulos_total = 0
    for met, grp in df.groupby("metrica_codigo"):
        ult_ano  = grp["ano"].max()
        snapshot = grp[grp["ano"] == ult_ano]
        nulos    = snapshot["valor"].isna().sum()
        if nulos > 0:
            erro(f"NaN em {met} (ano={ult_ano}): {nulos} registos")
            nulos_total += nulos
    if nulos_total == 0:
        ok("Sem valores nulos no último ano de cada métrica")


def check_outliers(df: pd.DataFrame) -> list:
    suspeitos = []
    for (metrica, ano), grupo in df.groupby(["metrica_codigo", "ano"]):
        vals = grupo["valor"].dropna()
        if len(vals) < 4:
            continue
        std = vals.std()
        if std == 0:
            continue
        z = np.abs((vals - vals.mean()) / std)
        for idx in z[z > 3].index:
            row = df.loc[idx]
            msg = f"{metrica} · {row['nome']} · {ano}: {row['valor']:.2f} (Z>3)"
            aviso(msg)
            suspeitos.append(msg)
    return suspeitos


def check_scores(df: pd.DataFrame) -> None:
    df_num = df[df["valor_normalizado"].notna()]
    fora   = df_num[(df_num["valor_normalizado"] < 0) | (df_num["valor_normalizado"] > 1)]
    if not fora.empty:
        erro(f"{len(fora)} scores fora de [0,1]")
    else:
        ok("Todos os scores normalizados em [0,1]")


def check_ranges(df: pd.DataFrame) -> None:
    fora_total = 0
    for met, (lo, hi) in RANGES.items():
        sub  = df[df["metrica_codigo"] == met]["valor"].dropna()
        fora = sub[(sub < lo) | (sub > hi)]
        if len(fora) > 0:
            aviso(f"{met}: {len(fora)} valores fora de [{lo}, {hi}] (ex: {fora.values[:3]})")
            fora_total += len(fora)
    if fora_total == 0:
        ok("Todos os valores dentro dos intervalos plausíveis")


def check_duplicados(df: pd.DataFrame) -> None:
    dups = df.duplicated(subset=["codigo_ine", "metrica_codigo", "ano"]).sum()
    if dups > 0:
        erro(f"Duplicados (codigo_ine, metrica_codigo, ano): {dups}")
    else:
        ok("Sem duplicados")


def check_inversao(df: pd.DataFrame) -> None:
    for met in METRICAS_INVERTER:
        sub = df[(df["metrica_codigo"] == met) & df["valor_normalizado"].notna()]
        if sub.empty:
            continue
        idx_max_val  = sub["valor"].idxmax()
        score_pior   = sub.loc[idx_max_val, "valor_normalizado"]
        mun_pior     = sub.loc[idx_max_val, "nome"]
        if score_pior > 0.1:
            erro(f"{met}: inversão incorrecta — {mun_pior} tem valor mais alto "
                 f"({sub.loc[idx_max_val,'valor']:.2f}) mas score={score_pior:.3f} (esperado ≈ 0)")
        else:
            ok(f"{met}: inversão correcta ({mun_pior} = pior, score={score_pior:.3f})")


def check_desfasamento_anos(df: pd.DataFrame) -> None:
    anos_max = df.groupby("metrica_codigo")["ano"].max()
    ano_ref  = anos_max.max()
    desfasadas = anos_max[anos_max < ano_ref - 1]
    if not desfasadas.empty:
        for met, ano in desfasadas.items():
            aviso(f"{met}: último ano={ano} vs referência={ano_ref} ({ano_ref - ano} anos de desfasamento)")
    else:
        ok("Anos máximos consistentes entre métricas (diferença ≤ 1 ano)")


def main() -> None:
    print("\n=== VALIDATE · Cluster 6 — Sociedade ===\n")

    path = STAGING_DIR / "soc_transformed.parquet"
    if not path.exists():
        erro("soc_transformed.parquet não encontrado — corre transform primeiro")
        return

    df = pd.read_parquet(path)
    print(f"  Carregados {len(df)} registos\n")

    print("[ Schema ]")
    check_schema(df)

    print("\n[ Cobertura municipal ]")
    check_cobertura(df)

    print("\n[ Métricas ]")
    check_metricas(df)

    print("\n[ Nulos por métrica ]")
    nulos = check_nulos(df)

    print("\n[ Nulos no último ano ]")
    check_nulos_ultimo_ano(df)

    print("\n[ Outliers (Z-score > 3) ]")
    outliers = check_outliers(df)

    print("\n[ Scores normalizados ]")
    check_scores(df)

    print("\n[ Intervalos plausíveis ]")
    check_ranges(df)

    print("\n[ Duplicados ]")
    check_duplicados(df)

    print("\n[ Consistência de anos ]")
    check_desfasamento_anos(df)

    print("\n[ Inversão de métricas ]")
    check_inversao(df)

    print("\n[ Estatísticas por métrica ]")
    stats = (
        df.groupby("metrica_codigo")["valor"]
        .agg(["count", "min", "max", "mean", "std"])
        .round(3)
    )
    print(stats.to_string())

    report = {
        "timestamp":    datetime.now().isoformat(),
        "cluster":      "Sociedade",
        "total_rows":   len(df),
        "n_metricas":   int(df["metrica_codigo"].nunique()),
        "n_municipios": int(df["codigo_ine"].nunique()),
        "avisos":       AVISOS,
        "erros":        ERROS,
        "nulos_pct":    nulos,
        "outliers":     outliers,
        "stats":        stats.reset_index().to_dict(orient="records"),
    }

    report_path = STAGING_DIR / "soc_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    if ERROS:
        print(f"  ✗ {len(ERROS)} erro(s) — corrigir antes do load")
        sys.exit(1)
    elif AVISOS:
        print(f"  ⚠ {len(AVISOS)} aviso(s) — pode prosseguir com cautela")
    else:
        print("  ✓ Sem problemas — pronto para load")
    print(f"  Relatório: data/staging/soc_quality_report.json")


if __name__ == "__main__":
    main()
