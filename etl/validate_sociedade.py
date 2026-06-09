import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from etl.utils import (
    STAGING_DIR,
    aviso, erro, ok, resetar_log,
    get_avisos, get_erros,
    check_cobertura, check_scores_normalizados,
    imprimir_resumo_validacao,
)

_METRICAS_ESPERADAS = {
    "soc_pop_total_cim", "soc_tx_natalidade", "soc_tx_mortalidade",
    "soc_saldo_natural", "soc_pct_pop_estrangeira", "soc_densidade_pop",
    "soc_variacao_pop_2011_2021", "soc_saldo_natural_acumulado",
}

_METRICAS_INVERTER = {
    "soc_tx_mortalidade",
    "soc_variacao_pop_2011_2021",
}

_RANGES = {
    "soc_pop_total_cim":           (1_000,   200_000),
    "soc_tx_natalidade":           (0,        30),
    "soc_tx_mortalidade":          (0,        30),
    "soc_saldo_natural":           (-500,     500),
    "soc_pct_pop_estrangeira":     (0,        50),
    "soc_densidade_pop":           (1,        500),
    "soc_variacao_pop_2011_2021":  (-30,      30),
    "soc_saldo_natural_acumulado": (-10_000,  10_000),
}

_SCHEMA_ESPERADO = {
    "codigo_ine", "nome", "ano", "metrica_codigo",
    "valor", "valor_normalizado", "valor_texto", "categoria",
}


def check_schema(df: pd.DataFrame) -> None:
    em_falta = _SCHEMA_ESPERADO - set(df.columns)
    extras   = set(df.columns) - _SCHEMA_ESPERADO
    if em_falta:
        erro(f"Colunas em falta no schema: {sorted(em_falta)}")
    else:
        ok("Schema alinhado com Governança (referência)")
    for col in sorted(extras):
        aviso(f"Coluna extra no schema: {col}")


def check_metricas(df: pd.DataFrame) -> None:
    presentes = set(df["metrica_codigo"].unique())
    em_falta  = _METRICAS_ESPERADAS - presentes
    if em_falta:
        erro(f"Métricas esperadas em falta: {sorted(em_falta)}")
    else:
        ok("8 métricas esperadas presentes")
    for m in sorted(presentes - _METRICAS_ESPERADAS):
        aviso(f"Métrica extra inesperada: {m}")


def check_nulos(df: pd.DataFrame) -> dict:
    result: dict[str, float] = {}
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


def check_outliers_soc(df: pd.DataFrame) -> list[str]:
    suspeitos: list[str] = []
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


def check_ranges(df: pd.DataFrame) -> None:
    fora_total = 0
    for met, (lo, hi) in _RANGES.items():
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
    for met in _METRICAS_INVERTER:
        sub = df[(df["metrica_codigo"] == met) & df["valor_normalizado"].notna()]
        if sub.empty:
            continue
        idx_max_val = sub["valor"].idxmax()
        score_pior  = sub.loc[idx_max_val, "valor_normalizado"]
        mun_pior    = sub.loc[idx_max_val, "nome"]
        if score_pior > 0.1:
            erro(f"{met}: inversão incorrecta — {mun_pior} tem valor mais alto "
                 f"({sub.loc[idx_max_val,'valor']:.2f}) mas score={score_pior:.3f} (esperado ≈ 0)")
        else:
            ok(f"{met}: inversão correcta ({mun_pior} = pior, score={score_pior:.3f})")


def check_desfasamento_anos(df: pd.DataFrame) -> None:
    anos_max    = df.groupby("metrica_codigo")["ano"].max()
    ano_ref     = anos_max.max()
    desfasadas  = anos_max[anos_max < ano_ref - 1]
    if not desfasadas.empty:
        for met, ano in desfasadas.items():
            aviso(f"{met}: último ano={ano} vs referência={ano_ref} ({ano_ref - ano} anos de desfasamento)")
    else:
        ok("Anos máximos consistentes entre métricas (diferença ≤ 1 ano)")


def main() -> None:
    resetar_log()
    print("\n=== VALIDATE · Cluster 6 — Sociedade ===\n")

    path = STAGING_DIR / "soc_transformed.parquet"
    if not path.exists():
        erro("soc_transformed.parquet não encontrado — corre transform primeiro"); return

    df = pd.read_parquet(path)
    print(f"  Carregados {len(df)} registos\n")

    print("[ Schema ]");              check_schema(df)
    print("\n[ Cobertura municipal ]"); check_cobertura(df, "Sociedade")
    print("\n[ Métricas ]");           check_metricas(df)
    print("\n[ Nulos por métrica ]");  nulos = check_nulos(df)
    print("\n[ Nulos no último ano ]"); check_nulos_ultimo_ano(df)
    print("\n[ Outliers (Z-score > 3) ]"); outliers = check_outliers_soc(df)
    print("\n[ Scores normalizados ]"); check_scores_normalizados(df, "Sociedade")
    print("\n[ Intervalos plausíveis ]"); check_ranges(df)
    print("\n[ Duplicados ]");          check_duplicados(df)
    print("\n[ Consistência de anos ]"); check_desfasamento_anos(df)
    print("\n[ Inversão de métricas ]"); check_inversao(df)

    print("\n[ Estatísticas por métrica ]")
    stats = df.groupby("metrica_codigo")["valor"].agg(["count", "min", "max", "mean", "std"]).round(3)
    print(stats.to_string())

    report_path = STAGING_DIR / "soc_quality_report.json"
    report = {
        "timestamp":    datetime.now().isoformat(),
        "cluster":      "Sociedade",
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

    report_path_str = report_path
    if get_erros():
        print(f"\n{'='*50}")
        print(f"  ✗ {len(get_erros())} erro(s) — corrigir antes do load")
        print(f"  Relatório: {report_path_str}")
        sys.exit(1)
    else:
        imprimir_resumo_validacao(report_path_str)


if __name__ == "__main__":
    main()
