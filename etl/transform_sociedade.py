import numpy as np
import pandas as pd

from etl.utils import (
    STAGING_DIR, PT_CODIGO,
    normalizar_scores, enforce_schema,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_METRICAS_INVERTER = {
    "soc_tx_mortalidade",
    "soc_variacao_pop_2011_2021",
}


def registos_metrica(df_mun: pd.DataFrame, codigo: str, ano: int | None = None) -> pd.DataFrame:
    rows = []
    for _, row in df_mun.iterrows():
        rows.append({
            "codigo_ine":         row["codigo_ine"],
            "nome":               row["municipio"],   
            "metrica_codigo":     codigo,
            "ano":                ano if ano is not None else row.get("ano"),
            "valor":              row["valor"],
            "valor_normalizado":  None,
        })
    return pd.DataFrame(rows)


# ── Carregar staging ──────────────────────────────────────────────────────────

def carregar_staging() -> dict[str, pd.DataFrame]:
    return {
        "nados_vivos":    pd.read_parquet(STAGING_DIR / "soc_nados_vivos.parquet"),
        "obitos":         pd.read_parquet(STAGING_DIR / "soc_obitos.parquet"),
        "pop_estrangeira":pd.read_parquet(STAGING_DIR / "soc_pop_estrangeira.parquet"),
        "censos_2021":    pd.read_parquet(STAGING_DIR / "soc_censos_2021.parquet"),
        "saldo_natural":  pd.read_parquet(STAGING_DIR / "soc_saldo_natural.parquet"),
        "variacao_populacional": pd.read_parquet(STAGING_DIR / "soc_variacao_populacional.parquet"),
        "populacao_media_anual": pd.read_parquet(STAGING_DIR / "soc_populacao_media_anual.parquet"),
        "areas_2011":     pd.read_parquet(STAGING_DIR / "soc_areas_2011.parquet"),
    }


# ── Transformações ────────────────────────────────────────────────────────────

def calc_pop_total(stg: dict) -> pd.DataFrame:
    df = stg["censos_2021"].copy()
    df["metrica_codigo"]    = "soc_pop_total_cim"
    df["valor_normalizado"] = None
    return df.rename(columns={"municipio": "nome"})[
        ["codigo_ine", "nome", "metrica_codigo", "ano", "valor", "valor_normalizado"]
    ]


def calc_taxas_demograficas(stg: dict) -> pd.DataFrame:
    # Com a série anual (2021-2025) a população passou a variar por ano —
    # o join tem de ser por (codigo_ine, ano), não só por município, senão
    # cada nados-vivos/óbito ficaria multiplicado por todos os anos de
    # população disponíveis (linhas duplicadas).
    pop = (stg["censos_2021"][stg["censos_2021"]["codigo_ine"] != PT_CODIGO]
           [["codigo_ine", "ano", "valor"]]
           .rename(columns={"valor": "pop_2021"}))

    nv = stg["nados_vivos"].copy()
    ob = stg["obitos"].copy()

    # Merge nados-vivos com óbitos para anos comuns
    merged = nv.merge(
        ob[["codigo_ine", "ano", "valor"]].rename(columns={"valor": "obitos"}),
        on=["codigo_ine", "ano"], how="inner"
    ).rename(columns={"valor": "nados_vivos"})

    # Junta a população do mesmo ano (não a de 2021 fixa como antes) —
    # para anos sem população anual disponível (ex.: antes de 2021), a
    # taxa fica NULL em vez de usar um valor desatualizado.
    merged = merged.merge(pop, on=["codigo_ine", "ano"], how="left")
    merged = merged[merged["pop_2021"] > 0]

    records = []
    for _, row in merged.iterrows():
        cod  = row["codigo_ine"]
        mun  = row["municipio"]
        ano  = row["ano"]
        nv_  = row["nados_vivos"]
        ob_  = row["obitos"]
        pop_ = row["pop_2021"]

        for codigo, valor in [
            ("soc_tx_natalidade",  (nv_ / pop_ * 1000) if pd.notna(nv_) and pd.notna(pop_) else np.nan),
            ("soc_tx_mortalidade", (ob_ / pop_ * 1000) if pd.notna(ob_) and pd.notna(pop_) else np.nan),
        ]:
            records.append({
                "codigo_ine": cod, "nome": mun,
                "metrica_codigo": codigo, "ano": ano,
                "valor": round(valor, 4) if pd.notna(valor) else np.nan,
                "valor_normalizado": None,
            })

    return pd.DataFrame(records)


def calc_pct_estrangeira(stg: dict) -> pd.DataFrame:
    """soc_pct_pop_estrangeira — por ano (população do mesmo ano, 2021-2025)."""
    pop = (stg["censos_2021"][stg["censos_2021"]["codigo_ine"] != PT_CODIGO]
           [["codigo_ine", "ano", "valor"]]
           .rename(columns={"valor": "pop_2021"}))

    df = stg["pop_estrangeira"].copy()
    df = df.merge(pop, on=["codigo_ine", "ano"], how="left")
    df = df[df["pop_2021"] > 0]
    df["valor_calc"] = (df["valor"] / df["pop_2021"] * 100).round(4)

    records = []
    for _, row in df.iterrows():
        records.append({
            "codigo_ine":     row["codigo_ine"],
            "nome":           row["municipio"],
            "metrica_codigo": "soc_pct_pop_estrangeira",
            "ano":            row["ano"],
            "valor":          row["valor_calc"],
            "valor_normalizado": None,
        })
    return pd.DataFrame(records)


def calc_densidade(stg: dict) -> pd.DataFrame:
    pop_serie = stg["censos_2021"]
    pop_serie = pop_serie[pop_serie["codigo_ine"] != PT_CODIGO]

    areas = (stg["areas_2011"]
             .query("metrica == 'Area_km2'")
             .set_index("codigo_ine")["valor"]
             .rename("area_km2"))

    df = pop_serie.merge(areas, on="codigo_ine", how="left")
    df = df[df["area_km2"] > 0]
    df["valor"] = (df["valor"] / df["area_km2"]).round(2)

    records = []
    for _, row in df.iterrows():
        records.append({
            "codigo_ine":     row["codigo_ine"],
            "nome":           row["municipio"],
            "metrica_codigo": "soc_densidade_pop",
            "ano":            int(row["ano"]),
            "valor":          row["valor"],
            "valor_normalizado": None,
        })
    return pd.DataFrame(records)


def calc_variacao_pop(stg: dict) -> pd.DataFrame:
    # Métrica fixa "2011_2021" — usa sempre o ano 2021 da série anual, mesmo
    # que ANO_REFERENCIA_POPULACAO (usado noutros clusters como denominador
    # per capita) seja outro. Se não houver 2021 na série, não faz sentido
    # calcular esta métrica com outro ano (o nome ficaria enganador).
    serie_2021 = stg["censos_2021"]
    serie_2021 = serie_2021[(serie_2021["codigo_ine"] != PT_CODIGO) &
                             (serie_2021["ano"] == 2021)]

    pop2021 = (serie_2021
               .set_index("codigo_ine")[["municipio", "valor"]]
               .rename(columns={"valor": "pop_2021"}))

    pop2011 = (stg["areas_2011"]
               .query("metrica == 'Pop_2011'")
               .set_index("codigo_ine")["valor"]
               .rename("pop_2011"))

    df = pop2021.join(pop2011)
    df = df[df["pop_2011"] > 0]
    df["valor"] = ((df["pop_2021"] - df["pop_2011"]) / df["pop_2011"] * 100).round(4)
    df = df.reset_index()

    records = []
    for _, row in df.iterrows():
        records.append({
            "codigo_ine":     row["codigo_ine"],
            "nome":           row["municipio"],
            "metrica_codigo": "soc_variacao_pop_2011_2021",
            "ano":            2021,
            "valor":          row["valor"],
            "valor_normalizado": None,
        })
    return pd.DataFrame(records)


def calc_saldo_natural_direto(stg: dict) -> pd.DataFrame:
    df = stg["saldo_natural"].copy()
    df["metrica_codigo"]    = "soc_saldo_natural"
    df["valor_normalizado"] = None
    return df.rename(columns={"municipio": "nome"})[
        ["codigo_ine", "nome", "metrica_codigo", "ano", "valor", "valor_normalizado"]
    ]


def calc_variacao_populacional_anual(stg: dict) -> pd.DataFrame:
    df = stg["variacao_populacional"].copy()
    df["metrica_codigo"]    = "soc_variacao_populacional_anual"
    df["valor_normalizado"] = None
    return df.rename(columns={"municipio": "nome"})[
        ["codigo_ine", "nome", "metrica_codigo", "ano", "valor", "valor_normalizado"]
    ]


def calc_populacao_media_anual(stg: dict) -> pd.DataFrame:
    """soc_pop_media_anual — fonte direta INE. Sem Portugal nesta fonte."""
    df = stg["populacao_media_anual"].copy()
    df["metrica_codigo"]    = "soc_pop_media_anual"
    df["valor_normalizado"] = None
    return df.rename(columns={"municipio": "nome"})[
        ["codigo_ine", "nome", "metrica_codigo", "ano", "valor", "valor_normalizado"]
    ]


def calc_saldo_acumulado(stg: dict) -> pd.DataFrame:
    ANO_INI = 2019
    ANO_FIM = 2024
 
    nv = stg["nados_vivos"]
    ob = stg["obitos"]
 
    merged = nv.merge(
        ob[["codigo_ine", "ano", "valor"]].rename(columns={"valor": "obitos"}),
        on=["codigo_ine", "ano"], how="inner"
    ).rename(columns={"valor": "nados_vivos"})
 
    # Filtrar para o período de monitorização (pré-COVID até ao presente)
    merged = merged[(merged["ano"] >= ANO_INI) & (merged["ano"] <= ANO_FIM)]
 
    merged["saldo"] = merged["nados_vivos"] - merged["obitos"]
    acum = (merged
            .groupby(["codigo_ine", "municipio"])["saldo"]
            .sum()
            .reset_index()
            .rename(columns={"saldo": "valor"}))
 
    ano_min = int(merged["ano"].min())
    ano_max = int(merged["ano"].max())
 
    acum["metrica_codigo"]    = "soc_saldo_natural_acumulado"
    acum["ano"]               = ano_max
    acum["valor"]             = acum["valor"].round(0)
    acum["valor_normalizado"] = None
    acum["periodo_referencia"] = f"{ano_min}-{ano_max}"
 
    print(f"     Período de acumulação: {ano_min}–{ano_max}")
    return acum.rename(columns={"municipio": "nome"})[
        ["codigo_ine", "nome", "metrica_codigo", "ano",
         "valor", "valor_normalizado", "periodo_referencia"]
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== TRANSFORM · Cluster 6 — Sociedade ===\n")
    stg = carregar_staging()

    partes = []

    print("[6.1] Pop. residente CIM + Portugal (INE, série anual 2021-2025)")
    partes.append(calc_pop_total(stg))

    print("[6.2] Taxas demográficas (natalidade / mortalidade)")
    partes.append(calc_taxas_demograficas(stg))

    print("[6.2b] Saldo natural (fonte direta INE, com Portugal)")
    partes.append(calc_saldo_natural_direto(stg))

    print("[6.2c] Variação populacional anual (com Portugal)")
    partes.append(calc_variacao_populacional_anual(stg))

    print("[6.2d] População média anual")
    partes.append(calc_populacao_media_anual(stg))

    print("[6.3] % Pop. estrangeira")
    partes.append(calc_pct_estrangeira(stg))

    print("[6.4] Densidade populacional")
    partes.append(calc_densidade(stg))

    print("[6.5] Variação pop 2011–2021")
    partes.append(calc_variacao_pop(stg))

    print("[6.6] Saldo natural acumulado")
    partes.append(calc_saldo_acumulado(stg))

    df_all = pd.concat(partes, ignore_index=True)

    # Capturar o período de acumulação para diagnóstico antes de remover a coluna
    if "periodo_referencia" in df_all.columns:
        periodo = df_all[df_all["periodo_referencia"].notna()]["periodo_referencia"].iloc[0] \
                  if df_all["periodo_referencia"].notna().any() else "n/a"
        print(f"   Período saldo acumulado: {periodo}")

    df_all = df_all.drop(columns=["periodo_referencia"], errors="ignore")

    print("\n→ Normalizando métricas...")
    df_all = normalizar_scores(df_all, metricas_inverter=_METRICAS_INVERTER)
    df_all = enforce_schema(df_all)

    df_all.to_parquet(STAGING_DIR / "soc_transformed.parquet", index=False)

    print("\n✓ Transform concluído")
    print(f"  Total registos:  {len(df_all)}")
    print(f"  Métricas únicas: {df_all['metrica_codigo'].nunique()}")
    print(f"  Municípios:      {df_all['codigo_ine'].nunique()}")
    anos = sorted([int(a) for a in df_all["ano"].dropna().unique()])
    print(f"  Anos cobertos:   {anos}")
    print(f"\n  Métricas calculadas:")
    for m in sorted(df_all["metrica_codigo"].unique()):
        n = df_all[df_all["metrica_codigo"] == m]["codigo_ine"].nunique()
        print(f"    {m:45s} ({n} municípios)")


if __name__ == "__main__":
    main()
