import pandas as pd
import numpy as np
from pathlib import Path

STAGING_DIR = Path("data/staging")

MUNICIPIOS = {
    "1403": "Almeirim",        "1404": "Alpiarça",
    "1103": "Azambuja",        "1405": "Benavente",
    "1406": "Cartaxo",         "1407": "Chamusca",
    "1409": "Coruche",         "1412": "Golegã",
    "1414": "Rio Maior",       "1415": "Salvaterra de Magos",
    "1416": "Santarém",
}
N_MUN = len(MUNICIPIOS)


# ── Normalização ───────────────────────────────────────────────

def normalizar(series: pd.Series, inverter: bool = False) -> pd.Series:
    """Min-max sobre os valores disponíveis. inverter=True → menor é melhor."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    norm = (series - mn) / (mx - mn)
    return 1 - norm if inverter else norm


# ── Métricas de eleições ───────────────────────────────────────

def calcular_metricas_eleicoes(df: pd.DataFrame, sufixo: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        base = {"codigo_ine": row["codigo_ine"], "nome": row["nome"], "ano": row["ano"]}
        e = safe_num(row.get("eleitores"))
        v = safe_num(row.get("votantes"))
        a = safe_num(row.get("abstencao"))

        if a is None:
            continue

        if e is not None and e > 100:
            
            abst_pct = round(a / e * 100, 2)
            part_pct = round(v / e * 100, 2) if v is not None else round(100 - abst_pct, 2)
        elif 0.0 <= a <= 100.0:
            
            abst_pct = round(a, 2)
            part_pct = round(v, 2) if (v is not None and 0.0 <= v <= 100.0) else round(100 - abst_pct, 2)
        else:
            continue

        if not (0 <= abst_pct <= 100 and 0 <= part_pct <= 100):
            continue

        rows.append({**base, "metrica_codigo": f"gov_abstencao_{sufixo}_pct",    "valor": abst_pct})
        rows.append({**base, "metrica_codigo": f"gov_participacao_{sufixo}_pct", "valor": part_pct})

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["codigo_ine", "nome", "ano", "metrica_codigo", "valor"])


def safe_num(v) -> float | None:
    """Converte para float; devolve None se NaN/None/não-numérico."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def calcular_evolucao_abstencao(df_ar: pd.DataFrame) -> pd.DataFrame:
    """Evolução abstenção AR: último ano disponível - primeiro ano disponível."""
    rows = []
    df_abst = df_ar[df_ar["metrica_codigo"] == "gov_abstencao_ar_pct"].copy()
    if df_abst.empty:
        return pd.DataFrame()

    for cod in df_abst["codigo_ine"].unique():
        serie = df_abst[df_abst["codigo_ine"] == cod].sort_values("ano")
        if len(serie) >= 2:
            ini = serie.iloc[0]
            fim = serie.iloc[-1]
            rows.append({
                "codigo_ine":     cod,
                "nome":           ini["nome"],
                "ano":            int(fim["ano"]),
                "metrica_codigo": "gov_evolucao_abstencao_pp",
                "valor":          round(fim["valor"] - ini["valor"], 2),
            })
    return pd.DataFrame(rows)


# ── Métricas de resultados ─────────────────────────────────────

def calcular_metricas_resultados(df_result: pd.DataFrame) -> pd.DataFrame:
    """
    A partir dos resultados autárquicas 2025, calcula:
    - partido vencedor por município (texto)
    - % municípios CIM por categoria de partido
    """
    rows = []
    ano = int(df_result["ano"].iloc[0]) if not df_result.empty else 2025

    # Partido vencedor por município (métrica de texto)
    for _, r in df_result.iterrows():
        rows.append({
            "codigo_ine":     r["codigo_ine"],
            "nome":           r["nome"],
            "ano":            ano,
            "metrica_codigo": "gov_partido_vencedor_cm",
            "valor":          None,
            "valor_texto":    r["partido"],
            "categoria":      r["categoria"],
        })

    # % municípios CIM por categoria
    n_total = len(df_result)
    if n_total > 0:
        contagens = df_result["categoria"].value_counts()
        for _, r in df_result.iterrows():
            cat = r["categoria"]
            pct = round(contagens.get(cat, 0) / n_total * 100, 1)
            rows.append({
                "codigo_ine":     r["codigo_ine"],
                "nome":           r["nome"],
                "ano":            ano,
                "metrica_codigo": "gov_pct_mun_partido",
                "valor":          pct,
                "valor_texto":    cat,
                "categoria":      cat,
            })

    return pd.DataFrame(rows)


# ── Métricas digitais ──────────────────────────────────────────

def calcular_metricas_digital(df_bl, df_tel, df_tv, pop_total: dict) -> pd.DataFrame:
    rows = []

    for df_src, codigo in [
        (df_bl,  "gov_banda_larga_100hab"),
        (df_tel, "gov_telefone_100hab"),
    ]:
        for _, row in df_src.iterrows():
            v = safe_num(row.get("valor"))
            if v is not None:
                rows.append({
                    "codigo_ine":     row["codigo_ine"],
                    "nome":           row["nome"],
                    "ano":            row["ano"],
                    "metrica_codigo": codigo,
                    "valor":          round(v, 4),
                })

    for _, row in df_tv.iterrows():
        v = safe_num(row.get("valor"))
        if v is not None:
            cod = row["codigo_ine"]

            rows.append({
                "codigo_ine":     cod,
                "nome":           row["nome"],
                "ano":            row["ano"],
                "metrica_codigo": "gov_tv_assinantes_abs",
                "valor":          round(v, 4),
            })
            pop = pop_total.get(cod)
            if pop and pop > 0:
                rows.append({
                    "codigo_ine":     cod,
                    "nome":           row["nome"],
                    "ano":            row["ano"],
                    "metrica_codigo": "gov_tv_100hab",
                    "valor":          round(v / pop * 100, 4),
                })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["codigo_ine", "nome", "ano", "metrica_codigo", "valor"])

    # TCMA banda larga: ((BL_último / BL_primeiro)^(1/(n-1)) - 1) * 100
    if not df_bl.empty:
        pivot = df_bl.pivot_table(index="codigo_ine", columns="ano", values="valor", aggfunc="first")
        anos_disp = sorted(pivot.columns)
        if len(anos_disp) >= 2:
            ano_ini, ano_fim = anos_disp[0], anos_disp[-1]
            n_anos = ano_fim - ano_ini
            for cod in pivot.index:
                bl_ini = safe_num(pivot.loc[cod].get(ano_ini))
                bl_fim = safe_num(pivot.loc[cod].get(ano_fim))
                if bl_ini and bl_fim and bl_ini > 0 and n_anos > 0:
                    tcma = ((bl_fim / bl_ini) ** (1 / n_anos) - 1) * 100
                    nome = df_bl[df_bl["codigo_ine"] == cod]["nome"].iloc[0]
                    df = pd.concat([df, pd.DataFrame([{
                        "codigo_ine":     cod,
                        "nome":           nome,
                        "ano":            ano_fim,
                        "metrica_codigo": "gov_tcma_banda_larga_pct",
                        "valor":          round(tcma, 4),
                    }])], ignore_index=True)

    return df


# ── Normalização final ─────────────────────────────────────────

METRICAS_INVERTER = {
    "gov_abstencao_ar_pct",
    "gov_abstencao_aut_pct",
    "gov_abstencao_pres_pct",
    "gov_evolucao_abstencao_pp",
}


METRICAS_SEM_NORMALIZACAO = {
    "gov_partido_vencedor_cm",
    "gov_tv_assinantes_abs",   
}


def normalizar_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["valor_normalizado"] = np.nan

    for (metrica, ano), grupo in df.groupby(["metrica_codigo", "ano"]):
        if metrica in METRICAS_SEM_NORMALIZACAO:
            continue
        vals = grupo["valor"].dropna()
        if len(vals) < 2:
            df.loc[grupo.index, "valor_normalizado"] = 0.5
            continue
        inv = metrica in METRICAS_INVERTER
        df.loc[grupo.index, "valor_normalizado"] = normalizar(
            df.loc[grupo.index, "valor"], inverter=inv
        ).values

    return df


# ── Main ───────────────────────────────────────────────────────

def main():
    print("\n=== TRANSFORM · Cluster 1 — Governança ===\n")

    print("[ 1.1 ] Participação Cívica")
    df_ar     = pd.read_parquet(STAGING_DIR / "gov_eleicoes_ar.parquet")
    df_aut    = pd.read_parquet(STAGING_DIR / "gov_eleicoes_autarquias.parquet")
    df_pres   = pd.read_parquet(STAGING_DIR / "gov_eleicoes_presidenciais.parquet")
    df_result = pd.read_parquet(STAGING_DIR / "gov_resultados_autarquias.parquet")

    metr_ar    = calcular_metricas_eleicoes(df_ar,   "ar")
    metr_aut   = calcular_metricas_eleicoes(df_aut,  "aut")
    metr_pres  = calcular_metricas_eleicoes(df_pres, "pres")
    evolucao   = calcular_evolucao_abstencao(metr_ar)
    resultados = calcular_metricas_resultados(df_result)

    print(f"     AR:            {len(metr_ar)} registos")
    print(f"     Autarquias:    {len(metr_aut)} registos")
    print(f"     Presidenciais: {len(metr_pres)} registos")
    print(f"     Evolução:      {len(evolucao)} registos")
    print(f"     Resultados:    {len(resultados)} registos")

    print("\n[ 1.2 ] Digital")
    soc_path = STAGING_DIR / "soc_censos_2021.parquet"
    if soc_path.exists():
        df_pop = pd.read_parquet(soc_path)
        pop_total = df_pop.set_index("codigo_ine")["valor"].to_dict()
        print(f"     Pop Censos 2021 carregada — {len(pop_total)} municípios")
    else:
        pop_total = {}
        print("     ⚠  soc_censos_2021.parquet não encontrado — gov_tv_100hab não será calculado")
        print("        Corre extract_sociedade.py primeiro")

    df_bl  = pd.read_parquet(STAGING_DIR / "gov_banda_larga.parquet")
    df_tel = pd.read_parquet(STAGING_DIR / "gov_telefone.parquet")
    df_tv  = pd.read_parquet(STAGING_DIR / "gov_tv.parquet")
    digital = calcular_metricas_digital(df_bl, df_tel, df_tv, pop_total)
    print(f"     Digital: {len(digital)} registos")

    df_numericas = pd.concat([metr_ar, metr_aut, metr_pres, evolucao, digital],
                             ignore_index=True)

    for col in ("valor_texto", "categoria"):
        if col not in df_numericas.columns:
            df_numericas[col] = None
        if col not in resultados.columns:
            resultados[col] = None

    df_all = pd.concat([df_numericas, resultados], ignore_index=True)
    df_all = normalizar_scores(df_all)

    df_all.to_parquet(STAGING_DIR / "gov_transformed.parquet", index=False)

    df_result.to_parquet(STAGING_DIR / "gov_partido_vencedor.parquet", index=False)

    print(f"\n✓ Transform concluído")
    print(f"  Total registos:  {len(df_all)}")
    print(f"  Métricas únicas: {df_all['metrica_codigo'].nunique()}")
    print(f"  Municípios:      {df_all['codigo_ine'].nunique()}")
    anos = sorted([int(a) for a in df_all["ano"].dropna().unique()])
    print(f"  Anos cobertos:   {anos}")


if __name__ == "__main__":
    main()