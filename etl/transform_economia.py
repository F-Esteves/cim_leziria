import pandas as pd
import numpy as np
from pathlib import Path

STAGING_DIR = Path("data/staging")

MUNICIPIOS = {
    "1403": "Almeirim",   "1404": "Alpiarça",   "1103": "Azambuja",
    "1405": "Benavente",  "1406": "Cartaxo",     "1407": "Chamusca",
    "1409": "Coruche",    "1412": "Golegã",      "1414": "Rio Maior",
    "1415": "Salvaterra de Magos",               "1416": "Santarém",
}

CAE_SECTOR = {
    "A": "agricultura",
    "B": "industria", "C": "industria", "D": "industria",
    "E": "industria", "F": "industria",
    "G": "comercio", "I": "comercio",
    "H": "servicos", "J": "servicos", "K": "servicos",
    "L": "servicos", "M": "servicos", "N": "servicos",
}

def safe_num(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None

def normalizar(series: pd.Series, inverter: bool = False) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    norm = (series - mn) / (mx - mn)
    return 1 - norm if inverter else norm


# ── 5.1 Emprego e Estrutura ───────────────────────────────────────────────────

def calcular_metricas_emprego(df_co: pd.DataFrame, df_cens_bruto: pd.DataFrame,
                               pop_ativa: dict) -> pd.DataFrame:
    """
    Métricas:
      eco_taxa_emprego_pct        — Empregados_Censos / Pop_ativa × 100
      eco_estrutura_agricultura_pct — Pop sector primário / Total_empregados × 100
      eco_estrutura_industria_pct   — Pop sector secundário / Total_empregados × 100
      eco_estrutura_servicos_pct    — Pop sector terciário / Total_empregados × 100
      eco_taxa_conta_propria_pct  — (Empregadores + Trab. isolados) / Total × 100
      eco_taxa_grandes_empregadores_pct — Empregadores ≥10 / Total × 100
    """
    rows = []

    # Filtrar apenas sexo H e M (ambos) — somar para obter total
    SEXOS_AMBOS = {"H", "M"}
    df = df_cens_bruto.copy()
    df_hm = df[df["sexo"].isin(SEXOS_AMBOS)].copy()

    # Total geral (todos sectores, todas situações) = soma de todas as células H+M
    total_por_mun = df_hm.groupby("codigo_ine")["valor"].sum().to_dict()

    # Taxa de emprego = empregados / pop_ativa × 100
    for cod, emp in total_por_mun.items():
        pop = pop_ativa.get(cod)
        v = safe_num(emp)
        if v and pop and pop > 0:
            rows.append({
                "codigo_ine": cod, "nome": MUNICIPIOS.get(cod, ""),
                "ano": 2021, "metrica_codigo": "eco_taxa_emprego_pct",
                "valor": round(v / pop * 100, 2),
                "flag_estimado": False,
            })

    # Estrutura setorial — os sectores no ficheiro wide são:
    # 'primario', 'secundario', 'terciario_social', 'terciario_econ'
    # Mapeamento → macro-sector para os indicadores
    SETOR_MAP_WIDE = {
        "primario":          "agricultura",
        "secundario":        "industria",
        "terciario_social":  "servicos",
        "terciario_econ":    "servicos",
    }

    setor_sum = (df_hm
                 .assign(macro=df_hm["setor"].map(SETOR_MAP_WIDE))
                 .dropna(subset=["macro"])
                 .groupby(["codigo_ine", "macro"])["valor"]
                 .sum())

    for cod in total_por_mun:
        tot = safe_num(total_por_mun.get(cod))
        if not tot or tot == 0:
            continue
        for macro in ["agricultura", "industria", "servicos"]:
            v = safe_num(setor_sum.get((cod, macro)))
            if v is not None:
                rows.append({
                    "codigo_ine": cod, "nome": MUNICIPIOS.get(cod, ""),
                    "ano": 2021,
                    "metrica_codigo": f"eco_estrutura_{macro}_pct",
                    "valor": round(v / tot * 100, 2),
                    "flag_estimado": False,
                })

    # Conta própria: emp_lt10 + emp_ge10 + conta_propria / total
    # Situações no parser wide: 'emp_lt10', 'emp_ge10', 'conta_propria', 'conta_outrem', 'outra'
    CP_SITS = {"emp_lt10", "emp_ge10", "conta_propria"}
    df_cp = df_hm[df_hm["situacao"].isin(CP_SITS)]
    cp_sum = df_cp.groupby("codigo_ine")["valor"].sum()
    for cod, v_cp in cp_sum.items():
        tot = safe_num(total_por_mun.get(cod))
        v = safe_num(v_cp)
        if v and tot and tot > 0:
            rows.append({
                "codigo_ine": cod, "nome": MUNICIPIOS.get(cod, ""),
                "ano": 2021, "metrica_codigo": "eco_taxa_conta_propria_pct",
                "valor": round(v / tot * 100, 2),
                "flag_estimado": False,
            })

    # Grandes empregadores (≥10 trabalhadores): situacao = 'emp_ge10'
    df_ge = df_hm[df_hm["situacao"] == "emp_ge10"]
    ge_sum = df_ge.groupby("codigo_ine")["valor"].sum()
    for cod, v_ge in ge_sum.items():
        tot = safe_num(total_por_mun.get(cod))
        v = safe_num(v_ge)
        if v and tot and tot > 0:
            rows.append({
                "codigo_ine": cod, "nome": MUNICIPIOS.get(cod, ""),
                "ano": 2021, "metrica_codigo": "eco_taxa_grandes_empregadores_pct",
                "valor": round(v / tot * 100, 2),
                "flag_estimado": False,
            })

    df_out = pd.DataFrame(rows)
    print(f"   Emprego: {len(df_out)} registos · {df_out['metrica_codigo'].nunique()} métricas")
    return df_out


# ── 5.2 Rendimento e Capacidade Fiscal ────────────────────────────────────────

def calcular_metricas_rendimento(df_rb, df_irs, df_ipc, df_ppc,
                                  pop_total: dict) -> pd.DataFrame:
    """
    Métricas:
      eco_rendimento_bruto_per_capita_e — RB_k€ × 1000 / pop
      eco_irs_per_capita_e              — IRS_k€ × 1000 / pop
      eco_taxa_esforco_irs_pct          — IRS / RB × 100
      eco_ipc_base100                   — valor directo do dataset
      eco_tcma_rendimento_bruto_pct     — TCMA 2018-2023
      eco_proporcao_pc_pct              — valor directo do dataset
    """
    rows = []

    rb_idx  = df_rb.set_index(["codigo_ine","ano"])["valor"]
    irs_idx = df_irs.set_index(["codigo_ine","ano"])["valor"]

    # Rendimento bruto e IRS per capita
    for idx in rb_idx.index:
        cod, ano = idx
        v_rb = safe_num(rb_idx[idx])
        pop = pop_total.get(cod)
        if v_rb and pop and pop > 0:
            rows.append({
                "codigo_ine": cod, "nome": MUNICIPIOS.get(cod,""),
                "ano": int(ano), "metrica_codigo": "eco_rendimento_bruto_per_capita_e",
                "valor": round(v_rb * 1000 / pop, 2),
                "flag_estimado": False,
            })

    for idx in irs_idx.index:
        cod, ano = idx
        v_irs = safe_num(irs_idx[idx])
        pop = pop_total.get(cod)
        if v_irs and pop and pop > 0:
            rows.append({
                "codigo_ine": cod, "nome": MUNICIPIOS.get(cod,""),
                "ano": int(ano), "metrica_codigo": "eco_irs_per_capita_e",
                "valor": round(v_irs * 1000 / pop, 2),
                "flag_estimado": False,
            })

    # Taxa de esforço fiscal
    comuns = rb_idx.index.intersection(irs_idx.index)
    for idx in comuns:
        v_rb  = safe_num(rb_idx[idx])
        v_irs = safe_num(irs_idx[idx])
        if v_rb and v_irs and v_rb > 0:
            cod, ano = idx
            rows.append({
                "codigo_ine": cod, "nome": MUNICIPIOS.get(cod,""),
                "ano": int(ano), "metrica_codigo": "eco_taxa_esforco_irs_pct",
                "valor": round(v_irs / v_rb * 100, 4),
                "flag_estimado": False,
            })

    # IPC per capita (valor directo)
    for _, r in df_ipc.iterrows():
        v = safe_num(r["valor"])
        if v is not None:
            rows.append({
                "codigo_ine": r["codigo_ine"], "nome": r["nome"],
                "ano": int(r["ano"]), "metrica_codigo": "eco_ipc_base100",
                "valor": round(v, 2),
                "flag_estimado": False,
            })

    # Proporção PC (valor directo)
    for _, r in df_ppc.iterrows():
        v = safe_num(r["valor"])
        if v is not None:
            rows.append({
                "codigo_ine": r["codigo_ine"], "nome": r["nome"],
                "ano": int(r["ano"]), "metrica_codigo": "eco_proporcao_pc_pct",
                "valor": round(v, 4),
                "flag_estimado": False,
            })

    # TCMA rendimento bruto 2018-2023
    pivot = df_rb.pivot_table(index="codigo_ine", columns="ano", values="valor", aggfunc="first")
    anos_disp = sorted(pivot.columns)
    if len(anos_disp) >= 2:
        a_ini, a_fim = anos_disp[0], anos_disp[-1]
        n = a_fim - a_ini
        for cod in pivot.index:
            v_ini = safe_num(pivot.loc[cod].get(a_ini))
            v_fim = safe_num(pivot.loc[cod].get(a_fim))
            if v_ini and v_fim and v_ini > 0 and n > 0:
                tcma = ((v_fim / v_ini) ** (1/n) - 1) * 100
                rows.append({
                    "codigo_ine": cod, "nome": MUNICIPIOS.get(cod,""),
                    "ano": int(a_fim), "metrica_codigo": "eco_tcma_rendimento_bruto_pct",
                    "valor": round(tcma, 4),
                    "flag_estimado": False,
                })

    df_out = pd.DataFrame(rows)
    print(f"   Rendimento: {len(df_out)} registos · {df_out['metrica_codigo'].nunique()} métricas")
    return df_out


# ── 5.3 Empresarialidade ──────────────────────────────────────────────────────

def calcular_metricas_empresarialidade(df_nasc, df_mort, df_sobr,
                                        df_vn_tot, df_vn_sect,
                                        pop_total: dict) -> pd.DataFrame:
    """
    Métricas:
      eco_taxa_natalidade_emp_pct       — nascidas / stock_empresas × 100
      eco_taxa_mortalidade_emp_pct      — cessadas / stock_empresas × 100
      eco_taxa_sobrevivencia_1ano_pct   — sobrev_t / nascidas_(t-1) × 100
      eco_vn_per_capita_e               — VN_total / pop

    NOTA: o ficheiro volume-de-negocios.xls fornece apenas o Total CAE por
    município, sem desagregação sectorial ao nível municipal. As métricas
    eco_estrutura_vn_* não podem ser calculadas a partir desta fonte.
    """
    rows = []

    nasc_idx = df_nasc.set_index(["codigo_ine","ano"])["valor"]
    mort_idx = df_mort.set_index(["codigo_ine","ano"])["valor"]
    sobr_idx = df_sobr.set_index(["codigo_ine","ano"])["valor"]

    comuns = nasc_idx.index.intersection(mort_idx.index)
    for idx in comuns:
        n = safe_num(nasc_idx[idx])
        m = safe_num(mort_idx[idx])
        if n is None or m is None:
            continue
        cod, ano = idx
        s_prev = safe_num(sobr_idx.get((cod, ano - 1)))
        if s_prev is None:
            continue
        denominador = s_prev + n + m
        if denominador and denominador > 0:
            rows.append({"codigo_ine": cod, "nome": MUNICIPIOS.get(cod, ""),
                         "ano": int(ano), "metrica_codigo": "eco_taxa_natalidade_emp_pct",
                         "valor": round(n / denominador * 100, 2),
                         "flag_estimado": False})
            rows.append({"codigo_ine": cod, "nome": MUNICIPIOS.get(cod, ""),
                         "ano": int(ano), "metrica_codigo": "eco_taxa_mortalidade_emp_pct",
                         "valor": round(m / denominador * 100, 2),
                         "flag_estimado": False})

    # Taxa de sobrevivência a 1 ano
    for idx in sobr_idx.index:
        cod, ano = idx
        s = safe_num(sobr_idx[idx])
        n_prev = safe_num(nasc_idx.get((cod, ano - 1)))
        if s and n_prev and n_prev > 0:
            rows.append({"codigo_ine": cod, "nome": MUNICIPIOS.get(cod,""),
                         "ano": int(ano), "metrica_codigo": "eco_taxa_sobrevivencia_1ano_pct",
                         "valor": round(s / n_prev * 100, 2),
                         "flag_estimado": False})

    # VN per capita
    for _, r in df_vn_tot.iterrows():
        v = safe_num(r["valor"])
        pop = pop_total.get(r["codigo_ine"])
        if v and pop and pop > 0:
            rows.append({"codigo_ine": r["codigo_ine"], "nome": r["nome"],
                         "ano": int(r["ano"]), "metrica_codigo": "eco_vn_per_capita_e",
                         "valor": round(v / pop, 2),
                         "flag_estimado": False})

    df_out = pd.DataFrame(rows)
    print(f"   Empresarialidade: {len(df_out)} registos · {df_out['metrica_codigo'].nunique()} métricas")
    return df_out

# ── Normalização ──────────────────────────────────────────────────────────────

METRICAS_INVERTER = {
    "eco_taxa_mortalidade_emp_pct",
    "eco_taxa_esforco_irs_pct",
}
METRICAS_SEM_NORMALIZACAO = {
    "eco_proporcao_pc_pct",  
}

def normalizar_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["valor_normalizado"] = np.nan
    for (metrica, ano), grupo in df.groupby(["metrica_codigo","ano"]):
        if metrica in METRICAS_SEM_NORMALIZACAO:
            continue
        vals = grupo["valor"].dropna()
        if len(vals) < 2:
            df.loc[grupo.index, "valor_normalizado"] = 0.5
            continue
        inv = metrica in METRICAS_INVERTER
        df.loc[grupo.index, "valor_normalizado"] = normalizar(
            df.loc[grupo.index, "valor"], inverter=inv).values
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== TRANSFORM · Cluster 5 — Economia ===\n")

    soc_path = STAGING_DIR / "soc_censos_2021.parquet"
    if soc_path.exists():
        df_pop    = pd.read_parquet(soc_path)
        pop_total = df_pop.set_index("codigo_ine")["valor"].to_dict()
        pop_ativa = pop_total.copy()
        print(f"  Pop Censos 2021 carregada de soc_censos_2021.parquet — {len(pop_total)} municípios")
    else:
        print("  ⚠  soc_censos_2021.parquet não encontrado — métricas per capita serão NULL")
        print("     Corre extract_sociedade.py primeiro")
        pop_ativa = {}
        pop_total = {}

    print("[ 5.1 ] Emprego e Estrutura")
    df_co        = pd.read_parquet(STAGING_DIR / "eco_emprego_conta_outrem.parquet")
    df_cens_bruto = pd.read_parquet(STAGING_DIR / "eco_emprego_censos_bruto.parquet")
    metr_emp     = calcular_metricas_emprego(df_co, df_cens_bruto, pop_ativa)

    print("\n[ 5.2 ] Rendimento e Capacidade Fiscal")
    df_rb  = pd.read_parquet(STAGING_DIR / "eco_rendimento_bruto.parquet")
    df_irs = pd.read_parquet(STAGING_DIR / "eco_irs_liquidado.parquet")
    df_ipc = pd.read_parquet(STAGING_DIR / "eco_poder_compra_per_capita.parquet")
    df_ppc = pd.read_parquet(STAGING_DIR / "eco_proporcao_poder_compra.parquet")
    metr_rend = calcular_metricas_rendimento(df_rb, df_irs, df_ipc, df_ppc, pop_total)

    print("\n[ 5.3 ] Empresarialidade")
    df_nasc   = pd.read_parquet(STAGING_DIR / "eco_empresas_nascidas.parquet")
    df_mort   = pd.read_parquet(STAGING_DIR / "eco_empresas_mortas.parquet")
    df_sobr   = pd.read_parquet(STAGING_DIR / "eco_empresas_sobreviventes.parquet")
    df_vn_tot = pd.read_parquet(STAGING_DIR / "eco_volume_negocios_total.parquet")
    df_vn_sec = pd.read_parquet(STAGING_DIR / "eco_volume_negocios_sectores.parquet")
    metr_emp2 = calcular_metricas_empresarialidade(
        df_nasc, df_mort, df_sobr, df_vn_tot, df_vn_sec, pop_total)

    df_all = pd.concat([metr_emp, metr_rend, metr_emp2], ignore_index=True)
    df_all = normalizar_scores(df_all)

    if "flag_estimado" in df_all.columns:
        df_all[df_all["flag_estimado"] == True][  # noqa: E712
            ["codigo_ine", "nome", "ano", "metrica_codigo", "valor"]
        ].to_parquet(STAGING_DIR / "eco_estimados.parquet", index=False)
        df_all = df_all.drop(columns=["flag_estimado"])

    df_all["valor_texto"] = None
    df_all["categoria"]   = None

    df_all.to_parquet(STAGING_DIR / "eco_transformed.parquet", index=False)

    print(f"\n✓ Transform concluído")
    print(f"   Total registos:  {len(df_all)}")
    print(f"   Métricas únicas: {df_all['metrica_codigo'].nunique()}")
    print(f"   Municípios:      {df_all['codigo_ine'].nunique()}")
    anos = sorted([int(a) for a in df_all["ano"].dropna().unique()])
    print(f"   Anos cobertos:   {anos}")


if __name__ == "__main__":
    main()