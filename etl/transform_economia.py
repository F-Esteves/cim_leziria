import pandas as pd

from etl.utils import (
    STAGING_DIR, MUNICIPIOS,
    safe_float as safe_num,
    normalizar_scores, enforce_schema,
    carregar_populacao_referencia, carregar_populacao_serie,
)

CAE_SECTOR = {
    "A": "agricultura",
    "B": "industria", "C": "industria", "D": "industria",
    "E": "industria", "F": "industria",
    "G": "comercio", "I": "comercio",
    "H": "servicos", "J": "servicos", "K": "servicos",
    "L": "servicos", "M": "servicos", "N": "servicos",
}


# ── 5.1 Emprego e Estrutura ───────────────────────────────────────────────────

def calcular_metricas_emprego(df_cens_bruto: pd.DataFrame,
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
                                        pop_serie: dict) -> pd.DataFrame:
    """
    Métricas:
      eco_empresas_nascidas_n            — nascidas (valor absoluto)
      eco_empresas_mortas_n              — mortas (valor absoluto)
      eco_taxa_sobrevivencia_1ano_pct    — sobrev_t / nascidas_(t-1) × 100
      eco_vn_per_capita_e                — VN_total / pop (mesmo ano; inclui Portugal)
      eco_estrutura_vn_agricultura_pct   — VN Agricultura / VN Total × 100
      eco_estrutura_vn_industria_pct     — VN Indústria / VN Total × 100
      eco_estrutura_vn_servicos_pct      — VN Serviços / VN Total × 100

    A partir da nova fonte (INE, CAE Rev.3 detalhado, com Portugal), as
    métricas de estrutura sectorial passam a ser calculáveis — a fonte
    anterior só tinha o Total por município, sem desagregação sectorial.
    """
    rows = []

    nasc_idx = df_nasc.set_index(["codigo_ine","ano"])["valor"]
    mort_idx = df_mort.set_index(["codigo_ine","ano"])["valor"]
    sobr_idx = df_sobr.set_index(["codigo_ine","ano"])["valor"]

    # Anos com quebra de série — excluídos do cálculo
    ANOS_EXCLUIR = {2013}

    # Valores absolutos: nascidas e mortas por município e ano
    # As taxas percentuais foram removidas — o stock total de empresas activas
    # não está disponível nesta fonte (PORDATA), tornando o denominador incorrecto.
    # Usam-se valores absolutos directamente comparáveis entre municípios.
    comuns = nasc_idx.index.intersection(mort_idx.index)
    for idx in comuns:
        cod, ano = idx
        if ano in ANOS_EXCLUIR:
            continue
        n = safe_num(nasc_idx[idx])
        m = safe_num(mort_idx[idx])
        if n is None or m is None:
            continue
        rows.append({"codigo_ine": cod, "nome": MUNICIPIOS.get(cod, ""),
                     "ano": int(ano), "metrica_codigo": "eco_empresas_nascidas_n",
                     "valor": round(n, 0), "flag_estimado": False})
        rows.append({"codigo_ine": cod, "nome": MUNICIPIOS.get(cod, ""),
                     "ano": int(ano), "metrica_codigo": "eco_empresas_mortas_n",
                     "valor": round(m, 0), "flag_estimado": False})
        rows.append({"codigo_ine": cod, "nome": MUNICIPIOS.get(cod, ""),
                     "ano": int(ano), "metrica_codigo": "eco_saldo_empresarial_n",
                     "valor": round(n - m, 0), "flag_estimado": False})

    # VN per capita — população do MESMO ano (a série de VN é 2022-2024;
    # pop_serie inclui Portugal, para o benchmark nacional).
    for _, r in df_vn_tot.iterrows():
        v   = safe_num(r["valor"])
        pop = pop_serie.get((r["codigo_ine"], int(r["ano"])))
        if v and pop and pop > 0:
            rows.append({"codigo_ine": r["codigo_ine"], "nome": r["nome"],
                         "ano": int(r["ano"]), "metrica_codigo": "eco_vn_per_capita_e",
                         "valor": round(v / pop, 2),
                         "flag_estimado": False})

    # Estrutura sectorial (Agricultura / Indústria / Serviços) — % do Total
    if df_vn_sect is not None and not df_vn_sect.empty:
        tot_idx = df_vn_tot.set_index(["codigo_ine", "ano"])["valor"]
        metrica_por_cae = {
            "Agricultura": "eco_estrutura_vn_agricultura_pct",
            "Industria":   "eco_estrutura_vn_industria_pct",
            "Servicos":    "eco_estrutura_vn_servicos_pct",
        }
        for _, r in df_vn_sect.iterrows():
            cod, ano, cae, valor = r["codigo_ine"], int(r["ano"]), r["cae"], safe_num(r["valor"])
            total = tot_idx.get((cod, ano))
            metrica = metrica_por_cae.get(cae)
            if metrica and valor is not None and total and total > 0:
                rows.append({"codigo_ine": cod, "nome": r["nome"], "ano": ano,
                             "metrica_codigo": metrica,
                             "valor": round(valor / total * 100, 2),
                             "flag_estimado": False})

    df_out = pd.DataFrame(rows)
    print(f"   Empresarialidade: {len(df_out)} registos · {df_out['metrica_codigo'].nunique()} métricas")
    return df_out

# ── Normalização ──────────────────────────────────────────────────────────────

_METRICAS_INVERTER = {
    "eco_empresas_mortas_n",  # mais mortas = pior dinamismo empresarial
    "eco_taxa_esforco_irs_pct",
}
_METRICAS_SEM_NORMALIZACAO = {
    "eco_proporcao_pc_pct",
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== TRANSFORM · Cluster 5 — Economia ===\n")

    pop_total = carregar_populacao_referencia()
    # A taxa de emprego usa dados do Censos 2021 (numerador fixo — o INE só
    # atualiza o Censos de 10 em 10 anos, próximo em 2031). Antes dividia-se
    # por pop_total do ANO_REFERENCIA_POPULACAO (2025) — desfasamento de
    # anos com o numerador. Agora usa-se a população residente do MESMO
    # ano (2021), disponível na série anual do INE, para consistência
    # interna. Continua sem ser "população ativa" a sério (população em
    # idade de trabalhar / força de trabalho) — é população residente
    # total, mas ao menos o ano bate certo com o numerador.
    pop_ativa = carregar_populacao_referencia(ano=2021)
    if pop_total:
        print(f"  Pop. residente (INE, ano de referência) — {len(pop_total)} municípios")
    else:
        print("  ⚠  soc_censos_2021.parquet não encontrado — métricas per capita serão NULL")
        print("     Corre extract_sociedade.py primeiro")

    print("[ 5.1 ] Emprego e Estrutura")
    # NOTA: df_co (emprego por conta de outrem) é lido mas NÃO é usado por
    # calcular_metricas_emprego() — a função só trabalha com df_cens_bruto.
    # Ficava a mais na chamada (bug pré-existente: função só aceita 2
    # argumentos). Mantido aqui caso seja para vir a alimentar uma métrica
    # futura; por agora só serve de leitura sem efeito.
    df_co        = pd.read_parquet(STAGING_DIR / "eco_emprego_conta_outrem.parquet")
    df_cens_bruto = pd.read_parquet(STAGING_DIR / "eco_emprego_censos_bruto.parquet")
    metr_emp     = calcular_metricas_emprego(df_cens_bruto, pop_ativa)

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
        df_nasc, df_mort, df_sobr, df_vn_tot, df_vn_sec,
        carregar_populacao_serie(incluir_pt=True))

    df_all = pd.concat([metr_emp, metr_rend, metr_emp2], ignore_index=True)
    df_all = normalizar_scores(
        df_all,
        metricas_inverter=_METRICAS_INVERTER,
        metricas_sem_normalizacao=_METRICAS_SEM_NORMALIZACAO,
    )
    df_all = enforce_schema(df_all)

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