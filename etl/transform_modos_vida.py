import pandas as pd

from etl.utils import (
    STAGING_DIR,
    row_base,
    normalizar_scores, enforce_schema,
    carregar_populacao_referencia,
)

# Métricas onde menor = melhor
_METRICAS_INVERTER = {
    "mdv_hab_medico",
    "mdv_hab_farmaceutico",
    "mdv_acidentes_vitimas_1000hab",
    "mdv_criminalidade_total",
    "mdv_criminalidade_patrimonio",
    "mdv_criminalidade_integridade_fisica",
    "mdv_sem_escolaridade_pct",
    "mdv_mortos_acidentes",
    "mdv_feridos_acidentes",
    "mdv_alojamentos_vagos_pct",
}

# Métricas sem normalização (valores CIM-level, não por município)
_METRICAS_SEM_NORMALIZACAO = {
    "mdv_utentes_csp",
    "mdv_consultas_presenciais",
    "mdv_consultas_total",
    "mdv_alojamentos_total",
    "mdv_alojamentos_familares",
    "mdv_alojamentos_uso_sazonal",
    "mdv_alojamentos_vagos",
    "mdv_ensino_superior_n",
}



# ── 4.1 Saúde ─────────────────────────────────────────────────

def transform_saude() -> pd.DataFrame:
    rows = []

    # Hab/médico e hab/farmacêutico — passagem directa
    df = pd.read_parquet(STAGING_DIR / "mdv_hab_medico.parquet")
    for _, r in df.iterrows():
        if pd.notna(r["valor"]):
            rows.append(row_base(r["codigo_ine"], r["nome"], int(r["ano"]),
                                 f"mdv_{r['metrica']}", round(float(r["valor"]), 2)))

    # Profissionais de saúde — passagem directa
    df = pd.read_parquet(STAGING_DIR / "mdv_profissionais.parquet")
    for _, r in df.iterrows():
        if pd.notna(r["valor"]):
            rows.append(row_base(r["codigo_ine"], r["nome"], int(r["ano"]),
                                 f"mdv_{r['metrica']}", round(float(r["valor"]), 0)))

    # Utentes CSP e % com MdF — nível ULS Lezíria
    df = pd.read_parquet(STAGING_DIR / "mdv_utentes_csp.parquet")
    pop_total = carregar_populacao_referencia()

    for _, r in df.iterrows():
        if pd.notna(r["valor"]):
            cod  = r["codigo_ine"]
            nome = r["nome"]
            ano  = int(r["ano"])
            met  = r["metrica"]
            rows.append(row_base(cod, nome, ano,
                                 f"mdv_{met}", round(float(r["valor"]), 2)))
        

    # Consultas CSP — nível ULS Lezíria
    df = pd.read_parquet(STAGING_DIR / "mdv_consultas_csp.parquet")
    for _, r in df.iterrows():
        if pd.notna(r["valor"]):
            cod  = r["codigo_ine"]
            nome = r["nome"]
            ano  = int(r["ano"])
            met  = r["metrica"]
            rows.append(row_base(cod, nome, ano,
                                 f"mdv_{met}", round(float(r["valor"]), 0)))
            # Consultas por habitante — âmbito ULS (não municipal), interpretação limitada
            # Mantida como indicador de tendência, não de comparação inter-municipal
            if met == "consultas_total":
                pop = pop_total.get(cod)
                if pop and pop > 0:
                    rows.append(row_base(cod, nome, ano, "mdv_consultas_por_hab",
                                         round(float(r["valor"]) / pop, 4)))

    df_metr = pd.DataFrame(rows)
    print(f"     Saúde: {len(df_metr)} registos · {df_metr['metrica_codigo'].nunique()} métricas")
    return df_metr


# ── 4.2 Segurança ─────────────────────────────────────────────

def transform_seguranca() -> pd.DataFrame:
    rows = []

    # Acidentes com vítimas /1000hab
    df = pd.read_parquet(STAGING_DIR / "mdv_acidentes_vitimas.parquet")
    for _, r in df.iterrows():
        if pd.notna(r["valor"]):
            rows.append(row_base(r["codigo_ine"], r["nome"], int(r["ano"]),
                                 "mdv_acidentes_vitimas_1000hab", round(float(r["valor"]), 4)))

    # Feridos e mortos em acidentes
    df = pd.read_parquet(STAGING_DIR / "mdv_feridos_mortos.parquet")
    for _, r in df.iterrows():
        if pd.notna(r["valor"]):
            rows.append(row_base(r["codigo_ine"], r["nome"], int(r["ano"]),
                                 f"mdv_{r['metrica']}", round(float(r["valor"]), 0)))

    # Criminalidade — total + categorias relevantes
    df = pd.read_parquet(STAGING_DIR / "mdv_criminalidade.parquet")
    for _, r in df.iterrows():
        if pd.notna(r["valor"]):
            rows.append(row_base(r["codigo_ine"], r["nome"], int(r["ano"]),
                                 f"mdv_{r['metrica']}", round(float(r["valor"]), 2)))

    df_metr = pd.DataFrame(rows)

    # Evolução taxa de criminalidade total (p.p./ano)
    crim_tot = df_metr[df_metr["metrica_codigo"] == "mdv_criminalidade_total"].copy()
    if not crim_tot.empty:
        pivot = crim_tot.pivot_table(index="codigo_ine", columns="ano", values="valor")
        anos  = sorted(pivot.columns)
        if len(anos) >= 2:
            ano_ini, ano_fim = anos[0], anos[-1]
            n = ano_fim - ano_ini
            for cod in pivot.index:
                v_ini = pivot.loc[cod].get(ano_ini)
                v_fim = pivot.loc[cod].get(ano_fim)
                if v_ini is not None and v_fim is not None and n > 0:
                    nome = df_metr[df_metr["codigo_ine"] == cod]["nome"].iloc[0]
                    rows.append(row_base(cod, nome, int(ano_fim),
                                         "mdv_evolucao_criminalidade_pp",
                                         round((v_fim - v_ini) / n, 4)))

    df_metr = pd.DataFrame(rows)
    print(f"     Segurança: {len(df_metr)} registos · {df_metr['metrica_codigo'].nunique()} métricas")
    return df_metr


# ── 4.3 Educação ──────────────────────────────────────────────

def transform_educacao() -> pd.DataFrame:
    rows = []

    df_sem = pd.read_parquet(STAGING_DIR / "mdv_sem_escolaridade.parquet")
    for _, r in df_sem.iterrows():
        if pd.notna(r["valor"]):
            rows.append(row_base(r["codigo_ine"], r["nome"], int(r["ano"]),
                                 "mdv_sem_escolaridade_pct", round(float(r["valor"]), 2)))

    df_sup = pd.read_parquet(STAGING_DIR / "mdv_ensino_superior.parquet")
    pop_total = carregar_populacao_referencia()

    for _, r in df_sup.iterrows():
        if pd.notna(r["valor"]):
            cod  = r["codigo_ine"]
            nome = r["nome"]
            ano  = int(r["ano"])
            n    = float(r["valor"])
            rows.append(row_base(cod, nome, ano, "mdv_ensino_superior_n", round(n, 0)))
            pop = pop_total.get(cod)
            if pop and pop > 0:
                rows.append(row_base(cod, nome, ano, "mdv_ensino_superior_pct",
                                     round(n / pop * 100, 2)))

    df_metr = pd.DataFrame(rows)

    # Novas fontes — pass-through directo (já vêm prontas da extração,
    # incluindo Portugal). Cada parquet já tem uma coluna "metrica" com o
    # nome final (sem prefixo "mdv_" — adicionado aqui, por convenção).
    novos_parquets = [
        "mdv_ensino_superior_inscritos",
        "mdv_ensino_nao_superior",
        "mdv_ensino_secundario_orientado",
        "mdv_tx_retencao_desistencia",
        "mdv_tx_transicao_conclusao",
    ]
    extra_rows = []
    for nome_parquet in novos_parquets:
        df_novo = pd.read_parquet(STAGING_DIR / f"{nome_parquet}.parquet")
        for _, r in df_novo.iterrows():
            if pd.notna(r["valor"]):
                extra_rows.append(row_base(
                    r["codigo_ine"], r["nome"], int(r["ano"]),
                    f"mdv_{r['metrica']}", round(float(r["valor"]), 2),
                ))

    df_metr = pd.concat([df_metr, pd.DataFrame(extra_rows)], ignore_index=True)
    print(f"     Educação: {len(df_metr)} registos · {df_metr['metrica_codigo'].nunique()} métricas")
    return df_metr


# ── 4.4 Turismo ───────────────────────────────────────────────

def transform_turismo() -> pd.DataFrame:
    rows = []

    df = pd.read_parquet(STAGING_DIR / "mdv_dormidas.parquet")
    for _, r in df.iterrows():
        if pd.notna(r["valor"]):
            rows.append(row_base(r["codigo_ine"], r["nome"], int(r["ano"]),
                                 "mdv_dormidas_100hab", round(float(r["valor"]), 2)))

    df_metr = pd.DataFrame(rows)

    # Evolução dormidas (p.p./ano)
    pivot = df_metr.pivot_table(index="codigo_ine", columns="ano", values="valor")
    anos  = sorted(pivot.columns)
    if len(anos) >= 2:
        ano_ini, ano_fim = anos[0], anos[-1]
        n = ano_fim - ano_ini
        for cod in pivot.index:
            v_ini = pivot.loc[cod].get(ano_ini)
            v_fim = pivot.loc[cod].get(ano_fim)
            if v_ini is not None and v_fim is not None and n > 0:
                nome = df_metr[df_metr["codigo_ine"] == cod]["nome"].iloc[0]
                rows.append(row_base(cod, nome, int(ano_fim),
                                     "mdv_evolucao_dormidas_pp",
                                     round((v_fim - v_ini) / n, 4)))

    df_aloj_tipo = pd.read_parquet(STAGING_DIR / "mdv_alojamentos_tipo.parquet")
    for _, r in df_aloj_tipo.iterrows():
        if pd.notna(r["valor"]):
            rows.append(row_base(r["codigo_ine"], r["nome"], int(r["ano"]),
                                 f"mdv_{r['metrica']}", round(float(r["valor"]), 0)))

    df_metr = pd.DataFrame(rows)
    print(f"     Turismo: {len(df_metr)} registos · {df_metr['metrica_codigo'].nunique()} métricas")
    return df_metr


# ── 4.5 Habitação ─────────────────────────────────────────────

def transform_habitacao() -> pd.DataFrame:
    rows = []

    df = pd.read_parquet(STAGING_DIR / "mdv_alojamentos.parquet")
    for _, r in df.iterrows():
        if pd.notna(r["valor"]):
            rows.append(row_base(r["codigo_ine"], r["nome"], int(r["ano"]),
                                 f"mdv_{r['metrica']}", round(float(r["valor"]), 0)))

    df_metr = pd.DataFrame(rows)

    # Taxa de alojamentos vagos (%)
    pivot_total = df_metr[df_metr["metrica_codigo"] == "mdv_alojamentos_total"] \
                      .set_index(["codigo_ine", "ano"])["valor"]
    pivot_vagos = df_metr[df_metr["metrica_codigo"] == "mdv_alojamentos_vagos"] \
                      .set_index(["codigo_ine", "ano"])["valor"]

    for (cod, ano) in pivot_total.index:
        total = pivot_total.get((cod, ano))
        vagos = pivot_vagos.get((cod, ano))
        if total and vagos and total > 0:
            nome = df_metr[df_metr["codigo_ine"] == cod]["nome"].iloc[0]
            rows.append(row_base(cod, nome, int(ano),
                                 "mdv_alojamentos_vagos_pct",
                                 round(vagos / total * 100, 2)))

    # Taxa de uso sazonal (%)
    pivot_saz = df_metr[df_metr["metrica_codigo"] == "mdv_alojamentos_uso_sazonal"] \
                    .set_index(["codigo_ine", "ano"])["valor"]
    for (cod, ano) in pivot_total.index:
        total = pivot_total.get((cod, ano))
        saz   = pivot_saz.get((cod, ano))
        if total and saz and total > 0:
            nome = df_metr[df_metr["codigo_ine"] == cod]["nome"].iloc[0]
            rows.append(row_base(cod, nome, int(ano),
                                 "mdv_alojamentos_sazonal_pct",
                                 round(saz / total * 100, 2)))

    df_metr = pd.DataFrame(rows)
    print(f"     Habitação: {len(df_metr)} registos · {df_metr['metrica_codigo'].nunique()} métricas")
    return df_metr


# ── Main ───────────────────────────────────────────────────────

def main():
    print("\n=== TRANSFORM · Cluster 4 — Modos de Vida ===\n")

    print("[ 4.1 ] Saúde")
    df_saude = transform_saude()

    print("[ 4.2 ] Segurança")
    df_seg = transform_seguranca()

    print("[ 4.3 ] Educação")
    df_educ = transform_educacao()

    print("[ 4.4 ] Turismo")
    df_tur = transform_turismo()

    print("[ 4.5 ] Habitação")
    df_hab = transform_habitacao()

    df_all = pd.concat([df_saude, df_seg, df_educ, df_tur, df_hab], ignore_index=True)
    df_all = normalizar_scores(
        df_all,
        metricas_inverter=_METRICAS_INVERTER,
        metricas_sem_normalizacao=_METRICAS_SEM_NORMALIZACAO,
    )
    df_all = enforce_schema(df_all)
    df_all.to_parquet(STAGING_DIR / "mdv_transformed.parquet", index=False)

    print(f"\n✓ Transform concluído")
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
