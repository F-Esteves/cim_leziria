import pandas as pd

from etl.utils import (
    STAGING_DIR,
    row_base,
    normalizar_scores, enforce_schema,
)

_METRICAS_INVERTER = {
    "mob_registo_total_1000hab",
}

_METRICAS_SEM_NORMALIZACAO = {
    "mob_evolucao_veiculos_pp",
    "mob_crescimento_ve_pct",
}



# ── 3.1 Veículos ───────────────────────────────────────────────

def transform_veiculos() -> pd.DataFrame:
    print("  → Transformando veículos INE")
    df = pd.read_parquet(STAGING_DIR / "mob_veiculos.parquet")

    tipo_metrica = {
        "total":    "mob_registo_total_1000hab",
        "ligeiros": "mob_registo_ligeiros_1000hab",
        "pesados":  "mob_registo_pesados_1000hab",
        "tratores": "mob_registo_tratores_1000hab",
    }

    rows = []
    for _, row in df.iterrows():
        metrica = tipo_metrica.get(row["tipo"])
        if metrica and pd.notna(row["valor"]):
            rows.append(row_base(row["codigo_ine"], row["nome"],
                                 int(row["ano"]), metrica, round(float(row["valor"]), 4)))

    df_metr = pd.DataFrame(rows)

    # Evolução (p.p./ano): (2024 − 2021) / 3 para a métrica total
    pivot = df_metr[df_metr["metrica_codigo"] == "mob_registo_total_1000hab"] \
               .pivot_table(index="codigo_ine", columns="ano", values="valor")

    anos_disp = sorted(pivot.columns)
    if len(anos_disp) >= 2:
        ano_ini, ano_fim = anos_disp[0], anos_disp[-1]
        n = ano_fim - ano_ini
        for cod in pivot.index:
            v_ini = pivot.loc[cod].get(ano_ini)
            v_fim = pivot.loc[cod].get(ano_fim)
            if v_ini is not None and v_fim is not None and n > 0:
                evolucao = (v_fim - v_ini) / n
                nome = df_metr[df_metr["codigo_ine"] == cod]["nome"].iloc[0]
                rows.append(row_base(cod, nome, int(ano_fim),
                                     "mob_evolucao_veiculos_pp", round(evolucao, 4)))

    df_metr = pd.DataFrame(rows)
    print(f"     {len(df_metr)} registos · {df_metr['metrica_codigo'].nunique()} métricas")
    return df_metr


# ── 3.2 Pontos de Carregamento VE ─────────────────────────────

def transform_pontos_ve() -> pd.DataFrame:
    print("  → Transformando pontos de carregamento VE")
    df_ac = pd.read_parquet(STAGING_DIR / "mob_pontos_acesso.parquet")
    df_tp = pd.read_parquet(STAGING_DIR / "mob_pontos_tipo.parquet")

    soc_path = STAGING_DIR / "soc_censos_2021.parquet"
    if soc_path.exists():
        pop_total = pd.read_parquet(soc_path).set_index("codigo_ine")["valor"].to_dict()
    else:
        pop_total = {}
        print("     ⚠  soc_censos_2021.parquet não encontrado — mob_ve_por_1000hab não será calculado")

    rows = []
    ano  = 2024

    dez_ac = df_ac[df_ac["mes"] == "Dezembro"].copy()
    pivot_ac = dez_ac.pivot_table(
        index=["codigo_ine", "nome"],
        columns="tipo",
        values="n_pontos",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    for _, row in pivot_ac.iterrows():
        cod   = row["codigo_ine"]
        nome  = row["nome"]
        total = float(row.get("total", 0))
        pub   = float(row.get("publico", 0))
        priv  = float(row.get("privado", 0))

        rows.append(row_base(cod, nome, ano, "mob_ve_total", round(total, 0)))

        pop = pop_total.get(cod)
        if pop and pop > 0:
            rows.append(row_base(cod, nome, ano, "mob_ve_por_1000hab",
                                 round(total / pop * 1000, 4)))

        if total > 0:
            rows.append(row_base(cod, nome, ano, "mob_ve_publicos_pct",
                                 round(pub / total * 100, 2)))
            rows.append(row_base(cod, nome, ano, "mob_ve_privados_pct",
                                 round(priv / total * 100, 2)))

    dez_tp = df_tp[df_tp["mes"] == "Dezembro"].copy()
    pivot_tp = dez_tp.pivot_table(
        index=["codigo_ine", "nome"],
        columns="tipo",
        values="n_pontos",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    for _, row in pivot_tp.iterrows():
        cod   = row["codigo_ine"]
        nome  = row["nome"]
        total = float(row.get("total", 0))
        semi  = float(row.get("semirrapido", 0))
        rap   = float(row.get("rapido", 0))
        ultra = float(row.get("ultrarapido", 0))

        if total > 0:
            rows.append(row_base(cod, nome, ano, "mob_ve_rapidos_pct",
                                 round((rap + ultra) / total * 100, 2)))
            rows.append(row_base(cod, nome, ano, "mob_ve_semirrapidos_pct",
                                 round(semi / total * 100, 2)))

    # ── Crescimento anual ((Dez − Jan) / Jan × 100) ───────────
    jan_total = df_ac[(df_ac["mes"] == "Janeiro") & (df_ac["tipo"] == "total")]
    dez_total = df_ac[(df_ac["mes"] == "Dezembro") & (df_ac["tipo"] == "total")]

    for cod in jan_total["codigo_ine"].unique():
        v_jan = jan_total[jan_total["codigo_ine"] == cod]["n_pontos"].sum()
        v_dez = dez_total[dez_total["codigo_ine"] == cod]["n_pontos"].sum()
        if v_jan > 0:
            crescimento = (v_dez - v_jan) / v_jan * 100
            nome = df_ac[df_ac["codigo_ine"] == cod]["nome"].iloc[0]
            rows.append(row_base(cod, nome, ano, "mob_ve_crescimento_pct",
                                 round(crescimento, 2)))

    df_metr = pd.DataFrame(rows)
    print(f"     {len(df_metr)} registos · {df_metr['metrica_codigo'].nunique()} métricas")
    return df_metr


# ── Main ───────────────────────────────────────────────────────

def main():
    print("\n=== TRANSFORM · Cluster 3 — Mobilidade ===\n")

    print("[ 3.1 ] Veículos")
    df_veiculos = transform_veiculos()

    print("\n[ 3.2 ] Pontos de Carregamento VE")
    df_pontos = transform_pontos_ve()

    df_all = pd.concat([df_veiculos, df_pontos], ignore_index=True)
    df_all = normalizar_scores(
        df_all,
        metricas_inverter=_METRICAS_INVERTER,
        metricas_sem_normalizacao=_METRICAS_SEM_NORMALIZACAO,
    )
    df_all = enforce_schema(df_all)

    df_all["valor_texto"] = None
    df_all["categoria"]   = None

    df_all.to_parquet(STAGING_DIR / "mob_transformed.parquet", index=False)

    print(f"\n✓ Transform concluído")
    print(f"  Total registos:  {len(df_all)}")
    print(f"  Métricas únicas: {df_all['metrica_codigo'].nunique()}")
    print(f"  Municípios:      {df_all['codigo_ine'].nunique()}")
    anos = sorted([int(a) for a in df_all["ano"].dropna().unique()])
    print(f"  Anos cobertos:   {anos}")
    print(f"\n  Métricas calculadas:")
    for m in sorted(df_all["metrica_codigo"].unique()):
        n = df_all[df_all["metrica_codigo"] == m]["codigo_ine"].nunique()
        print(f"    {m:40s} ({n} municípios)")


if __name__ == "__main__": 
    main()
