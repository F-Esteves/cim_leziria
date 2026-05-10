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

# Métricas onde menor = melhor (para normalização invertida)
METRICAS_INVERTER = {
    "amb_taxa_aterro_pct",
    "amb_var_consumo_anual_pct",
}

ANOS_INCOMPLETOS = {2020, 2025}


def normalizar(series: pd.Series, inverter: bool = False) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    norm = (series - mn) / (mx - mn)
    return 1 - norm if inverter else norm


def normalizar_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["valor_normalizado"] = np.nan
    for (metrica, ano), grupo in df.groupby(["metrica_codigo", "ano"]):
        vals = grupo["valor"].dropna()
        if len(vals) < 2:
            df.loc[grupo.index, "valor_normalizado"] = 0.5
            continue
        inv = metrica in METRICAS_INVERTER
        df.loc[grupo.index, "valor_normalizado"] = normalizar(
            df.loc[grupo.index, "valor"], inverter=inv
        ).values
    return df


def row_base(codigo_ine, nome, ano, metrica, valor):
    return {
        "codigo_ine":     codigo_ine,
        "nome":           nome,
        "ano":            ano,
        "metrica_codigo": metrica,
        "valor":          valor,
    }


# ── 2.1 Energia — Consumos ─────────────────────────────────────

def transform_consumos() -> pd.DataFrame:
    print("  → Transformando consumos EREDES")
    df = pd.read_parquet(STAGING_DIR / "amb_consumos.parquet")

    TENSAO_BT = "Baixa Tensão"
    TENSAO_AT = "Muito Alta, Alta e Média Tensões"

    rows = []

    # Agregar por município × ano (soma de todos os meses)
    anual = df.groupby(["codigo_ine", "nome", "ano", "tensao"])["energia_kwh"].sum().reset_index()

    pivot = anual.pivot_table(
        index=["codigo_ine", "nome", "ano"],
        columns="tensao",
        values="energia_kwh",
        aggfunc="sum",
    ).reset_index()

    bt_col = TENSAO_BT if TENSAO_BT in pivot.columns else None
    at_col = TENSAO_AT if TENSAO_AT in pivot.columns else None

    for _, row in pivot.iterrows():
        cod  = row["codigo_ine"]
        nome = row["nome"]
        ano  = int(row["ano"])
        if ano in ANOS_INCOMPLETOS:
            continue

        bt    = float(row[bt_col]) if bt_col and pd.notna(row[bt_col]) else 0.0
        at    = float(row[at_col]) if at_col and pd.notna(row[at_col]) else 0.0
        total = bt + at

        if total > 0:
            rows.append(row_base(cod, nome, ano, "amb_consumo_total_kwh", round(total, 0)))
            rows.append(row_base(cod, nome, ano, "amb_consumo_bt_kwh",    round(bt, 0)))
            rows.append(row_base(cod, nome, ano, "amb_consumo_at_kwh",    round(at, 0)))
            rows.append(row_base(cod, nome, ano, "amb_pct_consumo_bt",    round(bt / total * 100, 2)))

    df_metr = pd.DataFrame(rows)
    totais_ano = df_metr[df_metr["metrica_codigo"] == "amb_consumo_total_kwh"].copy()
    pivot_t = totais_ano.pivot_table(index="codigo_ine", columns="ano", values="valor")

    anos_completos = sorted([a for a in pivot_t.columns if a not in ANOS_INCOMPLETOS])
    if len(anos_completos) >= 2:
        ano_ini_tcma = anos_completos[0]
        ano_fim_tcma = anos_completos[-1]
        n_anos_tcma  = ano_fim_tcma - ano_ini_tcma

        for cod in pivot_t.index:
            c_ini = pivot_t.loc[cod].get(ano_ini_tcma)
            c_fim = pivot_t.loc[cod].get(ano_fim_tcma)
            if c_ini and c_fim and c_ini > 0 and n_anos_tcma > 0:
                tcma = ((c_fim / c_ini) ** (1 / n_anos_tcma) - 1) * 100
                nome = df_metr[df_metr["codigo_ine"] == cod]["nome"].iloc[0]
                df_metr = pd.concat([df_metr, pd.DataFrame([
                    row_base(cod, nome, ano_fim_tcma, "amb_tcma_consumo_pct", round(tcma, 4))
                ])], ignore_index=True)

    anos_sorted = sorted(totais_ano["ano"].unique())
    for i in range(1, len(anos_sorted)):
        ano_ant = anos_sorted[i - 1]
        ano_cur = anos_sorted[i]
        if ano_ant in ANOS_INCOMPLETOS or ano_cur in ANOS_INCOMPLETOS:
            continue
        for cod in pivot_t.index:
            c_ant = pivot_t.loc[cod].get(ano_ant)
            c_cur = pivot_t.loc[cod].get(ano_cur)
            if c_ant and c_cur and c_ant > 0:
                var = (c_cur - c_ant) / c_ant * 100
                nome = df_metr[df_metr["codigo_ine"] == cod]["nome"].iloc[0]
                df_metr = pd.concat([df_metr, pd.DataFrame([
                    row_base(cod, nome, int(ano_cur), "amb_var_consumo_anual_pct", round(var, 4))
                ])], ignore_index=True)

    print(f"     {len(df_metr)} registos · {df_metr['metrica_codigo'].nunique()} métricas")
    return df_metr


# ── 2.1 Energia — Contadores ───────────────────────────────────

def transform_contadores() -> pd.DataFrame:
    print("  → Transformando contadores EREDES")
    df = pd.read_parquet(STAGING_DIR / "amb_contadores.parquet")

    rows = []

    # Último mês de cada ano como referência (snapshot de fim de ano)
    ultimo_mes = df.groupby(["codigo_ine", "nome", "ano"])["mes"].max().reset_index()
    df_ultimo  = df.merge(ultimo_mes, on=["codigo_ine", "nome", "ano", "mes"])

    anual = df_ultimo.groupby(["codigo_ine", "nome", "ano", "smart"])["n_cpes"].sum().reset_index()
    pivot = anual.pivot_table(
        index=["codigo_ine", "nome", "ano"],
        columns="smart",
        values="n_cpes",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    sim_col = "Sim" if "Sim" in pivot.columns else None
    nao_col = "Não" if "Não" in pivot.columns else None

    for _, row in pivot.iterrows():
        cod   = row["codigo_ine"]
        nome  = row["nome"]
        ano   = int(row["ano"])
        sim   = float(row[sim_col]) if sim_col else 0.0
        nao   = float(row[nao_col]) if nao_col else 0.0
        total = sim + nao

        if total > 0:
            rows.append(row_base(cod, nome, ano, "amb_n_cpes_total",         round(total, 0)))
            rows.append(row_base(cod, nome, ano, "amb_pct_contadores_smart", round(sim / total * 100, 2)))

    df_metr = pd.DataFrame(rows)
    print(f"     {len(df_metr)} registos · {df_metr['metrica_codigo'].nunique()} métricas")
    return df_metr


# ── 2.1 Energia — Comunidades ──────────────────────────────────

def transform_comunidades() -> pd.DataFrame:
    print("  → Transformando comunidades de energia")
    df = pd.read_parquet(STAGING_DIR / "amb_comunidades.parquet")

    rows = []

    # Último período disponível por município
    ultimo = df.groupby(["codigo_ine", "nome"])[["ano", "mes"]].max().reset_index()
    df_ult = df.merge(ultimo, on=["codigo_ine", "nome", "ano", "mes"])

    agg_acc = df_ult.groupby(["codigo_ine", "nome", "ano"]).agg(
        n_comunidades=("contagem", "count"),
        n_membros=("contagem", "sum"),
    ).reset_index()

    for _, row in agg_acc.iterrows():
        cod  = row["codigo_ine"]
        nome = row["nome"]
        ano  = int(row["ano"])
        rows.append(row_base(cod, nome, ano, "amb_n_acc",       int(row["n_comunidades"])))
        rows.append(row_base(cod, nome, ano, "amb_membros_acc", int(row["n_membros"])))

    df_metr = pd.DataFrame(rows)
    print(f"     {len(df_metr)} registos · {df_metr['metrica_codigo'].nunique()} métricas")
    return df_metr


# ── 2.2 Resíduos ───────────────────────────────────────────────

def transform_residuos() -> pd.DataFrame:
    print("  → Transformando resíduos APA")
    df = pd.read_parquet(STAGING_DIR / "amb_residuos.parquet")

    # Pop Censos 2021 — para cálculo per capita (Cluster 6 — Sociedade)
    soc_path = STAGING_DIR / "soc_censos_2021.parquet"
    if soc_path.exists():
        pop_total = pd.read_parquet(soc_path).set_index("codigo_ine")["valor"].to_dict()
    else:
        pop_total = {}
        print("     ⚠  soc_censos_2021.parquet não encontrado — amb_residuos_per_capita não será calculado")

    rows = []

    for _, row in df.iterrows():
        cod  = row["codigo_ine"]
        nome = row["nome"]
        ano  = int(row["ano"]) if pd.notna(row["ano"]) else None
        if not ano:
            continue

        total   = row.get("total_ton")
        aterro  = row.get("aterro_ton")
        val_en  = row.get("val_energetica_ton")  or 0
        val_org = row.get("val_organica_ton")    or 0
        val_mul = row.get("val_multimaterial_ton") or 0

        if total and total > 0:
            rows.append(row_base(cod, nome, ano, "amb_residuos_total_ton", round(total, 1)))

            # Per capita: toneladas × 1000 / pop = kg/hab
            pop = pop_total.get(cod)
            if pop and pop > 0:
                rows.append(row_base(cod, nome, ano, "amb_residuos_per_capita_kg_hab",
                                     round(total * 1000 / pop, 2)))

            if aterro is not None:
                rows.append(row_base(cod, nome, ano, "amb_taxa_aterro_pct",
                                     round(aterro / total * 100, 2)))

            val_total = val_en + val_org + val_mul
            if val_total > 0:
                rows.append(row_base(cod, nome, ano, "amb_taxa_valorizacao_pct",
                                     round(val_total / total * 100, 2)))

            if val_mul > 0:
                rows.append(row_base(cod, nome, ano, "amb_taxa_reciclagem_pct",
                                     round(val_mul / total * 100, 2)))

    df_metr = pd.DataFrame(rows)

    # Evolução taxa valorização (p.p./ano) entre primeiro e último ano disponíveis
    val = df_metr[df_metr["metrica_codigo"] == "amb_taxa_valorizacao_pct"]
    if not val.empty:
        pivot_v   = val.pivot_table(index="codigo_ine", columns="ano", values="valor")
        anos_sort = sorted(pivot_v.columns)
        if len(anos_sort) >= 2:
            ano_ini = anos_sort[0]
            ano_fim = anos_sort[-1]
            n_anos  = ano_fim - ano_ini
            for cod in pivot_v.index:
                v_ini = pivot_v.loc[cod].get(ano_ini)
                v_fim = pivot_v.loc[cod].get(ano_fim)
                if v_ini is not None and v_fim is not None and n_anos > 0:
                    evolucao = (v_fim - v_ini) / n_anos
                    nome = df_metr[df_metr["codigo_ine"] == cod]["nome"].iloc[0]
                    df_metr = pd.concat([df_metr, pd.DataFrame([
                        row_base(cod, nome, int(ano_fim), "amb_evolucao_valorizacao_pp", round(evolucao, 3))
                    ])], ignore_index=True)

    print(f"     {len(df_metr)} registos · {df_metr['metrica_codigo'].nunique()} métricas")
    return df_metr


# ── Main ───────────────────────────────────────────────────────

def main():
    print("\n=== TRANSFORM · Cluster 2 — Ambiente ===\n")

    print("[ 2.1 ] Energia")
    df_consumos    = transform_consumos()
    df_contadores  = transform_contadores()
    df_comunidades = transform_comunidades()

    print("\n[ 2.2 ] Resíduos")
    df_residuos = transform_residuos()

    df_all = pd.concat(
        [df_consumos, df_contadores, df_comunidades, df_residuos],
        ignore_index=True,
    )
    df_all = normalizar_scores(df_all)

    # Alinhar schema com Governança (referência)
    df_all["valor_texto"] = None
    df_all["categoria"]   = None

    df_all.to_parquet(STAGING_DIR / "amb_transformed.parquet", index=False)

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