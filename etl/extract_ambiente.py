import re
import pandas as pd
import yaml
from pathlib import Path

from etl.utils import (
    STAGING_DIR, MUNICIPIOS,
    encontrar_codigo, safe_float,
)

with open("config/sources.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

RAW_DIR = Path(cfg["raw_dir"])


# ── 2.1 Energia — Consumos ─────────────────────────────────────

def extrair_consumos() -> pd.DataFrame:
    cfg_c  = cfg["ambiente"]["energia"]["consumos"]
    path   = RAW_DIR / cfg_c["ficheiro"]
    anos   = cfg_c["anos"]

    print(f"  → Lendo {path.name}")
    df_raw = pd.read_excel(path)

    col_mun    = cfg_c["col_municipio"]   # "Concelho"
    col_ano    = cfg_c["col_ano"]         # "Ano"
    col_mes    = cfg_c["col_mes"]         # "Mês"
    col_tensao = cfg_c["col_tensao"]      # "Nível de Tensão"
    col_e      = cfg_c["col_energia"]     # "Energia Ativa"

    
    col_cod = cfg_c.get("col_codigo", "coddistritoconcelho")
    if col_cod in df_raw.columns:
        df_raw["codigo_ine"] = df_raw[col_cod].astype(str).str.strip()
        df_raw["codigo_ine"] = df_raw["codigo_ine"].where(
            df_raw["codigo_ine"].isin(MUNICIPIOS), other=None
        )
    else:
        df_raw["codigo_ine"] = df_raw[col_mun].apply(encontrar_codigo)

    df_filt = df_raw[
        df_raw["codigo_ine"].notna() &
        df_raw[col_ano].isin(anos)
    ].copy()

    if df_filt.empty:
        print("  ⚠ Nenhum registo encontrado")
        return pd.DataFrame()

    df_agg = (
        df_filt
        .groupby(["codigo_ine", col_mun, col_ano, col_mes, col_tensao])[col_e]
        .sum()
        .reset_index()
        .rename(columns={
            col_mun:    "nome",
            col_ano:    "ano",
            col_mes:    "mes",
            col_tensao: "tensao",
            col_e:      "energia_kwh",
        })
    )

    print(f"     {len(df_agg)} registos · {df_agg['codigo_ine'].nunique()} municípios · anos {sorted(df_agg['ano'].unique())}")
    return df_agg


# ── 2.1 Energia — Contadores ───────────────────────────────────

def extrair_contadores() -> pd.DataFrame:
    cfg_c = cfg["ambiente"]["energia"]["contadores"]
    path  = RAW_DIR / cfg_c["ficheiro"]
    anos  = cfg_c["anos"]

    print(f"  → Lendo {path.name}")
    df_raw = pd.read_excel(path)

    col_mun   = cfg_c["col_municipio"]   # "Concelho"
    col_ano   = cfg_c["col_ano"]         # "Ano"
    col_mes   = cfg_c["col_mes"]         # "Mês"
    col_cpes  = cfg_c["col_cpes"]        # "Número de CPE's"
    col_smart = cfg_c["col_smart"]       # "Inclui contador inteligente"

    
    col_cod = cfg_c.get("col_codigo", "Código Concelho")
    if col_cod in df_raw.columns:
        df_raw["codigo_ine"] = df_raw[col_cod].astype(str).str.strip()
        df_raw["codigo_ine"] = df_raw["codigo_ine"].where(
            df_raw["codigo_ine"].isin(MUNICIPIOS), other=None
        )
    else:
        df_raw["codigo_ine"] = df_raw[col_mun].apply(encontrar_codigo)

    df_filt = df_raw[
        df_raw["codigo_ine"].notna() &
        df_raw[col_ano].isin(anos)
    ].copy()

    if df_filt.empty:
        print("  ⚠ Nenhum registo encontrado")
        return pd.DataFrame()

    df_agg = (
        df_filt
        .groupby(["codigo_ine", col_mun, col_ano, col_mes, col_smart])[col_cpes]
        .sum()
        .reset_index()
        .rename(columns={
            col_mun:   "nome",
            col_ano:   "ano",
            col_mes:   "mes",
            col_smart: "smart",
            col_cpes:  "n_cpes",
        })
    )

    print(f"     {len(df_agg)} registos · {df_agg['codigo_ine'].nunique()} municípios · anos {sorted(df_agg['ano'].unique())}")
    return df_agg


# ── 2.1 Energia — Comunidades ──────────────────────────────────

def extrair_comunidades() -> pd.DataFrame:
    cfg_c = cfg["ambiente"]["energia"]["comunidades"]
    path  = RAW_DIR / cfg_c["ficheiro"]
    anos  = cfg_c["anos"]

    print(f"  → Lendo {path.name}")
    df_raw = pd.read_excel(path)

    col_mun      = cfg_c["col_municipio"]   # "Concelho"
    col_ano      = cfg_c["col_ano"]         # "Ano"
    col_mes      = cfg_c["col_mes"]         # "Mês"
    col_tipo     = cfg_c["col_tipo"]        # "Tipo ACC/CER"
    col_contagem = cfg_c["col_contagem"]    # "Contagem Tipo ACC/CER"


    col_cod = cfg_c.get("col_codigo", "Código Concelho")
    if col_cod in df_raw.columns:
        df_raw["codigo_ine"] = df_raw[col_cod].astype(str).str.strip()
        df_raw["codigo_ine"] = df_raw["codigo_ine"].where(
            df_raw["codigo_ine"].isin(MUNICIPIOS), other=None
        )
    else:
        df_raw["codigo_ine"] = df_raw[col_mun].apply(encontrar_codigo)

    df_filt = df_raw[
        df_raw["codigo_ine"].notna() &
        df_raw[col_ano].isin(anos)
    ].copy()

    if df_filt.empty:
        print("  ⚠ Nenhum registo encontrado (normal se dados ainda não publicados)")
        return pd.DataFrame(columns=["codigo_ine", "nome", "ano", "mes", "tipo", "contagem"])

    df_agg = (
        df_filt
        .groupby(["codigo_ine", col_mun, col_ano, col_mes, col_tipo])[col_contagem]
        .sum()
        .reset_index()
        .rename(columns={
            col_mun:      "nome",
            col_ano:      "ano",
            col_mes:      "mes",
            col_tipo:     "tipo",
            col_contagem: "contagem",
        })
    )

    print(f"     {len(df_agg)} registos · {df_agg['codigo_ine'].nunique()} municípios · anos {sorted(df_agg['ano'].unique())}")
    return df_agg


# ── 2.2 Resíduos ───────────────────────────────────────────────

def extrair_residuos() -> pd.DataFrame:
    cfg_r  = cfg["ambiente"]["residuos"]
    path   = RAW_DIR / cfg_r["ficheiro"]
    anos   = cfg_r["anos"]   # já em ordem [2024, 2023, 2022, 2021]
    engine = "xlrd" if str(path).endswith(".xls") else None
    print(f"  → Lendo {path.name}")

    kw = {"engine": engine} if engine else {}
    df_raw = pd.read_excel(path, sheet_name="Quadro", header=None, **kw)

    n_cat      = 4
    bloco_cols = n_cat * 2   # cada categoria ocupa 2 colunas (valor + NaN)

    rows = []
    for _, row in df_raw.iterrows():
        nome_raw = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        cod_raw  = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        if not nome_raw or not cod_raw:
            continue

        if cod_raw == "PT" or nome_raw == "Portugal":
            codigo, nome = "PT", "Portugal"
        else:
            codigo = encontrar_codigo(cod_raw)
            if codigo not in MUNICIPIOS:
                continue
            nome = MUNICIPIOS[codigo]

        for i, ano in enumerate(anos):
            base = 2 + i * bloco_cols   # colunas 0,1 são nome/código
            try:
                aterro  = safe_float(row.iloc[base])
                val_en  = safe_float(row.iloc[base + 2])
                val_org = safe_float(row.iloc[base + 4])
                val_mul = safe_float(row.iloc[base + 6])
            except IndexError:
                continue
            if all(v is None for v in (aterro, val_en, val_org, val_mul)):
                continue

            total = sum(v or 0 for v in (aterro, val_en, val_org, val_mul))
            rows.append({
                "codigo_ine":            codigo,
                "nome":                  nome,
                "ano":                   ano,
                "total_ton":             round(total, 1) if total else None,
                "aterro_ton":            aterro,
                "val_energetica_ton":    val_en,
                "val_organica_ton":      val_org,
                "val_multimaterial_ton": val_mul,
            })

    df = pd.DataFrame(rows)
    n_mun = df[df["codigo_ine"] != "PT"]["codigo_ine"].nunique()
    print(f"     {len(df)} registos · {n_mun} municípios + Portugal · anos {sorted(df['ano'].unique())}")
    return df


# ── Main ───────────────────────────────────────────────────────

def main():
    print("\n=== EXTRACT · Cluster 2 — Ambiente ===\n")

    print("[ 2.1 ] Energia")
    df_consumos    = extrair_consumos()
    df_contadores  = extrair_contadores()
    df_comunidades = extrair_comunidades()

    df_consumos.to_parquet(STAGING_DIR    / "amb_consumos.parquet",    index=False)
    df_contadores.to_parquet(STAGING_DIR  / "amb_contadores.parquet",  index=False)
    df_comunidades.to_parquet(STAGING_DIR / "amb_comunidades.parquet", index=False)

    print("\n[ 2.2 ] Resíduos")
    df_residuos = extrair_residuos()
    df_residuos.to_parquet(STAGING_DIR / "amb_residuos.parquet", index=False)

    print("\n✓ Extract concluído — ficheiros em data/staging/")
    for f in ["amb_consumos.parquet", "amb_contadores.parquet",
              "amb_comunidades.parquet", "amb_residuos.parquet"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()