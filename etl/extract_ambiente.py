import pandas as pd
import yaml
import unicodedata
import re
from pathlib import Path

CONFIG_PATH = Path("config/sources.yaml")
STAGING_DIR = Path("data/staging")
STAGING_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

RAW_DIR    = Path(cfg["raw_dir"])
MUNICIPIOS = cfg["municipios"]   # {codigo_ine: nome}

# ── Utilitários ────────────────────────────────────────────────

def normalizar_texto(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip().lower()

NORM_TO_INE = {normalizar_texto(nome): cod for cod, nome in MUNICIPIOS.items()}
ALIASES = {
    "salvaterra de magos": "1415",
    "santarém":            "1416",
    "azambuja":            "1103",
    "golega":              "1412",
}
NORM_TO_INE.update({normalizar_texto(k): v for k, v in ALIASES.items()})


def encontrar_codigo(valor_raw: str) -> str | None:
    raw = str(valor_raw).strip()

    m = re.search(r"1D3(\d{4})", raw)
    if m:
        cod = m.group(1)
        if cod in MUNICIPIOS:
            return cod

    if raw.isdigit() and raw in MUNICIPIOS:
        return raw

    norm = normalizar_texto(raw)

    if norm in NORM_TO_INE:
        return NORM_TO_INE[norm]

    for chave, cod in NORM_TO_INE.items():
        if chave and chave in norm:
            return cod

    return None


def safe_float(v) -> float | None:
    try:
        return float(str(v).replace(",", ".").replace(" ", "").replace("\xa0", ""))
    except Exception:
        return None


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
    cfg_r = cfg["ambiente"]["residuos"]
    path  = RAW_DIR / cfg_r["ficheiro"]
    anos  = sorted(cfg_r["anos"], reverse=True)   # INE: mais recente primeiro

    print(f"  → Lendo {path.name}")
    df_raw = pd.read_excel(path, skiprows=cfg_r["skiprows"], header=0)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    col_mun = df_raw.columns[0]   # nome do município
    col_cod = df_raw.columns[1]   # código INE ("1D31403")

    # Descarta linha de totais regionais ("Lezíria do Tejo" / "1D3") e rodapés
    df_raw = df_raw.dropna(subset=[col_mun]).copy()
    df_raw = df_raw[~df_raw[col_mun].astype(str).str.contains(
        r"INE|Última|http|fonte|©|^nan$", case=False, regex=True, na=False
    )].copy()

    def extrair_cod_ine(v) -> str | None:
        raw = str(v).strip()
        m = re.search(r"1D3(\d{4})", raw)
        if m:
            cod = m.group(1)
            return cod if cod in MUNICIPIOS else None
        return None

    df_raw["codigo_ine"] = df_raw[col_cod].apply(extrair_cod_ine)
    df_filt = df_raw[df_raw["codigo_ine"].notna()].copy()

    if df_filt.empty:
        print(f"  ⚠ Nenhum município encontrado em {path.name}")
        return pd.DataFrame()
    data_cols = [
        c for c in df_filt.columns
        if c not in (col_mun, col_cod, "codigo_ine")
        and pd.to_numeric(df_filt[c], errors="coerce").notna().any()
    ]
    n_destinos = 5
    n_anos     = len(anos)

    if len(data_cols) < n_anos * n_destinos:
        print(f"  ⚠ Apenas {len(data_cols)} colunas numéricas (esperadas {n_anos * n_destinos})")

    rows = []
    for i, ano in enumerate(anos):
        base = i * n_destinos
        if base + n_destinos > len(data_cols):
            print(f"  ⚠ Colunas insuficientes para o ano {ano} (base={base})")
            continue
        for _, row in df_filt.iterrows():
            rows.append({
                "codigo_ine":            row["codigo_ine"],
                "nome":                  row[col_mun],
                "ano":                   ano,
                "total_ton":             safe_float(row[data_cols[base]]),
                "aterro_ton":            safe_float(row[data_cols[base + 1]]),
                "val_energetica_ton":    safe_float(row[data_cols[base + 2]]),
                "val_organica_ton":      safe_float(row[data_cols[base + 3]]),
                "val_multimaterial_ton": safe_float(row[data_cols[base + 4]]),
            })

    df = pd.DataFrame(rows)
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · anos {sorted(df['ano'].dropna().unique())}")
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