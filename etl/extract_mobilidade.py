import pandas as pd
import yaml
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

def extrair_cod_ine(valor_raw: str) -> str | None:
    """Extrai código INE de 4 dígitos de strings no formato '1D31403: Almeirim'."""
    m = re.search(r"1D3(\d{4})", str(valor_raw))
    if m:
        cod = m.group(1)
        return cod if cod in MUNICIPIOS else None
    return None


def safe_num(v) -> float | None:
    """Converte para float; '-' e valores não numéricos devolvem None (→ 0 nos pontos)."""
    if v is None or str(v).strip() in ("-", "nan", ""):
        return None
    try:
        return float(str(v).replace(",", ".").replace("\xa0", "").replace(" ", ""))
    except (ValueError, TypeError):
        return None


def ler_ine_wide(path: Path, skiprows: int) -> tuple[pd.DataFrame, list[str]]:
    df_raw = pd.read_excel(path, skiprows=skiprows, header=0)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    col_mun = df_raw.columns[0]
    df_raw  = df_raw.dropna(subset=[col_mun]).copy()
    df_raw  = df_raw[~df_raw[col_mun].astype(str).str.contains(
        r"INE|Última|http|fonte|©|Mobi\.E|Sinais|Dado nulo|^nan$",
        case=False, regex=True, na=False
    )].copy()

    df_raw["codigo_ine"] = df_raw[col_mun].apply(extrair_cod_ine)
    df_filt = df_raw[df_raw["codigo_ine"].notna()].copy()

    # Colunas de dados: excluir col_mun e codigo_ine, manter só as com valores numéricos
    data_cols = [
        c for c in df_filt.columns
        if c not in (col_mun, "codigo_ine")
        and pd.to_numeric(df_filt[c], errors="coerce").notna().any()
    ]

    return df_filt, data_cols, col_mun


# ── 3.1 Veículos ───────────────────────────────────────────────

def extrair_veiculos() -> pd.DataFrame:
    cfg_v    = cfg["mobilidade"]["veiculos"]
    path     = RAW_DIR / cfg_v["ficheiro"]
    anos     = sorted(cfg_v["anos"], reverse=True)   # INE: 2024→2021
    n_tipos  = cfg_v["n_tipos"]                       # 4

    print(f"  → Lendo {path.name}")
    df_filt, data_cols, col_mun = ler_ine_wide(path, cfg_v["skiprows"])

    if df_filt.empty:
        print("  ⚠ Nenhum município encontrado")
        return pd.DataFrame()

    # Esperamos len(anos) × n_tipos colunas de dados
    esperadas = len(anos) * n_tipos
    if len(data_cols) < esperadas:
        print(f"  ⚠ {len(data_cols)} colunas numéricas (esperadas {esperadas})")

    tipos = ["total", "ligeiros", "pesados", "tratores"]
    rows  = []

    for i, ano in enumerate(anos):
        base = i * n_tipos
        for j, tipo in enumerate(tipos):
            col_idx = base + j
            if col_idx >= len(data_cols):
                break
            col = data_cols[col_idx]
            for _, row in df_filt.iterrows():
                v = safe_num(row[col])
                if v is not None:
                    rows.append({
                        "codigo_ine": row["codigo_ine"],
                        "nome":       row[col_mun],
                        "ano":        ano,
                        "tipo":       tipo,
                        "valor":      v,
                    })

    df = pd.DataFrame(rows)
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · "
          f"anos {sorted(df['ano'].unique())} · tipos {df['tipo'].unique().tolist()}")
    return df


# ── 3.2 Pontos de Carregamento — Tipo de Acesso ────────────────

def extrair_pontos_acesso() -> pd.DataFrame:
    cfg_a   = cfg["mobilidade"]["pontos_carregamento"]["acesso"]
    path    = RAW_DIR / cfg_a["ficheiro"]
    meses   = list(reversed(cfg_a["meses"]))   # Dez→Jan no ficheiro
    ano     = cfg_a["ano"]
    n_tipos = cfg_a["n_tipos"]   # 3

    print(f"  → Lendo {path.name}")
    df_filt, data_cols, col_mun = ler_ine_wide(path, cfg_a["skiprows"])

    if df_filt.empty:
        print("  ⚠ Nenhum município encontrado")
        return pd.DataFrame()

    tipos = ["total", "publico", "privado"]
    rows  = []

    for i, mes in enumerate(meses):
        base = i * n_tipos
        for j, tipo in enumerate(tipos):
            col_idx = base + j
            if col_idx >= len(data_cols):
                break
            col = data_cols[col_idx]
            for _, row in df_filt.iterrows():
                v = safe_num(row[col])
                rows.append({
                    "codigo_ine": row["codigo_ine"],
                    "nome":       row[col_mun],
                    "ano":        ano,
                    "mes":        mes,
                    "tipo":       tipo,
                    "n_pontos":   v if v is not None else 0,
                })

    df = pd.DataFrame(rows)
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · "
          f"{df['mes'].nunique()} meses · tipos {df['tipo'].unique().tolist()}")
    return df


# ── 3.2 Pontos de Carregamento — Tipo de Ponto ────────────────

def extrair_pontos_tipo() -> pd.DataFrame:
    cfg_t   = cfg["mobilidade"]["pontos_carregamento"]["tipo_ponto"]
    path    = RAW_DIR / cfg_t["ficheiro"]
    meses   = list(reversed(cfg_t["meses"]))   # Dez→Jan no ficheiro
    ano     = cfg_t["ano"]
    n_tipos = cfg_t["n_tipos"]   # 5

    print(f"  → Lendo {path.name}")
    df_filt, data_cols, col_mun = ler_ine_wide(path, cfg_t["skiprows"])

    if df_filt.empty:
        print("  ⚠ Nenhum município encontrado")
        return pd.DataFrame()

    tipos = ["total", "normal", "semirrapido", "rapido", "ultrarapido"]
    rows  = []

    for i, mes in enumerate(meses):
        base = i * n_tipos
        for j, tipo in enumerate(tipos):
            col_idx = base + j
            if col_idx >= len(data_cols):
                break
            col = data_cols[col_idx]
            for _, row in df_filt.iterrows():
                v = safe_num(row[col])
                rows.append({
                    "codigo_ine": row["codigo_ine"],
                    "nome":       row[col_mun],
                    "ano":        ano,
                    "mes":        mes,
                    "tipo":       tipo,
                    "n_pontos":   v if v is not None else 0,
                })

    df = pd.DataFrame(rows)
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · "
          f"{df['mes'].nunique()} meses · tipos {df['tipo'].unique().tolist()}")
    return df


# ── Main ───────────────────────────────────────────────────────

def main():
    print("\n=== EXTRACT · Cluster 3 — Mobilidade ===\n")

    print("[ 3.1 ] Veículos")
    df_veiculos = extrair_veiculos()
    df_veiculos.to_parquet(STAGING_DIR / "mob_veiculos.parquet", index=False)

    print("\n[ 3.2 ] Pontos de Carregamento VE")
    df_acesso = extrair_pontos_acesso()
    df_tipo   = extrair_pontos_tipo()
    df_acesso.to_parquet(STAGING_DIR / "mob_pontos_acesso.parquet", index=False)
    df_tipo.to_parquet(STAGING_DIR   / "mob_pontos_tipo.parquet",   index=False)

    print("\n✓ Extract concluído — ficheiros em data/staging/")
    for f in ["mob_veiculos.parquet", "mob_pontos_acesso.parquet", "mob_pontos_tipo.parquet"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
