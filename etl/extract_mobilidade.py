import pandas as pd
import yaml
from pathlib import Path

from etl.utils import (
    STAGING_DIR, MUNICIPIOS,
    encontrar_codigo as extrair_cod_ine,  
    safe_float as safe_num,
)

with open("config/sources.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

RAW_DIR = Path(cfg["raw_dir"])


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
    cfg_v  = cfg["mobilidade"]["veiculos"]
    path   = RAW_DIR / cfg_v["ficheiro"]
    anos   = cfg_v["anos"]   # já em ordem [2025, 2024, 2023, 2022, 2021]
    engine = "xlrd" if str(path).endswith(".xls") else None
    print(f"  → Lendo {path.name}")

    kw = {"engine": engine} if engine else {}
    df_raw = pd.read_excel(path, sheet_name="Quadro", header=None, **kw)

    n_cat      = 3   # Ligeiros, Pesados, Tratores agrícolas
    bloco_cols = n_cat * 2

    rows = []
    for _, row in df_raw.iterrows():
        nome_raw = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        cod_raw  = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        if not nome_raw or not cod_raw:
            continue

        codigo = extrair_cod_ine(cod_raw)
        if codigo not in MUNICIPIOS:
            continue
        nome = MUNICIPIOS[codigo]

        for i, ano in enumerate(anos):
            base = 2 + i * bloco_cols
            try:
                ligeiros = safe_num(row.iloc[base])
                pesados  = safe_num(row.iloc[base + 2])
                tratores = safe_num(row.iloc[base + 4])
            except IndexError:
                continue
            if all(v is None for v in (ligeiros, pesados, tratores)):
                continue

            total = sum(v or 0 for v in (ligeiros, pesados, tratores))
            for tipo, valor in [
                ("total",    round(total, 4)),
                ("ligeiros", ligeiros),
                ("pesados",  pesados),
                ("tratores", tratores),
            ]:
                if valor is not None:
                    rows.append({
                        "codigo_ine": codigo,
                        "nome":       nome,
                        "ano":        ano,
                        "tipo":       tipo,
                        "valor":      valor,
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
