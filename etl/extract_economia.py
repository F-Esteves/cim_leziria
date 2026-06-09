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

def ler_ine_pares_anos(path: Path, anos: list[int]) -> pd.DataFrame:
    """INE: col0=geo, cols ímpares (1,3,5...) = valores por ano."""
    df_raw = pd.read_excel(path, sheet_name=0, skiprows=0, header=None)
    df_dados = df_raw.iloc[9:].copy()
    cols_val = [1 + 2*i for i in range(len(anos))]
    df = df_dados[[0] + cols_val].copy()
    df.columns = ["geo"] + anos
    df["codigo_ine"] = df["geo"].apply(encontrar_codigo)
    df = df[df["codigo_ine"].notna()].copy()
    df["nome"] = df["codigo_ine"].map(MUNICIPIOS)
    df = df.drop(columns=["geo"])
    df_long = df.melt(id_vars=["codigo_ine", "nome"], var_name="ano", value_name="valor")
    df_long["ano"] = df_long["ano"].astype(int)
    df_long["valor"] = df_long["valor"].apply(safe_float)
    return df_long.dropna(subset=["valor"]).reset_index(drop=True)

def ler_pordata_transposto(path: Path, nome_serie: str) -> pd.DataFrame:
    """PORDATA: sheet Quadro, skiprows=11, anos em colunas."""
    df_raw = pd.read_excel(path, sheet_name="Quadro", skiprows=11, header=0)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    col_tipo = df_raw.columns[0]
    col_mun  = df_raw.columns[1]
    df_raw = df_raw[df_raw[col_tipo].astype(str).str.strip() == "Município"].copy()
    df_raw = df_raw.dropna(subset=[col_mun])
    df_raw[col_mun] = df_raw[col_mun].astype(str).str.strip()
    df_raw["codigo_ine"] = df_raw[col_mun].apply(encontrar_codigo)
    df_filt = df_raw[df_raw["codigo_ine"].notna()].copy()
    anos_cols = [c for c in df_raw.columns if re.fullmatch(r"\d{4}", str(c))]
    rows = []
    for _, row in df_filt.iterrows():
        for col_ano in anos_cols:
            v = safe_float(row[col_ano])
            if v is not None:
                rows.append({
                    "codigo_ine": row["codigo_ine"],
                    "nome": row[col_mun],
                    "ano": int(col_ano),
                    "valor": v,
                })
    df = pd.DataFrame(rows)
    print(f"   {nome_serie}: {len(df)} registos · {df['codigo_ine'].nunique()} mun · anos {sorted(df['ano'].unique())}")
    return df


# ── 5.1 Emprego e Estrutura ───────────────────────────────────────────────────

def extrair_emprego_conta_outrem() -> pd.DataFrame:
    cfg_e = cfg["economia"]["emprego"]["conta_outrem"]
    path  = RAW_DIR / cfg_e["ficheiro"]
    print(f" → {path.name}")
    df_raw = pd.read_excel(path, sheet_name=0, skiprows=0, header=None)

    # Identificar linha de dados (col[0] contém "1D3")
    data_start = next(
        (i for i, v in enumerate(df_raw.iloc[:, 0].astype(str))
         if re.search(r"1D3\d{4}", v)), None
    )
    if data_start is None:
        print("  ⚠ Nenhuma linha de dados encontrada")
        return pd.DataFrame(columns=["codigo_ine","nome","ano","valor"])

    # Identificar o ano (linha 7, col[1])
    ano_val = None
    for i in range(data_start):
        v = df_raw.iloc[i, 1]
        if isinstance(v, (int, float)) and 2010 < v < 2030:
            ano_val = int(v)
            break
        if isinstance(v, str) and re.fullmatch(r"\d{4}", v.strip()):
            ano_val = int(v.strip())
            break
    if ano_val is None:
        ano_val = 2023
        print(f"  ⚠ Ano não detectado — assumindo {ano_val}")

    # Colunas de valores: todas as ímpares com dados numéricos
    df_dados = df_raw.iloc[data_start:].copy()
    df_dados = df_dados[df_dados.iloc[:, 0].astype(str).str.contains(r"1D3\d{4}", na=False)]

    cols_val = [c for c in range(1, df_raw.shape[1], 2)]

    rows = []
    for _, row in df_dados.iterrows():
        cod = encontrar_codigo(str(row.iloc[0]))
        if cod is None:
            continue
        # Total = soma de todas as profissões
        vals = [safe_float(row.iloc[c]) for c in cols_val if c < len(row)]
        total = sum(v for v in vals if v is not None)
        if total > 0:
            rows.append({
                "codigo_ine": cod,
                "nome":       MUNICIPIOS[cod],
                "ano":        ano_val,
                "valor":      total,
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["codigo_ine","nome","ano","valor"])
    print(f"   {len(df)} registos · {df['codigo_ine'].nunique()} mun · anos {sorted(df['ano'].unique()) if not df.empty else []}")
    return df


def extrair_emprego_censos() -> pd.DataFrame:
    cfg_c = cfg["economia"]["emprego"]["censos_bruto"]
    path  = RAW_DIR / cfg_c["ficheiro"]
    print(f" → {path.name}")
    df_raw = pd.read_excel(path, sheet_name=0, skiprows=0, header=None, engine="xlrd")

    SECTORES  = ["primario", "secundario", "terciario_social", "terciario_econ"]
    SITUACOES = ["emp_lt10", "emp_ge10", "conta_propria", "conta_outrem", "outra"]

    # Construir mapa col_idx → (sexo, setor, situacao) com base na estrutura confirmada
    col_map: dict[int, tuple[str, str, str]] = {}
    for s_i, sexo in enumerate(["H", "M"]):
        base = 2 + s_i * 40          # H começa em col 2, M em col 42
        for sec_i, setor in enumerate(SECTORES):
            for sit_i, sit in enumerate(SITUACOES):
                col = base + sec_i * 10 + sit_i * 2
                col_map[col] = (sexo, setor, sit)

    records = []
    for _, row in df_raw.iloc[13:].iterrows():
        # col1 = 'DDDD: Nome' (ex: '1403: Almeirim')
        cell = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        m = re.match(r"(\d{4})\s*:\s*(.+)", cell)
        if not m:
            continue
        codigo = m.group(1)
        if codigo not in MUNICIPIOS:
            continue
        nome = MUNICIPIOS[codigo]
        ano_val = cfg_c.get("ano", 2021)
        for col, (sexo, setor, sit) in col_map.items():
            if col >= len(row):
                continue
            v = safe_float(row.iloc[col])
            if v is not None:
                records.append({
                    "codigo_ine": codigo,
                    "nome":       nome,
                    "sexo":       sexo,
                    "setor":      setor,
                    "situacao":   sit,
                    "ano":        ano_val,
                    "valor":      v,
                })

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["codigo_ine","nome","sexo","setor","situacao","ano","valor"])
    print(f"   Censos: {len(df)} linhas · {df['codigo_ine'].nunique()} mun "
          f"· sectores: {sorted(df['setor'].unique().tolist()) if not df.empty else []}")
    return df


# ── 5.2 Rendimento e Capacidade Fiscal ────────────────────────────────────────

def extrair_rendimento_bruto() -> pd.DataFrame:
    cfg_r = cfg["economia"]["rendimento"]["rendimento_bruto"]
    path  = RAW_DIR / cfg_r["ficheiro"]
    anos  = cfg_r["anos"]
    print(f" → {path.name}")
    df = ler_ine_pares_anos(path, anos)
    print(f"   {len(df)} registos · {df['codigo_ine'].nunique()} mun · anos {sorted(df['ano'].unique())}")
    return df

def extrair_irs_liquidado() -> pd.DataFrame:
    cfg_i = cfg["economia"]["rendimento"]["irs_liquidado"]
    path  = RAW_DIR / cfg_i["ficheiro"]
    anos  = cfg_i["anos"]
    print(f" → {path.name}")
    df = ler_ine_pares_anos(path, anos)
    print(f"   {len(df)} registos · {df['codigo_ine'].nunique()} mun · anos {sorted(df['ano'].unique())}")
    return df

def extrair_poder_compra_per_capita() -> pd.DataFrame:
    cfg_pc = cfg["economia"]["rendimento"]["poder_compra_per_capita"]
    path   = RAW_DIR / cfg_pc["ficheiro"]
    print(f" → {path.name}")
    return ler_pordata_transposto(path, "IPC per capita")

def extrair_proporcao_poder_compra() -> pd.DataFrame:
    cfg_pr = cfg["economia"]["rendimento"]["proporcao_poder_compra"]
    path   = RAW_DIR / cfg_pr["ficheiro"]
    print(f" → {path.name}")
    return ler_pordata_transposto(path, "Proporção PC")


# ── 5.3 Empresarialidade ──────────────────────────────────────────────────────

def extrair_demografica_empresas() -> dict[str, pd.DataFrame]:
    cfg_emp = cfg["economia"]["empresarialidade"]["nascimentos_mortes_sobrevivencia"]
    path    = RAW_DIR / cfg_emp["ficheiro"]
    print(f" → {path.name}")
    df_raw  = pd.read_excel(path, sheet_name="Quadro", skiprows=11, header=0)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    col_tipo = df_raw.columns[0]
    col_mun  = df_raw.columns[1]
    df_raw = df_raw[df_raw[col_tipo].astype(str).str.strip() == "Município"].copy()
    df_raw = df_raw.dropna(subset=[col_mun])
    df_raw[col_mun] = df_raw[col_mun].astype(str).str.strip()
    df_raw["codigo_ine"] = df_raw[col_mun].apply(encontrar_codigo)
    df_filt = df_raw[df_raw["codigo_ine"].notna()].copy()

    def extrair_bloco(sufixo: str) -> pd.DataFrame:
        pat = r"\d{4}(\.0)?" if sufixo == "" else rf"\d{{4}}\.{sufixo}"
        cols = [c for c in df_raw.columns if re.fullmatch(pat, str(c))]
        rows = []
        for _, row in df_filt.iterrows():
            for col in cols:
                ano_str = str(col).split(".")[0]
                v = safe_float(row[col])
                if v is not None:
                    rows.append({
                        "codigo_ine": row["codigo_ine"],
                        "nome": row[col_mun],
                        "ano": int(float(ano_str)),
                        "valor": v,
                    })
        return pd.DataFrame(rows)

    nasc = extrair_bloco(""); mort = extrair_bloco("1"); sobr = extrair_bloco("2")
    for nome, df in [("Nascidas", nasc), ("Mortas", mort), ("Sobrev.", sobr)]:
        print(f"   {nome}: {len(df)} reg · anos {sorted(df['ano'].unique()) if not df.empty else []}")
    return {"nascidas": nasc, "mortas": mort, "sobreviventes": sobr}


def extrair_volume_negocios() -> dict[str, pd.DataFrame]:
    cfg_vn = cfg["economia"]["empresarialidade"]["volume_negocios"]
    path = RAW_DIR / cfg_vn["ficheiro"]
    anos = cfg_vn["anos"]
    print(f" → {path.name}")

    df_raw = pd.read_excel(path, sheet_name=0, skiprows=0, header=None)

    rows_sect = []
    for _, row in df_raw.iterrows():
        cod = encontrar_codigo(row.iloc[0])
        if cod is None:
            continue

        cae_raw = str(row.iloc[1]).strip().upper() if pd.notna(row.iloc[1]) else ""
        if cae_raw in ("", "NAN", "TOTAL", "T"):
            continue

        nome = MUNICIPIOS[cod]

        for i, ano in enumerate(anos):
            col_idx = 2 + 2 * i
            if col_idx >= len(row):
                continue

            v = safe_float(row.iloc[col_idx])
            if v is None:
                continue

            rows_sect.append({
                "codigo_ine": cod,
                "nome": nome,
                "cae": cae_raw,
                "ano": int(ano),
                "valor": v,
            })

    df_sect = pd.DataFrame(rows_sect)

    if df_sect.empty:
        print(" ⚠ VN sectores vazio")
        df_tot = pd.DataFrame(columns=["codigo_ine", "nome", "ano", "valor"])
        return {"total": df_tot, "sectores": df_sect}

    df_tot = (
        df_sect
        .groupby(["codigo_ine", "nome", "ano"], as_index=False)["valor"]
        .sum()
    )

    print(
        f" VN total: {len(df_tot)} reg · {df_tot['codigo_ine'].nunique()} mun · "
        f"anos {sorted(df_tot['ano'].unique())}"
    )
    print(
        f" VN sectores: {len(df_sect)} reg · {df_sect['codigo_ine'].nunique()} mun"
    )

    return {"total": df_tot, "sectores": df_sect}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== EXTRACT · Cluster 5 — Economia ===\n")

    print("[ 5.1 ] Emprego e Estrutura")
    df_co   = extrair_emprego_conta_outrem()
    df_cens = extrair_emprego_censos()
    df_co.to_parquet(STAGING_DIR   / "eco_emprego_conta_outrem.parquet", index=False)
    df_cens.to_parquet(STAGING_DIR / "eco_emprego_censos_bruto.parquet", index=False)

    print("\n[ 5.2 ] Rendimento e Capacidade Fiscal")
    df_rb  = extrair_rendimento_bruto()
    df_irs = extrair_irs_liquidado()
    df_ipc = extrair_poder_compra_per_capita()
    df_ppc = extrair_proporcao_poder_compra()
    df_rb.to_parquet(STAGING_DIR  / "eco_rendimento_bruto.parquet", index=False)
    df_irs.to_parquet(STAGING_DIR / "eco_irs_liquidado.parquet", index=False)
    df_ipc.to_parquet(STAGING_DIR / "eco_poder_compra_per_capita.parquet", index=False)
    df_ppc.to_parquet(STAGING_DIR / "eco_proporcao_poder_compra.parquet", index=False)

    print("\n[ 5.3 ] Empresarialidade")
    demog = extrair_demografica_empresas()
    vn    = extrair_volume_negocios()
    demog["nascidas"].to_parquet(STAGING_DIR    / "eco_empresas_nascidas.parquet", index=False)
    demog["mortas"].to_parquet(STAGING_DIR      / "eco_empresas_mortas.parquet", index=False)
    demog["sobreviventes"].to_parquet(STAGING_DIR / "eco_empresas_sobreviventes.parquet", index=False)
    vn["total"].to_parquet(STAGING_DIR    / "eco_volume_negocios_total.parquet", index=False)
    vn["sectores"].to_parquet(STAGING_DIR / "eco_volume_negocios_sectores.parquet", index=False)

    print("\n✓ Extract concluído — ficheiros em data/staging/")


if __name__ == "__main__":
    main()