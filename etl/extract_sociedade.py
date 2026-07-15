import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from etl.utils import (
    STAGING_DIR, MUNICIPIOS,
    encontrar_codigo, safe_float,
)

# ── Configuração ──────────────────────────────────────────────────────────────
with open("config/sources.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

RAW_DIR = Path(cfg["raw_dir"])
LEZIRIA = list(MUNICIPIOS.keys())   # 11 códigos INE


# ── Parser genérico INE wide (Total / Masculino / Feminino × anos) ────────────

def parse_ine_sexo_wide(path: Path, metrica: str) -> pd.DataFrame:
    df_raw = pd.read_excel(path, header=None)

    # Encontrar linha com labels de sexo
    sexo_row_idx = None
    for i, row in df_raw.iterrows():
        labels = [str(v).strip() for v in row.values if pd.notna(v)]
        if "Total" in labels and "Masculino" in labels and "Feminino" in labels:
            sexo_row_idx = i
            break
    if sexo_row_idx is None:
        raise ValueError(f"Não encontrado cabeçalho Total/Masculino/Feminino em {path.name}")

    header_row_idx = sexo_row_idx + 1
    sexo_row  = df_raw.iloc[sexo_row_idx]
    year_row  = df_raw.iloc[header_row_idx]

    # Mapear índice de coluna → (sexo, ano)
    current_sexo = None
    col_map: dict[int, tuple[str, int]] = {}
    for ci, (sv, yv) in enumerate(zip(sexo_row.values, year_row.values)):
        if pd.notna(sv) and str(sv).strip() in ("Total", "Masculino", "Feminino"):
            current_sexo = str(sv).strip()
        if isinstance(yv, (int, float)) and 1960 < yv < 2030 and current_sexo:
            col_map[ci] = (current_sexo, int(yv))

    records = []
    for _, row in df_raw.iloc[header_row_idx + 1:].iterrows():
        municipio = row.iloc[1]
        if pd.isna(municipio) or str(municipio).strip() == "":
            continue
        municipio_str = str(municipio).strip()
        codigo = encontrar_codigo(municipio_str)
        if codigo not in LEZIRIA:
            continue
        nome = MUNICIPIOS[codigo]
        for ci, (sexo, yr) in col_map.items():
            val = row.iloc[ci]
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = np.nan
            records.append({
                "codigo_ine": codigo,
                "municipio":  nome,
                "ano":        yr,
                "sexo":       sexo,
                "metrica":    metrica,
                "valor":      val,
            })

    return pd.DataFrame(records)


# ── Extratores individuais ────────────────────────────────────────────────────

def extrair_nados_vivos() -> pd.DataFrame:
    cfg_s = cfg["sociedade"]["nados_vivos"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → {path.name}")
    df = parse_ine_sexo_wide(path, "Nados_vivos")
    df = df[df["sexo"] == "Total"].drop(columns=["sexo"])
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · "
          f"anos {sorted(df['ano'].unique())[:3]}…{sorted(df['ano'].unique())[-1]}")
    return df


def extrair_obitos() -> pd.DataFrame:
    cfg_s = cfg["sociedade"]["obitos"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → {path.name}")
    df = parse_ine_sexo_wide(path, "Obitos")
    df = df[df["sexo"] == "Total"].drop(columns=["sexo"])
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · "
          f"anos {sorted(df['ano'].unique())[:3]}…{sorted(df['ano'].unique())[-1]}")
    return df


def extrair_populacao_estrangeira() -> pd.DataFrame:
    cfg_s = cfg["sociedade"]["pop_estrangeira"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → {path.name}")
    df = parse_ine_sexo_wide(path, "Pop_estrangeira")
    df = df[df["sexo"] == "Total"].drop(columns=["sexo"])
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · "
          f"anos {sorted(df['ano'].unique())[:3]}…{sorted(df['ano'].unique())[-1]}")
    return df


def extrair_saldo_natural() -> pd.DataFrame:
    cfg_s  = cfg["sociedade"]["saldo_natural"]
    path   = RAW_DIR / cfg_s["ficheiro"]
    engine = "xlrd" if str(path).endswith(".xls") else None
    print(f"  → {path.name}")

    kw = {"engine": engine} if engine else {}
    df_raw = pd.read_excel(path, sheet_name="Quadro", header=None, **kw)

    # Encontrar a linha de anos (primeira linha com vários valores tipo ano,
    # que no ficheiro vêm como texto, ex.: "2025")
    def _como_ano(v) -> int | None:
        try:
            iv = int(float(v))
            return iv if 1990 < iv < 2100 else None
        except (TypeError, ValueError):
            return None

    anos_row_idx = None
    for i, row in df_raw.iterrows():
        vals = [v for v in row.values if _como_ano(v) is not None]
        if len(vals) >= 2:
            anos_row_idx = i
            break
    if anos_row_idx is None:
        raise ValueError(f"Não encontrada linha de anos em {path.name}")

    anos_row  = df_raw.iloc[anos_row_idx]
    ano_cols  = [(ci, _como_ano(v)) for ci, v in enumerate(anos_row.values) if _como_ano(v) is not None]

    records = []
    for _, row in df_raw.iloc[anos_row_idx + 2:].iterrows():
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

        for ci, ano in ano_cols:
            valor = safe_float(row.iloc[ci]) if ci < len(row) else None
            if valor is None:
                continue
            records.append({
                "codigo_ine": codigo,
                "municipio":  nome,
                "ano":        ano,
                "metrica":    "Saldo_natural",
                "valor":      valor,
            })

    df = pd.DataFrame(records)
    n_mun = df[df["codigo_ine"] != "PT"]["codigo_ine"].nunique()
    print(f"     {n_mun} municípios + Portugal · anos {sorted(df['ano'].unique())}")
    return df


def extrair_populacao_residente() -> pd.DataFrame:
    cfg_s  = cfg["sociedade"]["censos_2021"]
    path   = RAW_DIR / cfg_s["ficheiro"]
    engine = "xlrd" if str(path).endswith(".xls") else None
    print(f"  → {path.name}")

    kw = {"engine": engine} if engine else {}
    df_raw = pd.read_excel(path, sheet_name="Quadro", header=None, **kw)

    records = []
    ano_atual = None
    for _, row in df_raw.iterrows():
        ano_cell = row.iloc[0]
        if pd.notna(ano_cell):
            try:
                ano_atual = int(float(ano_cell))
            except (TypeError, ValueError):
                pass

        nome_raw = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        cod_raw  = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        if not nome_raw or ano_atual is None:
            continue

        val_h = safe_float(row.iloc[3]) if len(row) > 3 else None
        val_m = safe_float(row.iloc[5]) if len(row) > 5 else None
        if val_h is None and val_m is None:
            continue
        total = (val_h or 0) + (val_m or 0)

        if cod_raw == "PT" or nome_raw == "Portugal":
            codigo, nome = "PT", "Portugal"
        else:
            codigo = encontrar_codigo(cod_raw) or encontrar_codigo(nome_raw)
            if codigo not in LEZIRIA:
                continue
            nome = MUNICIPIOS[codigo]

        records.append({
            "codigo_ine": codigo,
            "municipio":  nome,
            "ano":        ano_atual,
            "metrica":    "Pop_residente",
            "valor":      total,
        })

    df = pd.DataFrame(records)
    n_mun = df[df["codigo_ine"] != "PT"]["codigo_ine"].nunique()
    n_pt  = (df["codigo_ine"] == "PT").sum()
    print(f"     {n_mun} municípios × {sorted(df['ano'].unique())} · "
          f"{n_pt} registos de Portugal")
    return df


def extrair_variacao_populacional() -> pd.DataFrame:
    cfg_v  = cfg["sociedade"]["variacao_populacional"]
    path   = RAW_DIR / cfg_v["ficheiro"]
    engine = "xlrd" if str(path).endswith(".xls") else None
    print(f"  → {path.name}")

    kw = {"engine": engine} if engine else {}
    df_raw = pd.read_excel(path, sheet_name="Quadro", header=None, **kw)

    def _como_ano(v) -> int | None:
        try:
            iv = int(float(v))
            return iv if 1990 < iv < 2100 else None
        except (TypeError, ValueError):
            return None

    anos_row_idx = None
    for i, row in df_raw.iterrows():
        vals = [v for v in row.values if _como_ano(v) is not None]
        if len(vals) >= 2:
            anos_row_idx = i
            break
    if anos_row_idx is None:
        raise ValueError(f"Não encontrada linha de anos em {path.name}")

    anos_row = df_raw.iloc[anos_row_idx]
    ano_cols = [(ci, _como_ano(v)) for ci, v in enumerate(anos_row.values) if _como_ano(v) is not None]

    records = []
    for _, row in df_raw.iloc[anos_row_idx + 2:].iterrows():
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

        for ci, ano in ano_cols:
            valor = safe_float(row.iloc[ci]) if ci < len(row) else None
            if valor is None:
                continue
            records.append({
                "codigo_ine": codigo, "municipio": nome, "ano": ano,
                "metrica": "Variacao_populacional", "valor": valor,
            })

    df = pd.DataFrame(records)
    n_mun = df[df["codigo_ine"] != "PT"]["codigo_ine"].nunique()
    print(f"     {n_mun} municípios + Portugal · anos {sorted(df['ano'].unique())}")
    return df


def extrair_populacao_media_anual() -> pd.DataFrame:
    cfg_v  = cfg["sociedade"]["populacao_media_anual"]
    path   = RAW_DIR / cfg_v["ficheiro"]
    engine = "xlrd" if str(path).endswith(".xls") else None
    print(f"  → {path.name}")

    kw = {"engine": engine} if engine else {}
    df_raw = pd.read_excel(path, sheet_name="Quadro", header=None, **kw)

    records = []
    ano_atual = None
    for _, row in df_raw.iterrows():
        ano_cell = row.iloc[0]
        if pd.notna(ano_cell):
            try:
                ano_atual = int(float(ano_cell))
            except (TypeError, ValueError):
                pass

        nome_raw = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        cod_raw  = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        if not nome_raw or ano_atual is None:
            continue

        codigo = encontrar_codigo(cod_raw)
        if codigo not in MUNICIPIOS:
            continue
        nome = MUNICIPIOS[codigo]

        valor = safe_float(row.iloc[3]) if len(row) > 3 else None
        if valor is None:
            continue
        records.append({
            "codigo_ine": codigo, "municipio": nome, "ano": ano_atual,
            "metrica": "Pop_media_anual", "valor": valor,
        })

    df = pd.DataFrame(records)
    print(f"     {df['codigo_ine'].nunique()} municípios · anos {sorted(df['ano'].unique())}")
    return df


def extrair_areas_2011() -> pd.DataFrame:
    cfg_s    = cfg["sociedade"]["areas_2011"]
    path     = RAW_DIR / cfg_s["ficheiro"]
    col_nome = cfg_s.get("col_nome", "Concelho")
    col_area = cfg_s.get("col_area_m2", "Área (m2)")
    col_pop  = cfg_s.get("col_pop", "População")
    print(f"  → {path.name}")

    df_raw = pd.read_excel(path)
    df_raw["codigo_ine"] = df_raw[col_nome].apply(encontrar_codigo)
    df_filt = df_raw[df_raw["codigo_ine"].isin(LEZIRIA)].copy()

    records = []
    for _, row in df_filt.iterrows():
        codigo   = row["codigo_ine"]
        nome     = MUNICIPIOS[codigo]
        area_m2  = safe_float(row.get(col_area))
        pop_2011 = safe_float(row.get(col_pop))
        area_km2 = area_m2 / 1_000_000 if area_m2 else None

        for metrica, valor in [
            ("Area_km2", area_km2),
            ("Pop_2011", pop_2011),
        ]:
            records.append({
                "codigo_ine": codigo,
                "municipio":  nome,
                "ano":        cfg_s.get("ano", 2011),
                "metrica":    metrica,
                "valor":      valor,
            })

    df = pd.DataFrame(records)
    print(f"     {df['codigo_ine'].nunique()} municípios · áreas e pop 2011")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== EXTRACT · Cluster 6 — Sociedade ===\n")

    print("[6.1] Nados-vivos")
    df_nv = extrair_nados_vivos()
    df_nv.to_parquet(STAGING_DIR / "soc_nados_vivos.parquet", index=False)

    print("\n[6.2] Óbitos")
    df_ob = extrair_obitos()
    df_ob.to_parquet(STAGING_DIR / "soc_obitos.parquet", index=False)

    print("\n[6.3] População estrangeira")
    df_est = extrair_populacao_estrangeira()
    df_est.to_parquet(STAGING_DIR / "soc_pop_estrangeira.parquet", index=False)

    print("\n[6.4] População residente (INE, anual 2021-2025, com Portugal)")
    df_c21 = extrair_populacao_residente()
    df_c21.to_parquet(STAGING_DIR / "soc_censos_2021.parquet", index=False)

    print("\n[6.5] Saldo natural (fonte direta INE, com Portugal)")
    df_sn = extrair_saldo_natural()
    df_sn.to_parquet(STAGING_DIR / "soc_saldo_natural.parquet", index=False)

    print("\n[6.6] Variação populacional (com Portugal)")
    df_vp = extrair_variacao_populacional()
    df_vp.to_parquet(STAGING_DIR / "soc_variacao_populacional.parquet", index=False)

    print("\n[6.7] População média anual residente")
    df_pma = extrair_populacao_media_anual()
    df_pma.to_parquet(STAGING_DIR / "soc_populacao_media_anual.parquet", index=False)

    print("\n[6.8] Áreas e pop 2011")
    df_ar = extrair_areas_2011()
    df_ar.to_parquet(STAGING_DIR / "soc_areas_2011.parquet", index=False)

    print("\n✓ Extract concluído — ficheiros em data/staging/")
    for f in ["soc_nados_vivos.parquet", "soc_obitos.parquet",
              "soc_pop_estrangeira.parquet", "soc_censos_2021.parquet",
              "soc_saldo_natural.parquet", "soc_variacao_populacional.parquet",
              "soc_populacao_media_anual.parquet", "soc_areas_2011.parquet"]:
        n = len(pd.read_parquet(STAGING_DIR / f))
        print(f"   {f}  ({n} registos)")


if __name__ == "__main__":
    main()
