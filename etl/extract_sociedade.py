import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# ── Configuração ──────────────────────────────────────────────────────────────
CONFIG_PATH = Path("config/sources.yaml")
STAGING_DIR = Path("data/staging")
STAGING_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

RAW_DIR    = Path(cfg["raw_dir"])
MUNICIPIOS = cfg["municipios"]  # {codigo_ine: nome}

LEZIRIA = list(MUNICIPIOS.keys())   # 11 códigos INE

# ── Utilitários ───────────────────────────────────────────────────────────────

def normalizar(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip().lower()

NORM_TO_INE = {normalizar(nome): cod for cod, nome in MUNICIPIOS.items()}
ALIASES = {
    "salvaterra de magos": "1415",
    "santarém": "1416",
    "azambuja": "1103",
    "golega": "1412",
}
NORM_TO_INE.update(ALIASES)


def encontrar_codigo(valor_raw: str) -> str | None:
    raw = str(valor_raw).strip()
    # Código embutido no nome INE, ex: "1406: Cartaxo"
    m = re.match(r"(\d{4})\s*:", raw)
    if m and m.group(1) in MUNICIPIOS:
        return m.group(1)
    if raw in MUNICIPIOS:
        return raw
    norm = normalizar(raw)
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


# ── Parser genérico INE wide (Total / Masculino / Feminino × anos) ────────────

def parse_ine_sexo_wide(path: Path, metrica: str) -> pd.DataFrame:
    """
    Estrutura INE:
      Linha X:   ... Total ... Masculino ... Feminino ...
      Linha X+1: ... ano1 ano2 ... (anos como int/float)
      Linha X+2+: dados por município
    """
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


def extrair_censos_2021() -> pd.DataFrame:
    cfg_s = cfg["sociedade"]["censos_2021"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    engine = "xlrd" if str(path).endswith(".xls") else None
    print(f"  → {path.name}")

    kw = {"engine": engine} if engine else {}
    df_raw = pd.read_excel(path, header=None, **kw)

    records = []
    for _, row in df_raw.iterrows():
        cell = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        m = re.match(r"(\d{4})\s*:\s*(.+)", cell)
        if not m:
            continue
        codigo = m.group(1)
        if codigo not in LEZIRIA:
            continue
        nome = MUNICIPIOS[codigo]
        # Somar as 20 células da crosstab (cols 2+): sem subtotais no ficheiro,
        # cada célula é uma combinação única → soma = população total residente.
        total = 0.0
        for v in row.values[2:]:
            try:
                fv = float(v)
                if not np.isnan(fv):
                    total += fv
            except (TypeError, ValueError):
                pass
        records.append({
            "codigo_ine": codigo,
            "municipio":  nome,
            "ano":        cfg_s.get("ano", 2021),
            "metrica":    "Pop_residente_Censos2021",
            "valor":      total,
        })

    df = pd.DataFrame(records)
    print(f"     {len(df)} municípios · Censos 2021")
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

    print("\n[6.4] Censos 2021")
    df_c21 = extrair_censos_2021()
    df_c21.to_parquet(STAGING_DIR / "soc_censos_2021.parquet", index=False)

    print("\n[6.5] Áreas e pop 2011")
    df_ar = extrair_areas_2011()
    df_ar.to_parquet(STAGING_DIR / "soc_areas_2011.parquet", index=False)

    print("\n✓ Extract concluído — ficheiros em data/staging/")
    for f in ["soc_nados_vivos.parquet", "soc_obitos.parquet",
              "soc_pop_estrangeira.parquet", "soc_censos_2021.parquet",
              "soc_areas_2011.parquet"]:
        n = len(pd.read_parquet(STAGING_DIR / f))
        print(f"   {f}  ({n} registos)")


if __name__ == "__main__":
    main()
