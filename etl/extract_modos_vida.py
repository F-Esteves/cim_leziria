import pandas as pd
import yaml
import re
import unicodedata
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
    "santarém": "1416",
    "azambuja": "1103",
    "golega":   "1412",
}
NORM_TO_INE.update({normalizar_texto(k): v for k, v in ALIASES.items()})


def extrair_cod_ine(valor_raw: str) -> str | None:
    """Trata os 3 formatos de código encontrados nos ficheiros:
       '1D31403: Almeirim' (INE), '1403:Almeirim' (INE Censos), nome puro (PORDATA)."""
    raw = str(valor_raw).strip()
    # INE formato 1D3: "1D31403: Almeirim"
    m = re.search(r"1D3(\d{4})", raw)
    if m:
        cod = m.group(1)
        return cod if cod in MUNICIPIOS else None
    # INE Censos: "1403:Almeirim" ou "1403: Almeirim"
    m = re.match(r"^(\d{4})\s*:", raw)
    if m:
        cod = m.group(1)
        return cod if cod in MUNICIPIOS else None
    # PORDATA: nome puro
    norm = normalizar_texto(raw)
    if norm in NORM_TO_INE:
        return NORM_TO_INE[norm]
    for chave, cod in NORM_TO_INE.items():
        if chave and chave in norm:
            return cod
    return None


def safe_float(v) -> float | None:
    if v is None:
        return None
    s = str(v).strip().replace(",", ".").replace("\xa0", "").replace(" ", "")
    if s in ("-", "…", "nan", ""):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def ano_limpo(v) -> int | None:
    """Normaliza anos com prefixo '┴ ' para inteiro."""
    try:
        return int(str(v).replace("┴", "").strip())
    except (ValueError, TypeError):
        return None


# ── Leitura PORDATA com múltiplos blocos de métricas ──────────

def ler_pordata_multi(path: Path, skiprows: int, metricas: list[str],
                      anos_cfg: list[int] | None = None) -> pd.DataFrame:

    df_raw = pd.read_excel(path, skiprows=skiprows, header=0)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    col_tipo = df_raw.columns[0]
    col_nome = df_raw.columns[1]

    # Filtrar só linhas de município
    df_raw = df_raw[df_raw[col_tipo].astype(str).str.strip() == "Município"].copy()
    df_raw = df_raw.dropna(subset=[col_nome])
    df_raw["codigo_ine"] = df_raw[col_nome].apply(extrair_cod_ine)
    df_filt = df_raw[df_raw["codigo_ine"].notna()].copy()

    if df_filt.empty:
        print(f"  ⚠ Nenhum município encontrado em {path.name}")
        return pd.DataFrame()

    # Mapear blocos por sufixo pandas
    blocos: dict[int, dict] = {}  # sufixo → {ano → col_name}
    for col in df_raw.columns:
        m = re.fullmatch(r"(┴ )?(\d{4})(\.\d+)?", str(col).strip())
        if not m:
            continue
        ano_val = ano_limpo(m.group(2))
        if ano_val is None:
            continue
        suf = int(m.group(3).lstrip(".")) if m.group(3) else 0
        blocos.setdefault(suf, {})[ano_val] = col

    rows = []
    for suf, nome_metrica in enumerate(metricas):
        mapa_ano_col = blocos.get(suf, {})
        anos_usar = [a for a in sorted(mapa_ano_col) if anos_cfg is None or a in anos_cfg]
        for ano in anos_usar:
            col = mapa_ano_col[ano]
            for _, row in df_filt.iterrows():
                v = safe_float(row[col])
                if v is not None:
                    rows.append({
                        "codigo_ine": row["codigo_ine"],
                        "nome":       row[col_nome],
                        "ano":        ano,
                        "metrica":    nome_metrica,
                        "valor":      v,
                    })

    return pd.DataFrame(rows)


# ── 4.1 Saúde ─────────────────────────────────────────────────

def extrair_hab_medico() -> pd.DataFrame:
    cfg_s = cfg["modos_vida"]["saude"]["hab_medico"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → Lendo {path.name}")
    df = ler_pordata_multi(path, cfg_s["skiprows"],
                           ["hab_medico", "hab_farmaceutico"],
                           cfg_s.get("anos"))
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · métricas {df['metrica'].unique().tolist()}")
    return df


def extrair_profissionais_saude() -> pd.DataFrame:
    cfg_s = cfg["modos_vida"]["saude"]["profissionais"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → Lendo {path.name}")
    df = ler_pordata_multi(path, cfg_s["skiprows"],
                           ["medicos", "enfermeiros", "farmaceuticos", "dentistas"],
                           cfg_s.get("anos"))
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · métricas {df['metrica'].unique().tolist()}")
    return df


def extrair_utentes_csp() -> pd.DataFrame:
 
    cfg_s = cfg["modos_vida"]["saude"]["utentes_csp"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → Lendo {path.name}")

    df_raw = pd.read_csv(path, sep=";", encoding="utf-8")
    col_aces  = cfg_s["col_aces"]
    col_per   = cfg_s["col_periodo"]
    col_ins   = cfg_s["col_inscritos"]
    col_mdf   = cfg_s["col_pct_mdf"]
    nome_aces = cfg_s["nome_aces"]

    # Filtrar ULS Lezíria (nome muda ao longo do tempo: "ACES Lezíria" → "CSP da ULS Lezíria")
    df = df_raw[df_raw[col_aces].str.contains(nome_aces, case=False, na=False)].copy()
    df["ano"] = df[col_per].str[:4].astype(int)
    df["mes"] = df[col_per].str[5:7].astype(int)

    # Snapshot de Dezembro de cada ano
    df_dez = df[df["mes"] == 12].copy()

    rows = []
    for _, row in df_dez.iterrows():
        ano = int(row["ano"])
        for cod in MUNICIPIOS:
            rows.append({
                "codigo_ine": cod,
                "nome":       MUNICIPIOS[cod],
                "ano":        ano,
                "metrica":    "utentes_csp",
                "valor":      safe_float(row[col_ins]),
            })
            rows.append({
                "codigo_ine": cod,
                "nome":       MUNICIPIOS[cod],
                "ano":        ano,
                "metrica":    "pct_utentes_mdf",
                "valor":      safe_float(row[col_mdf]),
            })

    df_out = pd.DataFrame(rows).dropna(subset=["valor"])
    print(f"     {len(df_out)} registos · anos {sorted(df_out['ano'].unique())}")
    return df_out


def extrair_consultas_csp() -> pd.DataFrame:
   
    cfg_s = cfg["modos_vida"]["saude"]["consultas_csp"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → Lendo {path.name}")

    df_raw = pd.read_csv(path, sep=";", encoding="utf-8")
    col_ent  = cfg_s["col_entidade"]
    col_per  = cfg_s["col_periodo"]
    col_pres = cfg_s["col_presenciais"]
    col_tot  = cfg_s["col_total"]
    nome     = cfg_s["nome_aces"]

    df = df_raw[df_raw[col_ent].str.contains(nome, case=False, na=False)].copy()
    df["ano"] = df[col_per].str[:4].astype(int)

    anual = df.groupby("ano")[[col_pres, col_tot]].sum().reset_index()

    rows = []
    for _, row in anual.iterrows():
        ano = int(row["ano"])
        for cod in MUNICIPIOS:
            rows.append({
                "codigo_ine": cod,
                "nome":       MUNICIPIOS[cod],
                "ano":        ano,
                "metrica":    "consultas_presenciais",
                "valor":      safe_float(row[col_pres]),
            })
            rows.append({
                "codigo_ine": cod,
                "nome":       MUNICIPIOS[cod],
                "ano":        ano,
                "metrica":    "consultas_total",
                "valor":      safe_float(row[col_tot]),
            })

    df_out = pd.DataFrame(rows).dropna(subset=["valor"])
    print(f"     {len(df_out)} registos · anos {sorted(df_out['ano'].unique())}")
    return df_out


# ── 4.2 Segurança ─────────────────────────────────────────────

def extrair_acidentes_vitimas() -> pd.DataFrame:
    cfg_s = cfg["modos_vida"]["seguranca"]["acidentes_vitimas"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → Lendo {path.name}")
    df = ler_pordata_multi(path, cfg_s["skiprows"],
                           ["acidentes_vitimas_1000hab"],
                           cfg_s.get("anos"))
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · anos {sorted(df['ano'].unique())}")
    return df


def extrair_feridos_mortos() -> pd.DataFrame:
    cfg_s = cfg["modos_vida"]["seguranca"]["feridos_mortos"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → Lendo {path.name}")
    df = ler_pordata_multi(path, cfg_s["skiprows"],
                           ["feridos_acidentes", "mortos_acidentes"],
                           cfg_s.get("anos"))
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · anos {sorted(df['ano'].unique())}")
    return df


def extrair_criminalidade() -> pd.DataFrame:

    cfg_s = cfg["modos_vida"]["seguranca"]["criminalidade"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    anos  = cfg_s["anos"]      # [2024, 2023, 2022, 2021]
    print(f"  → Lendo {path.name}")

    df_raw = pd.read_excel(path, header=None)

    # Categorias de crime (row 9, sem duplicados de bloco)
    cats_raw = df_raw.iloc[9].dropna().tolist()
    n_cats   = len(cats_raw) // len(anos)   # 7 categorias
    cats     = cats_raw[:n_cats]            # ['T: Total', '1: ...', ...]
    cat_keys = ["total", "integridade_fisica", "furto_esticao",
                "furto_veiculo", "alcool", "sem_habilitacao", "patrimonio"]

    rows = []
    # Dados começam na linha 11 (0-indexed)
    for i in range(11, len(df_raw)):
        raw_val = df_raw.iloc[i, 0]
        cod = extrair_cod_ine(str(raw_val))
        if cod is None:
            continue

        # As colunas de dados estão nas posições ímpares: 1, 3, 5, ...
        # Estrutura: [mun, val_2024_t, NaN, val_2024_cat1, NaN, ..., val_2023_t, NaN, ...]
        data_vals = [v for v in df_raw.iloc[i, 1:].tolist() if pd.notna(v)]

        for j, ano in enumerate(anos):
            base = j * n_cats
            for k, cat_key in enumerate(cat_keys):
                idx = base + k
                if idx < len(data_vals):
                    v = safe_float(data_vals[idx])
                    if v is not None:
                        rows.append({
                            "codigo_ine": cod,
                            "nome":       MUNICIPIOS.get(cod, cod),
                            "ano":        ano,
                            "metrica":    f"criminalidade_{cat_key}",
                            "valor":      v,
                        })

    df = pd.DataFrame(rows)
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · "
          f"anos {sorted(df['ano'].unique())} · categorias {df['metrica'].nunique()}")
    return df


# ── 4.3 Educação ──────────────────────────────────────────────

def extrair_sem_escolaridade() -> pd.DataFrame:
    """INE Censos 2021 — % pop. 15+ anos sem escolaridade completa (HM, H, M)."""
    cfg_s = cfg["modos_vida"]["educacao"]["sem_escolaridade"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → Lendo {path.name}")

    df_raw = pd.read_csv(path, sep=";", encoding="utf-8", header=0)
    col_geo = df_raw.columns[0]
    # Colunas: HM (total), H (homens), M (mulheres)
    col_hm  = [c for c in df_raw.columns if ":HM" in c or ":2021-T" in c][0]

    rows = []
    for _, row in df_raw.iterrows():
        cod = extrair_cod_ine(str(row[col_geo]))
        if cod is None:
            continue
        v = safe_float(row[col_hm])
        if v is not None:
            rows.append({
                "codigo_ine": cod,
                "nome":       MUNICIPIOS.get(cod, cod),
                "ano":        2021,
                "metrica":    "sem_escolaridade_pct",
                "valor":      v,
            })

    df = pd.DataFrame(rows)
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios")
    return df


def extrair_ensino_superior() -> pd.DataFrame:
    """INE Censos 2021 — população residente com ensino superior completo (total HM)."""
    cfg_s = cfg["modos_vida"]["educacao"]["ensino_superior"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → Lendo {path.name}")

    df_raw = pd.read_csv(path, sep=";", encoding="utf-8", header=0)
    col_geo  = df_raw.columns[0]
    # Primeira coluna após a geográfica = total HM
    col_total = df_raw.columns[1]

    rows = []
    for _, row in df_raw.iterrows():
        cod = extrair_cod_ine(str(row[col_geo]))
        if cod is None:
            continue
        # Valor pode ter espaços: "2 793" → 2793
        v = safe_float(str(row[col_total]).replace(" ", ""))
        if v is not None:
            rows.append({
                "codigo_ine": cod,
                "nome":       MUNICIPIOS.get(cod, cod),
                "ano":        2021,
                "metrica":    "ensino_superior_n",
                "valor":      v,
            })

    df = pd.DataFrame(rows)
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios")
    return df


# ── 4.4 Turismo ───────────────────────────────────────────────

def extrair_dormidas() -> pd.DataFrame:
    cfg_s = cfg["modos_vida"]["turismo"]["dormidas"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → Lendo {path.name}")
    df = ler_pordata_multi(path, cfg_s["skiprows"],
                           ["dormidas_100hab"],
                           cfg_s.get("anos"))
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · anos {sorted(df['ano'].unique())}")
    return df


# ── 4.5 Habitação ─────────────────────────────────────────────

def extrair_alojamentos() -> pd.DataFrame:

    cfg_s = cfg["modos_vida"]["habitacao"]["alojamentos"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → Lendo {path.name}")

    df_raw = pd.read_csv(path, sep=";", encoding="utf-8", header=0)
    col_geo = df_raw.columns[0]

    # Mapear colunas por posição (confirmado: col1=Total, col2=Familiares clássicos,
    # col3=Fam. não clássicos, col4=Familiares ocup. residência habitual,
    # col5=Familiares ocup. uso sazonal, col6=Familiares vagos, col7=Coletivos)
    col_map = {
        "alojamentos_total":          df_raw.columns[1],
        "alojamentos_familares":      df_raw.columns[2],
        "alojamentos_uso_sazonal":    df_raw.columns[5],
        "alojamentos_vagos":          df_raw.columns[6],
    }

    rows = []
    for _, row in df_raw.iterrows():
        cod = extrair_cod_ine(str(row[col_geo]))
        if cod is None:
            continue
        for metrica, col in col_map.items():
            v = safe_float(str(row[col]).replace(" ", ""))
            if v is not None:
                rows.append({
                    "codigo_ine": cod,
                    "nome":       MUNICIPIOS.get(cod, cod),
                    "ano":        2021,
                    "metrica":    metrica,
                    "valor":      v,
                })

    df = pd.DataFrame(rows)
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · "
          f"métricas {df['metrica'].unique().tolist()}")
    return df


# ── Main ───────────────────────────────────────────────────────

def main():
    print("\n=== EXTRACT · Cluster 4 — Modos de Vida ===\n")

    print("[ 4.1 ] Saúde")
    df_hab_med    = extrair_hab_medico()
    df_prof       = extrair_profissionais_saude()
    df_utentes    = extrair_utentes_csp()
    df_consultas  = extrair_consultas_csp()

    df_hab_med.to_parquet(STAGING_DIR   / "mdv_hab_medico.parquet",   index=False)
    df_prof.to_parquet(STAGING_DIR      / "mdv_profissionais.parquet", index=False)
    df_utentes.to_parquet(STAGING_DIR   / "mdv_utentes_csp.parquet",  index=False)
    df_consultas.to_parquet(STAGING_DIR / "mdv_consultas_csp.parquet",index=False)

    print("\n[ 4.2 ] Segurança")
    df_acid  = extrair_acidentes_vitimas()
    df_fer   = extrair_feridos_mortos()
    df_crim  = extrair_criminalidade()

    df_acid.to_parquet(STAGING_DIR / "mdv_acidentes_vitimas.parquet", index=False)
    df_fer.to_parquet(STAGING_DIR  / "mdv_feridos_mortos.parquet",    index=False)
    df_crim.to_parquet(STAGING_DIR / "mdv_criminalidade.parquet",     index=False)

    print("\n[ 4.3 ] Educação")
    df_educ_sem = extrair_sem_escolaridade()
    df_educ_sup = extrair_ensino_superior()

    df_educ_sem.to_parquet(STAGING_DIR / "mdv_sem_escolaridade.parquet", index=False)
    df_educ_sup.to_parquet(STAGING_DIR / "mdv_ensino_superior.parquet",  index=False)

    print("\n[ 4.4 ] Turismo")
    df_dorm = extrair_dormidas()
    df_dorm.to_parquet(STAGING_DIR / "mdv_dormidas.parquet", index=False)

    print("\n[ 4.5 ] Habitação")
    df_aloj = extrair_alojamentos()
    df_aloj.to_parquet(STAGING_DIR / "mdv_alojamentos.parquet", index=False)

    print("\n✓ Extract concluído — ficheiros em data/staging/")
    for f in ["mdv_hab_medico", "mdv_profissionais", "mdv_utentes_csp", "mdv_consultas_csp",
              "mdv_acidentes_vitimas", "mdv_feridos_mortos", "mdv_criminalidade",
              "mdv_sem_escolaridade", "mdv_ensino_superior", "mdv_dormidas", "mdv_alojamentos"]:
        print(f"  {f}.parquet")


if __name__ == "__main__":
    main()
