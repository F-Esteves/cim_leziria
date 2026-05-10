import pandas as pd
import yaml
import re
import unicodedata
from pathlib import Path

# ── Configuração ───────────────────────────────────────────────
CONFIG_PATH = Path("config/sources.yaml")
STAGING_DIR = Path("data/staging")
STAGING_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

RAW_DIR    = Path(cfg["raw_dir"])
MUNICIPIOS = cfg["municipios"]           # {codigo_ine: nome}


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


# ── Leitura genérica PORDATA eleitores ────────────────────────

def ler_pordata_eleicoes(path: Path, anos: list, skiprows: int = 11) -> pd.DataFrame:

    df_raw = pd.read_excel(path, skiprows=skiprows, header=0)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    col_tipo = df_raw.columns[0]  # "Âmbito Geográfico"
    col_mun  = df_raw.columns[1]  # "Anos" (nome do município na célula)

    df_raw = df_raw[df_raw[col_tipo].astype(str).str.strip() == "Município"].copy()
    df_raw = df_raw.dropna(subset=[col_mun])
    df_raw[col_mun] = df_raw[col_mun].astype(str).str.strip()
    df_raw["codigo_ine"] = df_raw[col_mun].apply(encontrar_codigo)
    df_filt = df_raw[df_raw["codigo_ine"].notna()].copy()

    if df_filt.empty:
        print(f"  ⚠ Nenhum município encontrado em {path.name}")
        return pd.DataFrame(columns=["codigo_ine", "nome", "ano", "eleitores", "votantes", "abstencao"])
    col_total   = {int(float(c.replace(".0", ""))): c
                   for c in df_raw.columns if re.fullmatch(r"\d{4}(\.0)?", str(c))}
    col_votantes = {int(float(c.replace(".1", ""))): c
                    for c in df_raw.columns if re.fullmatch(r"\d{4}\.1", str(c))}
    col_abstencao = {int(float(c.replace(".2", ""))): c
                     for c in df_raw.columns if re.fullmatch(r"\d{4}\.2", str(c))}

    rows = []
    for _, row in df_filt.iterrows():
        for ano in anos:
            rows.append({
                "codigo_ine": row["codigo_ine"],
                "nome":       row[col_mun],
                "ano":        ano,
                "eleitores":  safe_float(row[col_total[ano]])    if ano in col_total    else None,
                "votantes":   safe_float(row[col_votantes[ano]]) if ano in col_votantes else None,
                "abstencao":  safe_float(row[col_abstencao[ano]])if ano in col_abstencao else None,
            })
    return pd.DataFrame(rows)


# ── 1.1 Participação Cívica ────────────────────────────────────

def extrair_eleicoes_ar() -> pd.DataFrame:
    cfg_ar = cfg["governanca"]["participacao_civica"]["ar"]
    path   = RAW_DIR / cfg_ar["ficheiro"]
    print(f"  → Lendo {path.name}")
    df = ler_pordata_eleicoes(path, cfg_ar["anos"], cfg_ar["skiprows"])
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · anos {sorted(df['ano'].unique())}")
    return df


def extrair_eleicoes_autarquias() -> pd.DataFrame:
    cfg_aut = cfg["governanca"]["participacao_civica"]["autarquias"]
    path    = RAW_DIR / cfg_aut["ficheiro"]
    print(f"  → Lendo {path.name}")
    df_raw = pd.read_excel(path, skiprows=cfg_aut["skiprows"], header=0)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    col_tipo = df_raw.columns[0]
    col_mun  = df_raw.columns[1]

    df_raw = df_raw[df_raw[col_tipo].astype(str).str.strip() == "Município"].copy()
    df_raw = df_raw.dropna(subset=[col_mun])
    df_raw[col_mun] = df_raw[col_mun].astype(str).str.strip()
    df_raw["codigo_ine"] = df_raw[col_mun].apply(encontrar_codigo)
    df_filt = df_raw[df_raw["codigo_ine"].notna()].copy()

    # Detecta anos do bloco Total (colunas sem sufixo .1 ou .2)
    anos_detectados = sorted([
        int(float(c))
        for c in df_raw.columns
        if re.fullmatch(r"\d{4}(\.0)?", str(c))
    ])

    col_total    = {int(float(c.replace(".0",""))): c
                    for c in df_raw.columns if re.fullmatch(r"\d{4}(\.0)?", str(c))}
    col_votantes = {int(float(c.replace(".1",""))): c
                    for c in df_raw.columns if re.fullmatch(r"\d{4}\.1", str(c))}
    col_abstencao= {int(float(c.replace(".2",""))): c
                    for c in df_raw.columns if re.fullmatch(r"\d{4}\.2", str(c))}

    rows = []
    for _, row in df_filt.iterrows():
        for ano in anos_detectados:
            rows.append({
                "codigo_ine": row["codigo_ine"],
                "nome":       row[col_mun],
                "ano":        ano,
                "eleitores":  safe_float(row[col_total[ano]])     if ano in col_total     else None,
                "votantes":   safe_float(row[col_votantes[ano]])  if ano in col_votantes  else None,
                "abstencao":  safe_float(row[col_abstencao[ano]]) if ano in col_abstencao else None,
            })

    df = pd.DataFrame(rows)
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · anos {anos_detectados}")
    return df


def extrair_eleicoes_presidenciais() -> pd.DataFrame:
    cfg_p = cfg["governanca"]["participacao_civica"]["presidenciais"]
    path  = RAW_DIR / cfg_p["ficheiro"]
    print(f"  → Lendo {path.name}")
    df = ler_pordata_eleicoes(path, cfg_p["anos"], cfg_p["skiprows"])
    print(f"     {len(df)} registos · {df['codigo_ine'].nunique()} municípios · anos {sorted(df['ano'].unique())}")
    return df


def extrair_resultados_autarquias() -> pd.DataFrame:
    cfg_r       = cfg["governanca"]["participacao_civica"]["resultados_autarquias"]
    path        = RAW_DIR / cfg_r["ficheiro"]
    ano         = cfg_r["ano"]
    codigos_ods = cfg_r["codigos_ods"]

    print(f"  → Lendo {path.name}")
    df_raw = pd.read_excel(path, engine="odf", sheet_name=0, header=None)
    partido_nomes: dict[int, str] = {}
    for col_idx in range(8, 34):
        val = df_raw.iloc[3, col_idx] if col_idx < df_raw.shape[1] else None
        if val is not None and pd.notna(val) and str(val).strip() not in ("", "nan"):
            partido_nomes[col_idx] = str(val).strip()

    def categorizar_partido(p: str) -> str:
        pu = p.upper()
        if "PS" in pu and "PPD" not in pu and "PSD" not in pu:
            return "PS"
        if "PPD" in pu or "PSD" in pu:
            return "PSD/coligação"
        if "B.E." in pu or "BE" in pu or "LIVRE" in pu or "BLOCO" in pu or "L." in pu:
            return "Esquerda"
        return "GCE"

    rows = []

    for cod4, cod6 in codigos_ods.items():
        nome = MUNICIPIOS[cod4]
        mask = (df_raw[0].astype(str) == cod6) & (df_raw[3] == "CM")
        row  = df_raw[mask]
        if row.empty:
            print(f"  ⚠ {nome} ({cod6}) não encontrado no ODS")
            continue

        r = row.iloc[0]

        # ── Determinar partido vencedor ────────────────────────────────────
        # Passo 1: varrer todas as colunas de votos e encontrar a de maior valor
        votos: dict[int, float] = {}
        for col_idx in partido_nomes:
            v = safe_float(r[col_idx]) if col_idx < len(r) else None
            if v is not None and v > 0:
                votos[col_idx] = v

        winner_col = max(votos, key=votos.get) if votos else None
        # Passo 2: resolver o nome a partir da coluna vencedora
        if winner_col is None:
            partido = "Desconhecido"
        elif winner_col >= 27 and winner_col <= 29:
            idx_coal = winner_col - 27          # 0, 1 ou 2
            sig_raw  = r[34] if 34 < len(r) else None
            if sig_raw and pd.notna(sig_raw):
                partes = re.findall(r"\[([^\]]+)\]", str(sig_raw))
                partido = partes[idx_coal] if idx_coal < len(partes) else partes[0]
            else:
                partido = partido_nomes.get(winner_col, "Desconhecido")
        elif winner_col >= 30:
            idx_gce = winner_col - 30
            sig_raw = r[35] if 35 < len(r) else None
            if sig_raw and pd.notna(sig_raw):
                partes  = re.findall(r"\[([^\]]+)\]", str(sig_raw))
                partido = partes[idx_gce] if idx_gce < len(partes) else partes[0]
            else:
                partido = partido_nomes.get(winner_col, "Desconhecido")
        else:
            # Partido individual (cols 8–26)
            partido = partido_nomes.get(winner_col, "Desconhecido")

        rows.append({
            "codigo_ine": cod4,
            "nome":       nome,
            "ano":        ano,
            "partido":    partido,
            "categoria":  categorizar_partido(partido),
            "eleitores":  safe_float(r[4]),
            "votantes":   safe_float(r[5]),
        })

    df = pd.DataFrame(rows)
    print(f"     {len(df)} municípios · resultados 2025")
    return df


# ── 1.2 Digital ────────────────────────────────────────────────

def extrair_digital_serie(chave: str) -> pd.DataFrame:
    cfg_d = cfg["governanca"]["digital"][chave]
    path = RAW_DIR / cfg_d["ficheiro"]
    anos = sorted(cfg_d["anos"], reverse=True)  # INE: ordem mais recente → mais antigo

    print(f"  → Lendo {path.name}")

    # Deteção automática da linha com códigos INE (padrão 1D3XXXX)
    df_probe = pd.read_excel(path, header=None)
    primeira_linha = next(
        (i for i, val in enumerate(df_probe.iloc[:, 0].astype(str))
         if re.search(r"1D3\d{4}", val)),
        10  # fallback
    )
    # skiprows = primeira_linha - 1 para que o header seja a linha anterior aos dados
    df_raw = pd.read_excel(path, skiprows=primeira_linha - 1, header=0)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    col_mun = df_raw.columns[0]
    df_raw = df_raw.dropna(subset=[col_mun])
    df_raw[col_mun] = df_raw[col_mun].astype(str).str.strip()

    # Descarta linhas de rodapé
    df_raw = df_raw[~df_raw[col_mun].str.contains(
        r"INE|Última|http|fonte|©", case=False, regex=True, na=False
    )].copy()

    df_raw["codigo_ine"] = df_raw[col_mun].apply(encontrar_codigo)
    df_filt = df_raw[df_raw["codigo_ine"].notna()].copy()

    if df_filt.empty:
        print(f"  ⚠ Nenhum município encontrado em {path.name}")
        return pd.DataFrame(columns=["codigo_ine", "nome", "ano", "valor", "indicador"])

    data_cols_all = [c for c in df_filt.columns if c not in (col_mun, "codigo_ine")]
    data_cols = [
        c for c in data_cols_all
        if pd.to_numeric(df_filt[c], errors="coerce").notna().any()
    ][:len(anos)]

    if len(data_cols) < len(anos):
        print(f"  ⚠ Apenas {len(data_cols)} colunas de dados encontradas (esperadas {len(anos)})")

    colunas_anos = {data_cols[i]: anos[i] for i in range(len(data_cols))}

    rows = []
    for _, row in df_filt.iterrows():
        for col, ano in colunas_anos.items():
            rows.append({
                "codigo_ine": row["codigo_ine"],
                "nome": row[col_mun],
                "ano": ano,
                "valor": safe_float(row[col]),
                "indicador": chave,
            })

    df = pd.DataFrame(rows)
    print(f"  {len(df)} registos · {df['codigo_ine'].nunique()} municípios · anos {sorted(df['ano'].unique())}")
    return df


# ── Main ───────────────────────────────────────────────────────

def main():
    print("\n=== EXTRACT · Cluster 1 — Governança ===\n")

    print("[ 1.1 ] Participação Cívica")
    df_ar     = extrair_eleicoes_ar()
    df_aut    = extrair_eleicoes_autarquias()
    df_pres   = extrair_eleicoes_presidenciais()
    df_result = extrair_resultados_autarquias()

    df_ar.to_parquet(STAGING_DIR     / "gov_eleicoes_ar.parquet",           index=False)
    df_aut.to_parquet(STAGING_DIR    / "gov_eleicoes_autarquias.parquet",    index=False)
    df_pres.to_parquet(STAGING_DIR   / "gov_eleicoes_presidenciais.parquet", index=False)
    df_result.to_parquet(STAGING_DIR / "gov_resultados_autarquias.parquet",  index=False)

    print("\n[ 1.2 ] Digital")
    df_bl  = extrair_digital_serie("banda_larga")
    df_tel = extrair_digital_serie("telefone")
    df_tv  = extrair_digital_serie("tv")

    df_bl.to_parquet(STAGING_DIR  / "gov_banda_larga.parquet", index=False)
    df_tel.to_parquet(STAGING_DIR / "gov_telefone.parquet",    index=False)
    df_tv.to_parquet(STAGING_DIR  / "gov_tv.parquet",          index=False)

    print("\n✓ Extract concluído — ficheiros em data/staging/")
    for f in ["gov_eleicoes_ar.parquet", "gov_eleicoes_autarquias.parquet",
              "gov_eleicoes_presidenciais.parquet", "gov_resultados_autarquias.parquet",
              "gov_banda_larga.parquet", "gov_telefone.parquet", "gov_tv.parquet"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()