import re
import pandas as pd
import yaml
from pathlib import Path

from etl.utils import (
    STAGING_DIR, MUNICIPIOS,
    encontrar_codigo as extrair_cod_ine, safe_float,
)

with open("config/sources.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

RAW_DIR = Path(cfg["raw_dir"])


def ano_limpo(v) -> int | None:
    try:
        return int(str(v).replace("┴", "").strip())
    except (ValueError, TypeError):
        return None


# ── Leitura PORDATA com múltiplos blocos de métricas ──────────

def ler_pordata_multi(path: Path, skiprows: int, metricas: list[str],
                      anos_cfg: list[int] | None = None,
                      incluir_cim: bool = False) -> pd.DataFrame:

    df_raw = pd.read_excel(path, skiprows=skiprows, header=0)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    col_tipo = df_raw.columns[0]
    col_nome = df_raw.columns[1]

    # Filtrar linhas de município (+ opcionalmente a linha CIM/NUTS III)
    tipos_aceites = ["Município"]
    if incluir_cim:
        tipos_aceites.append("NUTS III")
    df_raw = df_raw[df_raw[col_tipo].astype(str).str.strip().isin(tipos_aceites)].copy()
    df_raw = df_raw.dropna(subset=[col_nome])

    def cod_linha(row):
        if incluir_cim and str(row[col_tipo]).strip() == "NUTS III" \
                and "Lezíria do Tejo" in str(row[col_nome]):
            return "1D3"
        return extrair_cod_ine(row[col_nome])

    df_raw["codigo_ine"] = df_raw.apply(cod_linha, axis=1)
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
                           cfg_s.get("anos"), incluir_cim=True)
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

def _valor_ine(v):
    if isinstance(v, str) and v.strip() == "-":
        return 0.0
    return safe_float(v)


def _valor_ine_shifted(row, col: int):
    if col >= len(row):
        return None
    primario = _valor_ine(row.iloc[col])
    if primario is not None:
        return primario
    if col + 1 < len(row):
        seguinte = row.iloc[col + 1]
        if isinstance(seguinte, str) and seguinte.strip() == "-":
            return 0.0
    return None


def _linha_codigo_nome(nome_raw: str, cod_raw: str):
    if cod_raw == "PT" or nome_raw == "Portugal":
        return "PT", "Portugal"
    codigo = extrair_cod_ine(cod_raw)
    if codigo not in MUNICIPIOS:
        return None, None
    return codigo, MUNICIPIOS[codigo]


def extrair_ensino_superior_inscritos() -> pd.DataFrame:
    cfg_s  = cfg["modos_vida"]["educacao"]["ensino_superior_inscritos"]
    path   = RAW_DIR / cfg_s["ficheiro"]
    engine = "xlrd" if str(path).endswith(".xls") else None
    print(f"  → Lendo {path.name}")

    kw = {"engine": engine} if engine else {}
    df_raw = pd.read_excel(path, sheet_name="Quadro", header=None, **kw)

    # Detetar os blocos de ano na linha "Período de referência dos dados"
    linha_anos = df_raw.iloc[7]
    blocos = []  # [(col_inicio, ano), ...]
    for ci, v in enumerate(linha_anos):
        if pd.notna(v):
            m = re.match(r"(\d{4})\s*/\s*\d{4}", str(v).strip())
            if m:
                blocos.append((ci, int(m.group(1))))

    rows = []
    for _, row in df_raw.iterrows():
        nome_raw = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        cod_raw  = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        codigo, nome = _linha_codigo_nome(nome_raw, cod_raw)
        if codigo is None:
            continue

        for base, ano in blocos:
            v_pt  = _valor_ine_shifted(row, base)
            v_est = _valor_ine_shifted(row, base + 2)
            if v_pt is None and v_est is None:
                continue
            rows.append({
                "codigo_ine": codigo, "nome": nome, "ano": ano,
                "metrica": "ensino_superior_inscritos_n", "valor": (v_pt or 0) + (v_est or 0),
            })

    df = pd.DataFrame(rows)
    n_mun = df[df["codigo_ine"] != "PT"]["codigo_ine"].nunique()
    print(f"     {n_mun} municípios + Portugal · anos {sorted(df['ano'].unique())} "
          f"(maioria = 0, sem ensino superior local)")
    return df


def extrair_ensino_nao_superior() -> pd.DataFrame:
    cfg_s  = cfg["modos_vida"]["educacao"]["ensino_nao_superior_matriculados"]
    path   = RAW_DIR / cfg_s["ficheiro"]
    engine = "xlrd" if str(path).endswith(".xls") else None
    print(f"  → Lendo {path.name}")

    kw = {"engine": engine} if engine else {}
    df_raw = pd.read_excel(path, sheet_name="Quadro", header=None, **kw)

    niveis = [
        ("pre_escolar",       3),
        ("basico_1ciclo",     7),
        ("basico_2ciclo",    11),
        ("basico_3ciclo",    15),
        ("secundario",       19),
        ("pos_secundario",   23),
    ]

    rows = []
    ano_atual = None
    for _, row in df_raw.iterrows():
        periodo_cell = row.iloc[0]
        if pd.notna(periodo_cell):
            m = re.match(r"(\d{4})", str(periodo_cell).strip())
            if m:
                ano_atual = int(m.group(1))

        nome_raw = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        cod_raw  = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        codigo, nome = _linha_codigo_nome(nome_raw, cod_raw)
        if codigo is None or ano_atual is None:
            continue

        total_geral = 0.0
        linha_valida = False   # confirma que esta é mesmo uma linha de dados
        for nivel, base in niveis:
            publico = _valor_ine(row.iloc[base])     if base     < len(row) else None
            privado = _valor_ine(row.iloc[base + 2])  if base + 2 < len(row) else None
            if base >= len(row):
                continue
            linha_valida = True
            subtotal = (publico or 0) + (privado or 0)
            total_geral += subtotal
            rows.append({
                "codigo_ine": codigo, "nome": nome, "ano": ano_atual,
                "metrica": f"ensino_matriculados_{nivel}_n", "valor": subtotal,
            })

        if linha_valida:
            rows.append({
                "codigo_ine": codigo, "nome": nome, "ano": ano_atual,
                "metrica": "ensino_nao_superior_total_n", "valor": total_geral,
            })

    df = pd.DataFrame(rows)
    n_mun = df[df["codigo_ine"] != "PT"]["codigo_ine"].nunique()
    print(f"     {n_mun} municípios + Portugal · anos {sorted(df['ano'].unique())} · "
          f"níveis: {[n for n,_ in niveis]}")
    return df


def extrair_ensino_secundario_orientado() -> pd.DataFrame:
    cfg_s  = cfg["modos_vida"]["educacao"]["ensino_secundario_orientado"]
    path   = RAW_DIR / cfg_s["ficheiro"]
    engine = "xlrd" if str(path).endswith(".xls") else None
    print(f"  → Lendo {path.name}")

    kw = {"engine": engine} if engine else {}
    df_raw = pd.read_excel(path, sheet_name="Quadro", header=None, **kw)

    N_CATEGORIAS = 6
    BLOCO_COLS   = N_CATEGORIAS * 2

    linha_anos = df_raw.iloc[7]
    blocos = []
    for ci, v in enumerate(linha_anos):
        if pd.notna(v):
            m = re.match(r"(\d{4})\s*/\s*\d{4}", str(v).strip())
            if m:
                blocos.append((ci, int(m.group(1))))

    rows = []
    for _, row in df_raw.iterrows():
        nome_raw = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        cod_raw  = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        codigo, nome = _linha_codigo_nome(nome_raw, cod_raw)
        if codigo is None:
            continue

        for base, ano in blocos:
            valores = [
                _valor_ine_shifted(row, base + off)
                for off in range(0, BLOCO_COLS, 2)
            ]
            if all(v is None for v in valores):
                continue
            rows.append({
                "codigo_ine": codigo, "nome": nome, "ano": ano,
                "metrica": "ensino_secundario_orientado_n",
                "valor": sum(v or 0 for v in valores),
            })

    df = pd.DataFrame(rows)
    n_mun = df[df["codigo_ine"] != "PT"]["codigo_ine"].nunique()
    print(f"     {n_mun} municípios + Portugal · anos {sorted(df['ano'].unique())}")
    return df


def _extrair_taxa_ine_por_sexo(cfg_key: str, metrica_prefixo: str) -> pd.DataFrame:
    cfg_s  = cfg["modos_vida"]["educacao"][cfg_key]
    path   = RAW_DIR / cfg_s["ficheiro"]
    engine = "xlrd" if str(path).endswith(".xls") else None
    print(f"  → Lendo {path.name}")

    kw = {"engine": engine} if engine else {}
    df_raw = pd.read_excel(path, sheet_name="Quadro", header=None, **kw)

    rows = []
    ano_atual = None
    for _, row in df_raw.iterrows():
        periodo_cell = row.iloc[0]
        if pd.notna(periodo_cell):
            m = re.match(r"(\d{4})", str(periodo_cell).strip())
            if m:
                ano_atual = int(m.group(1))

        nome_raw = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        cod_raw  = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        codigo, nome = _linha_codigo_nome(nome_raw, cod_raw)
        if codigo is None or ano_atual is None:
            continue
        v_h = _valor_ine_shifted(row, 3)
        v_m = _valor_ine_shifted(row, 5)
        for sexo, v in [("h", v_h), ("m", v_m)]:
            if v is not None:
                rows.append({
                    "codigo_ine": codigo, "nome": nome, "ano": ano_atual,
                    "metrica": f"{metrica_prefixo}_{sexo}_pct", "valor": v,
                })

    df = pd.DataFrame(rows)
    n_mun = df[df["codigo_ine"] != "PT"]["codigo_ine"].nunique()
    print(f"     {n_mun} municípios + Portugal · anos {sorted(df['ano'].unique())}")
    return df


def extrair_taxa_retencao_desistencia() -> pd.DataFrame:
    return _extrair_taxa_ine_por_sexo("taxa_retencao_desistencia", "tx_retencao_desistencia")


def extrair_taxa_transicao_conclusao() -> pd.DataFrame:
    return _extrair_taxa_ine_por_sexo("taxa_transicao_conclusao", "tx_transicao_conclusao")


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


def extrair_alojamentos_tipo() -> pd.DataFrame:
    cfg_s = cfg["modos_vida"]["turismo"]["alojamentos_tipo"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → Lendo {path.name}")

    df_mun = ler_pordata_multi(path, cfg_s["skiprows"],
                               ["alojamentos_turisticos_total_n"],
                               cfg_s.get("anos"))

    # Portugal: mesma lógica de blocos/sufixos, mas linha "NUTS 2024"
    df_raw = pd.read_excel(path, skiprows=cfg_s["skiprows"], header=0)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    col_tipo, col_nome = df_raw.columns[0], df_raw.columns[1]
    linha_pt = df_raw[(df_raw[col_tipo].astype(str).str.strip() == "NUTS 2024") &
                       (df_raw[col_nome].astype(str).str.strip() == "Portugal")]

    rows_pt = []
    if not linha_pt.empty:
        row = linha_pt.iloc[0]
        for col in df_raw.columns:
            m = re.fullmatch(r"(┴ )?(\d{4})", str(col).strip())  # bloco 0 = sem sufixo
            if not m:
                continue
            ano_val = ano_limpo(m.group(2))
            anos_cfg = cfg_s.get("anos")
            if ano_val is None or (anos_cfg and ano_val not in anos_cfg):
                continue
            v = safe_float(row[col])
            if v is not None:
                rows_pt.append({
                    "codigo_ine": "PT", "nome": "Portugal", "ano": ano_val,
                    "metrica": "alojamentos_turisticos_total_n", "valor": v,
                })

    df = pd.concat([df_mun, pd.DataFrame(rows_pt)], ignore_index=True)
    n_mun = df[df["codigo_ine"] != "PT"]["codigo_ine"].nunique()
    print(f"     {len(df)} registos · {n_mun} municípios + Portugal · anos {sorted(df['ano'].unique())}")
    return df


# ── 4.5 Habitação ─────────────────────────────────────────────

def extrair_alojamentos() -> pd.DataFrame:

    cfg_s = cfg["modos_vida"]["habitacao"]["alojamentos"]
    path  = RAW_DIR / cfg_s["ficheiro"]
    print(f"  → Lendo {path.name}")

    df_raw = pd.read_csv(path, sep=";", encoding="utf-8", header=0)
    col_geo = df_raw.columns[0]

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
    df_educ_sup_insc = extrair_ensino_superior_inscritos()
    df_educ_nao_sup   = extrair_ensino_nao_superior()
    df_educ_sec_orient = extrair_ensino_secundario_orientado()
    df_tx_retencao   = extrair_taxa_retencao_desistencia()
    df_tx_transicao  = extrair_taxa_transicao_conclusao()

    df_educ_sem.to_parquet(STAGING_DIR / "mdv_sem_escolaridade.parquet", index=False)
    df_educ_sup.to_parquet(STAGING_DIR / "mdv_ensino_superior.parquet",  index=False)
    df_educ_sup_insc.to_parquet(STAGING_DIR   / "mdv_ensino_superior_inscritos.parquet", index=False)
    df_educ_nao_sup.to_parquet(STAGING_DIR    / "mdv_ensino_nao_superior.parquet",       index=False)
    df_educ_sec_orient.to_parquet(STAGING_DIR / "mdv_ensino_secundario_orientado.parquet", index=False)
    df_tx_retencao.to_parquet(STAGING_DIR     / "mdv_tx_retencao_desistencia.parquet",   index=False)
    df_tx_transicao.to_parquet(STAGING_DIR    / "mdv_tx_transicao_conclusao.parquet",    index=False)

    print("\n[ 4.4 ] Turismo")
    df_dorm = extrair_dormidas()
    df_aloj_tipo = extrair_alojamentos_tipo()
    df_dorm.to_parquet(STAGING_DIR / "mdv_dormidas.parquet", index=False)
    df_aloj_tipo.to_parquet(STAGING_DIR / "mdv_alojamentos_tipo.parquet", index=False)

    print("\n[ 4.5 ] Habitação")
    df_aloj = extrair_alojamentos()
    df_aloj.to_parquet(STAGING_DIR / "mdv_alojamentos.parquet", index=False)

    print("\n✓ Extract concluído — ficheiros em data/staging/")
    for f in ["mdv_hab_medico", "mdv_profissionais", "mdv_utentes_csp", "mdv_consultas_csp",
              "mdv_acidentes_vitimas", "mdv_feridos_mortos", "mdv_criminalidade",
              "mdv_sem_escolaridade", "mdv_ensino_superior",
              "mdv_ensino_superior_inscritos", "mdv_ensino_nao_superior",
              "mdv_ensino_secundario_orientado", "mdv_tx_retencao_desistencia",
              "mdv_tx_transicao_conclusao",
              "mdv_dormidas", "mdv_alojamentos_tipo", "mdv_alojamentos"]:
        print(f"  {f}.parquet")


if __name__ == "__main__":
    main()
