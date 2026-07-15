import sys
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
import yaml

# ── Caminhos e configuração ────────────────────────────────────────────────────

CONFIG_PATH = Path("config/sources.yaml")
STAGING_DIR = Path("data/staging")

with open(CONFIG_PATH, encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

DB: dict = _cfg["database"]

# ── Municípios da CIM Lezíria do Tejo — lidos de config/sources.yaml ──────────

MUNICIPIOS: dict[str, str] = _cfg["municipios"]

# Dados geográficos (área) — secção municipios_geo do sources.yaml
_GEO: dict = _cfg.get("municipios_geo", {})

PT_CODIGO: str = "PT"
PT_NOME:   str = "Portugal"
ANO_REFERENCIA_POPULACAO: int = 2025

# ── Ligação à base de dados ────────────────────────────────────────────────────

def conectar() -> psycopg2.extensions.connection:
    """Abre e devolve uma ligação psycopg2 com os parâmetros de config/sources.yaml."""
    return psycopg2.connect(
        host=DB["host"],
        port=DB["port"],
        dbname=DB["dbname"],
        user=DB["user"],
        password=DB["password"],
    )


# ── Helpers de dimensão ────────────────────────────────────────────────────────

_PK_MAP = {
    "dim_municipio": "municipio_id",
    "dim_metrica":   "metrica_id",
    "dim_tempo":     "tempo_id",
}



def seed_dim_municipio(conn) -> int:
    rows = []
    for codigo, nome in MUNICIPIOS.items():
        geo      = _GEO.get(codigo, {})
        area     = geo.get("area_km2")
        rows.append((codigo, nome, area, "concelho"))
    rows.append((PT_CODIGO, PT_NOME, None, "nacional"))

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO dim_municipio (codigo_ine, nome, area_km2, tipo)
            VALUES %s
            ON CONFLICT (codigo_ine) DO UPDATE
                SET nome     = EXCLUDED.nome,
                    area_km2 = EXCLUDED.area_km2,
                    tipo     = EXCLUDED.tipo
            """,
            rows,
        )
    conn.commit()
    print(f"  ✓ dim_municipio: {len(rows)} entidades carregadas "
          f"({len(rows) - 1} concelhos + Portugal)")
    return len(rows)


def get_id(cur, table: str, col: str, val) -> int | None:
    """Devolve a PK de *table* onde *col* = *val*, ou None se não existir."""
    pk = _PK_MAP[table]
    cur.execute(f"SELECT {pk} FROM {table} WHERE {col} = %s", (val,))
    row = cur.fetchone()
    return row[0] if row else None


def get_tempo_id(cur, ano) -> int | None:
    """Devolve tempo_id para *ano*; insere na dim_tempo se ainda não existir."""
    if ano is None or (isinstance(ano, float) and pd.isna(ano)):
        return None
    ano = int(ano)
    cur.execute("SELECT tempo_id FROM dim_tempo WHERE ano = %s", (ano,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(
        "INSERT INTO dim_tempo (ano) VALUES (%s) ON CONFLICT (ano) DO NOTHING RETURNING tempo_id",
        (ano,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def carregar_lookup(conn, cluster: str) -> tuple[dict, dict, dict]:
    """
    Lê as dimensões de uma só vez e devolve (mun_map, met_map, tmp_map).

    *cluster* é o valor de dim_metrica.cluster usado para filtrar métricas
    (ex.: 'Governança', 'Ambiente', ...).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT codigo_ine, municipio_id FROM dim_municipio")
        mun_map = {row[0]: row[1] for row in cur.fetchall()}

        cur.execute(
            "SELECT codigo, metrica_id FROM dim_metrica WHERE cluster = %s",
            (cluster,),
        )
        met_map = {row[0]: row[1] for row in cur.fetchall()}

        cur.execute("SELECT ano, tempo_id FROM dim_tempo")
        tmp_map = {row[0]: row[1] for row in cur.fetchall()}

    return mun_map, met_map, tmp_map


# ── Upsert fact_indicadores ────────────────────────────────────────────────────

_UPSERT_INDICADORES_SQL = """
INSERT INTO fact_indicadores
    (municipio_id, metrica_id, tempo_id, valor, valor_normalizado, flag_estimado)
VALUES %s
ON CONFLICT (municipio_id, metrica_id, tempo_id)
DO UPDATE SET
    valor             = EXCLUDED.valor,
    valor_normalizado = EXCLUDED.valor_normalizado,
    flag_estimado     = EXCLUDED.flag_estimado
"""


def load_indicadores_bulk(
    conn,
    df: pd.DataFrame,
    mun_map: dict,
    met_map: dict,
    tmp_map: dict,
    *,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Converte *df* em tuplos e faz upsert em fact_indicadores.

    Devolve (n_inseridos_ou_actualizados, n_ignorados).
    Imprime avisos para municípios/métricas sem ID na dimensão.
    """
    rows: list[tuple] = []
    ignorados = 0
    sem_mun: set[str] = set()
    sem_met: set[str] = set()

    for _, r in df.iterrows():
        cod     = str(r["codigo_ine"])
        metrica = str(r["metrica_codigo"])
        ano_raw = r.get("ano")
        valor   = r.get("valor")
        vnorm   = r.get("valor_normalizado")

        if valor is None or (isinstance(valor, float) and pd.isna(valor)):
            ignorados += 1
            continue

        mun_id = mun_map.get(cod)
        met_id = met_map.get(metrica)
        ano    = int(ano_raw) if ano_raw is not None and pd.notna(ano_raw) else None
        tmp_id = tmp_map.get(ano)

        if mun_id is None:
            sem_mun.add(cod)
            ignorados += 1
            continue
        if met_id is None:
            sem_met.add(metrica)
            ignorados += 1
            continue

        vnorm_val = float(vnorm) if (vnorm is not None and pd.notna(vnorm)) else None
        estimado  = bool(r.get("flag_estimado", False))
        rows.append((mun_id, met_id, tmp_id, float(valor), vnorm_val, estimado))

    if sem_mun:
        print(f"  ⚠ Municípios sem ID em dim_municipio: {sorted(sem_mun)}")
    if sem_met:
        print(f"  ⚠ Métricas sem ID em dim_metrica: {sorted(sem_met)}")
        print("     → Correr o SQL de schema antes de re-executar o load")

    if dry_run:
        print(f"  [DRY-RUN] {len(rows)} rows prontas · {ignorados} ignoradas")
        return len(rows), ignorados

    if rows:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur, _UPSERT_INDICADORES_SQL, rows, page_size=500
            )
        conn.commit()

    return len(rows), ignorados


# ── População residente — denominador partilhado por todos os clusters ────────

def carregar_populacao_referencia(
    ano: int | None = None, *, incluir_pt: bool = False
) -> dict[str, float]:
    path = STAGING_DIR / "soc_censos_2021.parquet"
    if not path.exists():
        return {}

    df = pd.read_parquet(path)
    if not incluir_pt:
        df = df[df["codigo_ine"] != PT_CODIGO]

    df_cim  = df[df["codigo_ine"] != PT_CODIGO]
    ano_ref = ano or ANO_REFERENCIA_POPULACAO

    sub = df[df["ano"] == ano_ref]
    if sub.empty and not df_cim.empty:
        ano_ref = int(df_cim["ano"].max())
        sub = df[df["ano"] == ano_ref]

    return sub.set_index("codigo_ine")["valor"].to_dict()


def filtrar_populacao_cim(df_pop: pd.DataFrame, ano: int | None = None) -> pd.DataFrame:
    df_cim  = df_pop[df_pop["codigo_ine"] != PT_CODIGO]
    ano_ref = ano or ANO_REFERENCIA_POPULACAO

    sub = df_cim[df_cim["ano"] == ano_ref]
    if sub.empty and not df_cim.empty:
        ano_ref = int(df_cim["ano"].max())
        sub = df_cim[df_cim["ano"] == ano_ref]

    return sub


def carregar_populacao_serie(*, incluir_pt: bool = False) -> dict[tuple[str, int], float]:
    path = STAGING_DIR / "soc_censos_2021.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if not incluir_pt:
        df = df[df["codigo_ine"] != PT_CODIGO]
    return {
        (row["codigo_ine"], int(row["ano"])): row["valor"]
        for _, row in df.iterrows()
    }


# ── Cálculo de scores via SQL ──────────────────────────────────────────────────

_UPSERT_SCORES_SQL = """
INSERT INTO fact_scores
    (municipio_id, tempo_id, cluster, subcluster, score, score_pct)
SELECT
    f.municipio_id,
    f.tempo_id,
    me.cluster,
    me.subcluster,
    ROUND(AVG(f.valor_normalizado)::numeric, 4),
    ROUND((AVG(f.valor_normalizado) * 100)::numeric, 2)
FROM fact_indicadores f
JOIN dim_metrica me ON f.metrica_id = me.metrica_id
WHERE me.cluster = %s
  AND f.valor_normalizado IS NOT NULL
GROUP BY f.municipio_id, f.tempo_id, me.cluster, me.subcluster
ON CONFLICT (municipio_id, tempo_id, cluster, subcluster)
DO UPDATE SET
    score     = EXCLUDED.score,
    score_pct = EXCLUDED.score_pct
"""


def calcular_scores_sql(conn, cluster: str) -> None:
    """Recalcula fact_scores para *cluster* directamente via SQL agregado."""
    with conn.cursor() as cur:
        cur.execute(_UPSERT_SCORES_SQL, (cluster,))
    conn.commit()
    print("  ✓ Scores por subcluster calculados")


# ── Logging de validação ───────────────────────────────────────────────────────

_AVISOS: list[str] = []
_ERROS:  list[str] = []


def resetar_log() -> None:
    """Limpa as listas de avisos/erros (chamar no início de cada main de validate)."""
    _AVISOS.clear()
    _ERROS.clear()


def aviso(msg: str) -> None:
    print(f"  ⚠  {msg}")
    _AVISOS.append(msg)


def erro(msg: str) -> None:
    print(f"  ✗  {msg}")
    _ERROS.append(msg)


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def get_avisos() -> list[str]:
    return list(_AVISOS)


def get_erros() -> list[str]:
    return list(_ERROS)


def imprimir_resumo_validacao(report_path: Path) -> None:
    """Imprime o bloco final de resumo de validação."""
    print(f"\n{'='*50}")
    if _ERROS:
        print(f"  ✗ {len(_ERROS)} erro(s) — corrigir antes do load")
    elif _AVISOS:
        print(f"  ⚠ {len(_AVISOS)} aviso(s) — pode prosseguir com cautela")
    else:
        print("  ✓ Sem problemas — pronto para load")
    print(f"  Relatório: {report_path}")


# ── Checks de validação reutilizáveis ──────────────────────────────────────────

def check_cobertura(df: pd.DataFrame, label: str) -> None:
    """Verifica que os 11 municípios da CIM estão representados no DataFrame."""
    ausentes = set(MUNICIPIOS.keys()) - set(df["codigo_ine"].astype(str).unique())
    if ausentes:
        nomes = [MUNICIPIOS[c] for c in sorted(ausentes)]
        aviso(f"{label}: municípios em falta → {nomes}")
    else:
        ok(f"{label}: todos os 11 municípios presentes")


def check_nulos(df: pd.DataFrame, label: str) -> dict:
    """Calcula % de nulos por métrica; regista aviso se > 50%."""
    result: dict[str, float] = {}
    for metrica, grupo in df.groupby("metrica_codigo"):
        pct = grupo["valor"].isna().mean() * 100
        result[metrica] = round(pct, 1)
        if pct > 50:
            aviso(f"{label} · {metrica}: {pct:.0f}% nulos")
        elif pct > 0:
            print(f"     ℹ  {metrica}: {pct:.0f}% nulos")
    return result


def check_outliers(df: pd.DataFrame, label: str) -> list[str]:
    """Detecta outliers por Z-score > 3 dentro de cada (métrica, ano)."""
    import numpy as np

    suspeitos: list[str] = []
    for (metrica, ano), grupo in df.groupby(["metrica_codigo", "ano"]):
        vals = grupo["valor"].dropna()
        if len(vals) < 4:
            continue
        std = vals.std()
        if std == 0:
            continue
        z = np.abs((vals - vals.mean()) / std)
        for idx in z[z > 3].index:
            row = df.loc[idx]
            msg = f"{label} · {metrica} · {row['nome']} · {ano}: {row['valor']:.2f} (Z>3)"
            aviso(msg)
            suspeitos.append(msg)
    return suspeitos


def check_scores_normalizados(df: pd.DataFrame, label: str) -> None:
    """Verifica que valor_normalizado está em [0, 1]."""
    df_num = df[df["valor_normalizado"].notna()]
    fora   = df_num[(df_num["valor_normalizado"] < 0) | (df_num["valor_normalizado"] > 1)]
    if not fora.empty:
        erro(f"{label}: {len(fora)} scores fora de [0,1]")
    else:
        ok(f"{label}: todos os scores normalizados em [0,1]")


def check_percentagens(
    df: pd.DataFrame,
    *,
    ilimitadas: set[str] | None = None,
) -> None:
    """
    Verifica que métricas *_pct estão em [0, 100].
    *ilimitadas* é o conjunto de métricas de variação/crescimento a excluir.
    """
    ilimitadas = ilimitadas or set()
    bounded = [
        c for c in df["metrica_codigo"].unique()
        if c.endswith("_pct") and c not in ilimitadas
    ]
    for m in bounded:
        sub  = df[df["metrica_codigo"] == m]
        fora = sub[(sub["valor"] < 0) | (sub["valor"] > 100)]
        if not fora.empty:
            erro(f"{m}: {len(fora)} valores fora de [0,100]%")
        else:
            ok(f"{m}: valores em [0,100]%")

    for m in ilimitadas:
        sub = df[df["metrica_codigo"] == m]
        if sub.empty:
            continue
        vals = sub["valor"].dropna()
        print(f"     ℹ  {m}: [{vals.min():.1f}%, {vals.max():.1f}%] (ilimitada por natureza)")


# ── Runner genérico (partilhado pelos run_*.py) ────────────────────────────────

PASSOS = ["extract", "transform", "validate", "load"]


def correr_pipeline(
    cluster_label: str,
    modulo_base: str,
    desde: str | None = None,
    ate: str | None = None,
) -> None:
    """
    Corre os passos extract → transform → validate → load para um cluster.

    *cluster_label*  — texto apresentado no cabeçalho (ex.: "Cluster 1 — Governança")
    *modulo_base*    — prefixo do módulo ETL (ex.: "governanca")
    *desde* / *ate*  — nomes de passos opcionais para execução parcial
    """
    import importlib
    import time

    idx_desde = PASSOS.index(desde) if desde in (PASSOS if desde else []) else 0
    idx_ate   = PASSOS.index(ate)   if ate   in (PASSOS if ate   else []) else len(PASSOS) - 1
    passos    = PASSOS[idx_desde : idx_ate + 1]

    print(f"\n{'='*55}")
    print(f"  CIM Lezíria do Tejo · {cluster_label}")
    print(f"  Pipeline: {' → '.join(p.upper() for p in passos)}")
    print(f"{'='*55}")

    t_total = time.time()

    # Garantir que dim_municipio está populada antes de qualquer load
    if "load" in passos:
        try:
            conn = conectar()
            seed_dim_municipio(conn)
            conn.close()
        except Exception as exc:
            print(f"\n  ✗ Erro ao popular dim_municipio: {exc}")
            sys.exit(1)

    for passo in passos:
        t0 = time.time()
        print(f"\n{'='*55}\n  PASSO: {passo.upper()}\n{'='*55}")
        try:
            mod = importlib.import_module(f"etl.{passo}_{modulo_base}")
            mod.main()
            print(f"\n  → {passo} concluído em {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"\n  ✗ Erro em {passo}: {exc}")
            import traceback
            traceback.print_exc()
            print(f"\n  Pipeline interrompido em '{passo}'.")
            sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  ✓ Pipeline completo em {time.time() - t_total:.1f}s")
    print(f"{'='*55}\n")


def parse_args() -> tuple[str | None, str | None]:
    """Faz parse de --desde e --ate a partir de sys.argv."""
    desde = ate = None
    args  = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--desde" and i + 1 < len(args):
            desde = args[i + 1]
        if arg == "--ate"   and i + 1 < len(args):
            ate   = args[i + 1]
    return desde, ate


# ── Helpers de extracção (partilhados pelos extract_*.py) ─────────────────────

import re
import unicodedata
import numpy as np

# Lookup nome → código INE, com aliases de acentuação/abreviatura
_NORM_TO_INE: dict[str, str] = {}
_ALIASES: dict[str, str] = {
    "salvaterra de magos": "1415",
    "santarém":            "1416",
    "azambuja":            "1103",
    "golega":              "1412",
}


def _build_lookup() -> None:
    global _NORM_TO_INE
    _NORM_TO_INE = {normalizar_texto(nome): cod for cod, nome in MUNICIPIOS.items()}
    _NORM_TO_INE.update({normalizar_texto(k): v for k, v in _ALIASES.items()})


def normalizar_texto(s: str) -> str:
    """Remove acentos, normaliza espaços e converte para minúsculas."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip().lower()


# Construir o lookup após definir normalizar_texto e MUNICIPIOS
_build_lookup()


def encontrar_codigo(valor_raw: str) -> str | None:
    """
    Resolve um código INE a partir de vários formatos de entrada:
      - '1D31403: Almeirim'  (INE formato 1D3)
      - '1403: Almeirim'     (INE Censos)
      - '1403'               (código puro)
      - 'Almeirim'           (nome puro, via NORM_TO_INE)
    """
    raw = str(valor_raw).strip()

    # Formato INE: "1D31403: ..."
    m = re.search(r"1D3(\d{4})", raw)
    if m:
        cod = m.group(1)
        return cod if cod in MUNICIPIOS else None

    # Formato Censos: "1403: ..." ou "1403:..."
    m = re.match(r"^(\d{4})\s*:", raw)
    if m:
        cod = m.group(1)
        return cod if cod in MUNICIPIOS else None

    # Código numérico puro
    if raw.isdigit() and raw in MUNICIPIOS:
        return raw

    # Nome (com fallback parcial)
    norm = normalizar_texto(raw)
    if norm in _NORM_TO_INE:
        return _NORM_TO_INE[norm]
    for chave, cod in _NORM_TO_INE.items():
        if chave and chave in norm:
            return cod

    return None


def safe_float(v) -> float | None:
    """Converte para float; devolve None para vazios, traços e valores não numéricos."""
    if v is None:
        return None
    s = str(v).strip().replace(",", ".").replace("\xa0", "").replace(" ", "")
    if s in ("-", "…", "nan", ""):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ── Helpers de transformação (partilhados pelos transform_*.py) ───────────────

def row_base(
    codigo_ine: str,
    nome: str,
    ano: int | None,
    metrica_codigo: str,
    valor,
) -> dict:
    """Dicionário base com o schema mínimo de um registo de facto."""
    return {
        "codigo_ine":     codigo_ine,
        "nome":           nome,
        "ano":            ano,
        "metrica_codigo": metrica_codigo,
        "valor":          valor,
    }


def normalizar_minmax(series: pd.Series, inverter: bool = False) -> pd.Series:
    """Min-max sobre os valores disponíveis. inverter=True → menor é melhor."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    norm = (series - mn) / (mx - mn)
    return (1 - norm) if inverter else norm


def normalizar_scores(
    df: pd.DataFrame,
    *,
    metricas_inverter:        set[str] | None = None,
    metricas_sem_normalizacao: set[str] | None = None,
) -> pd.DataFrame:
    """
    Calcula valor_normalizado (min-max por métrica×ano) para todo o DataFrame.

    Parâmetros
    ----------
    metricas_inverter          — métricas onde o menor valor é melhor (score invertido)
    metricas_sem_normalizacao  — métricas textuais ou categóricas a saltar

    Portugal ('PT') é excluído do cálculo do min-max: é um valor de
    referência nacional, não um concorrente na escala dos 11 municípios da
    CIM. Se entrasse no min-max, distorcia o score de todos os concelhos
    sempre que o valor nacional fosse o mais alto/baixo do grupo. A linha de
    Portugal fica sempre com valor_normalizado = None (não tem "score" CIM).
    """
    inv  = metricas_inverter         or set()
    skip = metricas_sem_normalizacao or set()

    df["valor_normalizado"] = np.nan

    is_pt = df["codigo_ine"] == PT_CODIGO if "codigo_ine" in df.columns else pd.Series(False, index=df.index)

    for (metrica, ano), grupo_completo in df.groupby(["metrica_codigo", "ano"]):
        if metrica in skip:
            continue
        grupo = grupo_completo[~is_pt.loc[grupo_completo.index]]
        vals  = grupo["valor"].dropna()
        if len(vals) < 2:
            df.loc[grupo.index, "valor_normalizado"] = 0.5
            continue
        df.loc[grupo.index, "valor_normalizado"] = normalizar_minmax(
            df.loc[grupo.index, "valor"], inverter=(metrica in inv)
        ).round(4)

    return df


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que o DataFrame tem exactamente as colunas do schema canónico,
    pela ordem correcta. Colunas em falta são adicionadas com NaN.
    """
    SCHEMA = ["codigo_ine", "nome", "ano", "metrica_codigo",
              "valor", "valor_normalizado", "valor_texto", "categoria"]
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = None
    return df[SCHEMA]
