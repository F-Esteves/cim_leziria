import pandas as pd
import psycopg2
import psycopg2.extras
import yaml
from pathlib import Path

CONFIG_PATH = Path("config/sources.yaml")
STAGING_DIR = Path("data/staging")

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DB = cfg["database"]


# ── Ligação ────────────────────────────────────────────────────

def conectar():
    return psycopg2.connect(
        host=DB["host"], port=DB["port"],
        dbname=DB["dbname"], user=DB["user"], password=DB["password"],
    )

# ── Helpers ────────────────────────────────────────────────────

PK_MAP = {
    "dim_municipio": "municipio_id",
    "dim_metrica":   "metrica_id",
    "dim_tempo":     "tempo_id",
}

def get_id(cur, table: str, col: str, val) -> int | None:
    pk = PK_MAP[table]
    cur.execute(f"SELECT {pk} FROM {table} WHERE {col} = %s", (val,))
    row = cur.fetchone()
    return row[0] if row else None


def get_tempo_id(cur, ano) -> int | None:
    """Devolve tempo_id para o ano; insere se ainda não existir."""
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


# ── fact_indicadores ───────────────────────────────────────────

def load_indicadores(conn, df: pd.DataFrame) -> tuple[int, int]:
    cur = conn.cursor()
    cache_mun: dict = {}
    cache_met: dict = {}
    cache_tmp: dict = {}
    rows_pg:   list = []
    ignorados  = 0
    sem_mun:   set  = set()
    sem_met:   set  = set()

    for _, row in df.iterrows():
        cod     = row["codigo_ine"]
        metrica = row["metrica_codigo"]
        ano     = row.get("ano")
        valor   = row.get("valor")
        vnorm   = row.get("valor_normalizado")

        if valor is None or (isinstance(valor, float) and pd.isna(valor)):
            ignorados += 1
            continue

        if cod not in cache_mun:
            cache_mun[cod] = get_id(cur, "dim_municipio", "codigo_ine", cod)
        if metrica not in cache_met:
            cache_met[metrica] = get_id(cur, "dim_metrica", "codigo", metrica)
        if ano not in cache_tmp:
            cache_tmp[ano] = get_tempo_id(cur, ano)

        mun_id = cache_mun[cod]
        met_id = cache_met[metrica]
        tmp_id = cache_tmp.get(ano)

        if mun_id is None:
            sem_mun.add(cod); ignorados += 1; continue
        if met_id is None:
            sem_met.add(metrica); ignorados += 1; continue

        vnorm_val = None if (vnorm is None or (isinstance(vnorm, float) and pd.isna(vnorm))) \
                    else float(vnorm)
        rows_pg.append((mun_id, met_id, tmp_id, float(valor), vnorm_val, False))

    if rows_pg:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO fact_indicadores
                (municipio_id, metrica_id, tempo_id, valor, valor_normalizado, flag_estimado)
            VALUES %s
            ON CONFLICT (municipio_id, metrica_id, tempo_id)
            DO UPDATE SET
                valor             = EXCLUDED.valor,
                valor_normalizado = EXCLUDED.valor_normalizado
            """,
            rows_pg,
            page_size=500,
        )
        conn.commit()

    if sem_mun:
        print(f"  ⚠ Municípios sem ID em dim_municipio: {sorted(sem_mun)}")
    if sem_met:
        print(f"  ⚠ Métricas sem ID em dim_metrica: {sorted(sem_met)}")
        print(f"     → Adicionar ao schema antes de re-correr o load")

    cur.close()
    return len(rows_pg), ignorados


# ── fact_scores ────────────────────────────────────────────────

def calcular_scores(conn) -> None:
 
    cur = conn.cursor()
    cur.execute("""
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
        WHERE me.cluster = 'Modos de Vida'
          AND f.valor_normalizado IS NOT NULL
        GROUP BY f.municipio_id, f.tempo_id, me.cluster, me.subcluster
        ON CONFLICT (municipio_id, tempo_id, cluster, subcluster)
        DO UPDATE SET
            score     = EXCLUDED.score,
            score_pct = EXCLUDED.score_pct
    """)
    conn.commit()
    cur.close()
    print("  ✓ Scores por subcluster calculados")


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    print("\n=== LOAD · Cluster 4 — Modos de Vida ===\n")

    path = STAGING_DIR / "mdv_transformed.parquet"
    if not path.exists():
        print("  ✗ mdv_transformed.parquet não encontrado — corre transform primeiro")
        return

    df = pd.read_parquet(path)
    print(f"  {len(df)} registos a carregar\n")

    try:
        conn = conectar()
        print(f"  ✓ Ligado ao PostgreSQL · {DB['dbname']}@{DB['host']}\n")
    except Exception as e:
        print(f"  ✗ Erro de ligação: {e}")
        return

    print("[ fact_indicadores ]")
    ins, ign = load_indicadores(conn, df)
    print(f"  ✓ {ins} inseridos · {ign} ignorados")

    print("\n[ fact_scores ]")
    calcular_scores(conn)

    conn.close()
    print(f"\n✓ Load concluído — BD: {DB['dbname']}")


if __name__ == "__main__":
    main()