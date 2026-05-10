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


def conectar():
    return psycopg2.connect(
        host=DB["host"], port=DB["port"],
        dbname=DB["dbname"], user=DB["user"], password=DB["password"],
    )


PK_MAP = {
    "dim_municipio": "municipio_id",
    "dim_metrica":   "metrica_id",
    "dim_tempo":     "tempo_id",
}

def get_id(cur, table, col, val):
    pk = PK_MAP[table]
    cur.execute(f"SELECT {pk} FROM {table} WHERE {col} = %s", (val,))
    row = cur.fetchone()
    return row[0] if row else None

def get_tempo_id(cur, ano):
    if ano is None or (isinstance(ano, float) and pd.isna(ano)):
        return None
    ano = int(ano)
    cur.execute("SELECT tempo_id FROM dim_tempo WHERE ano = %s", (ano,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO dim_tempo (ano) VALUES (%s) ON CONFLICT (ano) DO NOTHING RETURNING tempo_id", (ano,))
    row = cur.fetchone()
    return row[0] if row else None


def load_indicadores(conn, df: pd.DataFrame):
    cur = conn.cursor()
    cache_mun, cache_met, cache_tmp = {}, {}, {}
    rows_pg = []
    ignorados = 0

    for _, row in df.iterrows():
        cod, metrica, ano = row["codigo_ine"], row["metrica_codigo"], row.get("ano")
        valor, vnorm = row.get("valor"), row.get("valor_normalizado")

        if pd.isna(valor) if valor is not None else True:
            ignorados += 1; continue

        if cod not in cache_mun:
            cache_mun[cod] = get_id(cur, "dim_municipio", "codigo_ine", cod)
        if metrica not in cache_met:
            cache_met[metrica] = get_id(cur, "dim_metrica", "codigo", metrica)
        if ano not in cache_tmp:
            cache_tmp[ano] = get_tempo_id(cur, ano)

        mun_id = cache_mun[cod]
        met_id = cache_met[metrica]
        tmp_id = cache_tmp.get(ano)

        if mun_id is None or met_id is None:
            ignorados += 1; continue

        vnorm_val = None if (vnorm is None or pd.isna(vnorm)) else float(vnorm)
        rows_pg.append((mun_id, met_id, tmp_id, float(valor), vnorm_val, False))

    if rows_pg:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO fact_indicadores
                   (municipio_id, metrica_id, tempo_id, valor, valor_normalizado, flag_estimado)
               VALUES %s
               ON CONFLICT (municipio_id, metrica_id, tempo_id)
               DO UPDATE SET valor=EXCLUDED.valor, valor_normalizado=EXCLUDED.valor_normalizado""",
            rows_pg, page_size=500,
        )
        conn.commit()

    cur.close()
    return len(rows_pg), ignorados


def load_partido_vencedor(conn):
    path = STAGING_DIR / "gov_partido_vencedor.parquet"
    if not path.exists():
        print("  ⚠ gov_partido_vencedor.parquet não encontrado — a saltar")
        return

    df = pd.read_parquet(path)
    cur = conn.cursor()
    rows_pg = []

    for _, row in df.iterrows():
        cur.execute("SELECT municipio_id FROM dim_municipio WHERE codigo_ine = %s", (row["codigo_ine"],))
        mun = cur.fetchone()
        if not mun:
            continue
        tmp_id = get_tempo_id(cur, row["ano"])
        rows_pg.append((
            mun[0], tmp_id,
            "autarquias",                      # tipo_eleicao fixo
            str(row["partido"]),
            str(row["categoria"]),
            int(row["votantes"]) if pd.notna(row.get("votantes")) else None,
        ))

    if rows_pg:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO fact_partido_vencedor
               (municipio_id, tempo_id, tipo_eleicao, partido, categoria, votos)
               VALUES %s
               ON CONFLICT (municipio_id, tempo_id, tipo_eleicao)
               DO UPDATE SET partido=EXCLUDED.partido,
                             categoria=EXCLUDED.categoria,
                             votos=EXCLUDED.votos""",
            rows_pg,
        )
    conn.commit()
    print(f"  ✓ {len(rows_pg)} registos de partido vencedor inseridos")
    cur.close()

def calcular_scores(conn):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO fact_scores (municipio_id, tempo_id, cluster, subcluster, score, score_pct)
        SELECT f.municipio_id, f.tempo_id, me.cluster, me.subcluster,
               ROUND(AVG(f.valor_normalizado)::numeric, 4),
               ROUND((AVG(f.valor_normalizado)*100)::numeric, 2)
        FROM fact_indicadores f
        JOIN dim_metrica me ON f.metrica_id = me.metrica_id
        WHERE me.cluster = 'Governança' AND f.valor_normalizado IS NOT NULL
        GROUP BY f.municipio_id, f.tempo_id, me.cluster, me.subcluster
        ON CONFLICT (municipio_id, tempo_id, cluster, subcluster)
        DO UPDATE SET score=EXCLUDED.score, score_pct=EXCLUDED.score_pct
    """)
    conn.commit()
    cur.close()
    print("  ✓ Scores por subcluster calculados")


def main():
    print("\n=== LOAD · Cluster 1 — Governança ===\n")

    path = STAGING_DIR / "gov_transformed.parquet"
    if not path.exists():
        print("  ✗ gov_transformed.parquet não encontrado"); return

    df = pd.read_parquet(path)
    print(f"  {len(df)} registos a carregar\n")

    try:
        conn = conectar()
        print(f"  ✓ Ligado ao PostgreSQL · {DB['dbname']}@{DB['host']}\n")
    except Exception as e:
        print(f"  ✗ Erro de ligação: {e}"); return

    print("[ fact_indicadores ]")
    ins, ign = load_indicadores(conn, df)
    print(f"  ✓ {ins} inseridos · {ign} ignorados")

    print("\n[ fact_partido_vencedor ]")
    load_partido_vencedor(conn)

    print("\n[ fact_scores ]")
    calcular_scores(conn)

    conn.close()
    print(f"\n✓ Load concluído — BD: {DB['dbname']}")
    print("  Views disponíveis: vw_indicadores · vw_scores · vw_ultimo_valor · vw_ranking")


if __name__ == "__main__":
    main()
