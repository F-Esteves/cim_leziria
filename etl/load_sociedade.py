from pathlib import Path
import pandas as pd
import psycopg2
import psycopg2.extras
import yaml

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


def carregar_lookup(conn) -> tuple[dict, dict, dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT codigo_ine, municipio_id FROM dim_municipio")
        mun_map = {row[0]: row[1] for row in cur.fetchall()}

        cur.execute("SELECT codigo, metrica_id FROM dim_metrica WHERE cluster = 'Sociedade'")
        met_map = {row[0]: row[1] for row in cur.fetchall()}

        cur.execute("SELECT ano, tempo_id FROM dim_tempo")
        tmp_map = {row[0]: row[1] for row in cur.fetchall()}

    return mun_map, met_map, tmp_map


def upsert_fact_indicadores(conn, df: pd.DataFrame,
                             mun_map: dict, met_map: dict, tmp_map: dict):
    rows = []
    ignorados = 0
    sem_mun, sem_met = set(), set()

    for _, row in df.iterrows():
        cod     = str(row["codigo_ine"])
        metrica = str(row["metrica_codigo"])
        ano     = int(row["ano"]) if pd.notna(row.get("ano")) else None
        valor   = row.get("valor")
        vnorm   = row.get("valor_normalizado")

        if valor is None or (isinstance(valor, float) and pd.isna(valor)):
            ignorados += 1
            continue

        mun_id   = mun_map.get(cod)
        met_id   = met_map.get(metrica)
        tempo_id = tmp_map.get(ano)

        if mun_id is None:
            sem_mun.add(cod); ignorados += 1; continue
        if met_id is None:
            sem_met.add(metrica); ignorados += 1; continue

        vnorm_val = float(vnorm) if (vnorm is not None and pd.notna(vnorm)) else None
        rows.append((mun_id, met_id, tempo_id, float(valor), vnorm_val, False))

    if rows:
        sql = """
        INSERT INTO fact_indicadores
            (municipio_id, metrica_id, tempo_id, valor, valor_normalizado, flag_estimado)
        VALUES %s
        ON CONFLICT (municipio_id, metrica_id, tempo_id)
        DO UPDATE SET
            valor             = EXCLUDED.valor,
            valor_normalizado = EXCLUDED.valor_normalizado,
            flag_estimado     = EXCLUDED.flag_estimado
        """
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows, page_size=500)
        conn.commit()

    if sem_mun: print(f"  ⚠ Municípios sem ID: {sorted(sem_mun)}")
    if sem_met:
        print(f"  ⚠ Métricas sem ID em dim_metrica: {sorted(sem_met)}")
        print(f"     → Adicionar ao schema antes de re-correr o load")

    print(f"  ✓ {len(rows)} inseridos/actualizados · {ignorados} ignorados")
    return len(rows), ignorados


def calcular_scores(conn):
    with conn.cursor() as cur:
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
            WHERE me.cluster = 'Sociedade'
              AND f.valor_normalizado IS NOT NULL
            GROUP BY f.municipio_id, f.tempo_id, me.cluster, me.subcluster
            ON CONFLICT (municipio_id, tempo_id, cluster, subcluster)
            DO UPDATE SET
                score     = EXCLUDED.score,
                score_pct = EXCLUDED.score_pct
        """)
    conn.commit()
    print("  ✓ Scores por subcluster calculados")


def main():
    print("\n=== LOAD · Cluster 6 — Sociedade ===\n")

    path = STAGING_DIR / "soc_transformed.parquet"
    if not path.exists():
        print("  ✗ soc_transformed.parquet não encontrado — corre transform primeiro")
        return

    df = pd.read_parquet(path)
    print(f"  {len(df)} registos a carregar\n")

    try:
        conn = conectar()
        print(f"  ✓ Ligado ao PostgreSQL · {DB['dbname']}@{DB['host']}\n")
    except Exception as e:
        print(f"  ✗ Erro de ligação: {e}")
        return

    mun_map, met_map, tmp_map = carregar_lookup(conn)
    print(f"  Lookup: {len(mun_map)} mun · {len(met_map)} métricas · {len(tmp_map)} anos\n")

    metricas_sem_id = set(df["metrica_codigo"].unique()) - set(met_map.keys())
    if metricas_sem_id:
        print(f"  ⚠ {len(metricas_sem_id)} métricas sem ID em dim_metrica:")
        for m in sorted(metricas_sem_id):
            print(f"     · {m}")
        print("     → Correr schema_sociedade.sql antes deste script!\n")

    print("[ fact_indicadores ]")
    upsert_fact_indicadores(conn, df, mun_map, met_map, tmp_map)

    print("\n[ fact_scores ]")
    calcular_scores(conn)

    conn.close()
    print(f"\n✓ Load concluído — BD: {DB['dbname']}")


if __name__ == "__main__":
    main()
