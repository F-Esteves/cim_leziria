import pandas as pd
import psycopg2.extras
from etl.utils import (
    STAGING_DIR, DB,
    conectar, get_tempo_id, carregar_lookup,
    load_indicadores_bulk, calcular_scores_sql,
)


def load_partido_vencedor(conn) -> None:
    path = STAGING_DIR / "gov_partido_vencedor.parquet"
    if not path.exists():
        print("  ⚠ gov_partido_vencedor.parquet não encontrado — a saltar")
        return

    df = pd.read_parquet(path)
    rows = []
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            cur.execute(
                "SELECT municipio_id FROM dim_municipio WHERE codigo_ine = %s",
                (row["codigo_ine"],),
            )
            mun = cur.fetchone()
            if not mun:
                continue
            tmp_id = get_tempo_id(cur, row["ano"])
            rows.append((
                mun[0], tmp_id,
                "autarquias",
                str(row["partido"]),
                str(row["categoria"]),
                int(row["votantes"]) if pd.notna(row.get("votantes")) else None,
            ))

        if rows:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO fact_partido_vencedor
                       (municipio_id, tempo_id, tipo_eleicao, partido, categoria, votos)
                   VALUES %s
                   ON CONFLICT (municipio_id, tempo_id, tipo_eleicao)
                   DO UPDATE SET partido=EXCLUDED.partido,
                                 categoria=EXCLUDED.categoria,
                                 votos=EXCLUDED.votos""",
                rows,
            )
    conn.commit()
    print(f"  ✓ {len(rows)} registos de partido vencedor inseridos")


def main() -> None:
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

    mun_map, met_map, tmp_map = carregar_lookup(conn, "Governança")

    print("[ fact_indicadores ]")
    ins, ign = load_indicadores_bulk(conn, df, mun_map, met_map, tmp_map)
    print(f"  ✓ {ins} inseridos · {ign} ignorados")

    print("\n[ fact_partido_vencedor ]")
    load_partido_vencedor(conn)

    print("\n[ fact_scores ]")
    calcular_scores_sql(conn, "Governança")

    conn.close()
    print(f"\n✓ Load concluído — BD: {DB['dbname']}")
    print("  Views disponíveis: vw_indicadores · vw_scores · vw_ultimo_valor · vw_ranking")


if __name__ == "__main__":
    main()
