import pandas as pd
from etl.utils import (
    STAGING_DIR, DB,
    conectar, carregar_lookup,
    load_indicadores_bulk, calcular_scores_sql,
)


def main() -> None:
    print("\n=== LOAD · Economia ===\n")

    path = STAGING_DIR / "eco_transformed.parquet"
    if not path.exists():
        print("  ✗ eco_transformed.parquet não encontrado — corre transform primeiro")
        return

    df = pd.read_parquet(path)
    print(f"  {len(df)} registos a carregar\n")

    try:
        conn = conectar()
        print(f"  ✓ Ligado ao PostgreSQL · {DB['dbname']}@{DB['host']}\n")
    except Exception as e:
        print(f"  ✗ Erro de ligação: {e}")
        return

    mun_map, met_map, tmp_map = carregar_lookup(conn, "Economia")
    metricas_sem_id = set(df["metrica_codigo"].unique()) - set(met_map.keys())
    if metricas_sem_id:
        print(f"  ⚠ {len(metricas_sem_id)} métricas sem ID em dim_metrica:")
        for m in sorted(metricas_sem_id):
            print(f"     · {m}")
        print("     → Correr schema.sql antes deste script!\n")

    print("[ fact_indicadores ]")
    ins, ign = load_indicadores_bulk(conn, df, mun_map, met_map, tmp_map)
    print(f"  ✓ {ins} inseridos · {ign} ignorados")

    print("\n[ fact_scores ]")
    calcular_scores_sql(conn, "Economia")

    conn.close()
    print(f"\n✓ Load concluído — BD: {DB['dbname']}")


if __name__ == "__main__":
    main()
