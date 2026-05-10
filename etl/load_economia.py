import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
import yaml
from pathlib import Path

CONFIG_PATH = Path("config/sources.yaml")
STAGING_DIR = Path("data/staging")

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DB = cfg["database"]   # host, port, dbname, user, password

# ── Ligação ────────────────────────────────────────────────────────────────────

def conectar():
    return psycopg2.connect(
        host=DB["host"], port=DB["port"],
        dbname=DB["dbname"], user=DB["user"], password=DB["password"],
    )

# ── Lookup helpers (lidos uma vez, mantidos em memória) ────────────────────────

def carregar_lookup(conn) -> tuple[dict, dict, dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT codigo_ine, municipio_id FROM dim_municipio;")
        mun_map = {row[0]: row[1] for row in cur.fetchall()}

        cur.execute("SELECT codigo, metrica_id FROM dim_metrica WHERE cluster = 'Economia';")
        metrica_map = {row[0]: row[1] for row in cur.fetchall()}

        cur.execute("SELECT ano, tempo_id FROM dim_tempo;")
        tempo_map = {row[0]: row[1] for row in cur.fetchall()}

    return mun_map, metrica_map, tempo_map


# ── Upsert fact_indicadores ────────────────────────────────────────────────────

UPSERT_SQL = """
INSERT INTO fact_indicadores
    (municipio_id, metrica_id, tempo_id, valor, valor_normalizado, flag_estimado)
VALUES %s
ON CONFLICT (municipio_id, metrica_id, tempo_id)
DO UPDATE SET
    valor              = EXCLUDED.valor,
    valor_normalizado  = EXCLUDED.valor_normalizado,
    flag_estimado      = EXCLUDED.flag_estimado;
"""

def upsert_indicadores(conn, df: pd.DataFrame,
                        mun_map: dict, metrica_map: dict, tempo_map: dict,
                        dry_run: bool = False) -> tuple[int, int]:
    """
    Converte df → tuples e faz upsert em fact_indicadores.
    Retorna (n_inseridos_ou_atualizados, n_ignorados).
    """
    rows = []
    ignorados = 0

    for _, r in df.iterrows():
        mun_id     = mun_map.get(str(r["codigo_ine"]))
        metrica_id = metrica_map.get(str(r["metrica_codigo"]))
        ano        = int(r["ano"]) if pd.notna(r["ano"]) else None
        tempo_id   = tempo_map.get(ano) if ano is not None else None

        if mun_id is None or metrica_id is None:
            ignorados += 1
            continue

        valor     = float(r["valor"])           if pd.notna(r.get("valor"))            else None
        val_norm  = float(r["valor_normalizado"]) if pd.notna(r.get("valor_normalizado")) else None
        estimado  = bool(r.get("flag_estimado", False))

        rows.append((mun_id, metrica_id, tempo_id, valor, val_norm, estimado))

    if dry_run:
        print(f"   [DRY-RUN] {len(rows)} rows prontas · {ignorados} ignoradas")
        return len(rows), ignorados

    if not rows:
        return 0, ignorados

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, UPSERT_SQL, rows, page_size=500)
    conn.commit()

    return len(rows), ignorados


# ── Upsert fact_scores (scores agregados por subcluster) ──────────────────────

UPSERT_SCORES_SQL = """
INSERT INTO fact_scores
    (municipio_id, tempo_id, cluster, subcluster, score, score_pct)
VALUES %s
ON CONFLICT (municipio_id, tempo_id, cluster, subcluster)
DO UPDATE SET
    score     = EXCLUDED.score,
    score_pct = EXCLUDED.score_pct;
"""

def calcular_e_inserir_scores(conn, df: pd.DataFrame,
                               mun_map: dict, tempo_map: dict,
                               dry_run: bool = False) -> int:
    SUBCLUSTERS = {
        "Emprego e Estrutura": [
            "eco_taxa_emprego_pct",
            "eco_estrutura_agricultura_pct", "eco_estrutura_industria_pct",
            "eco_estrutura_servicos_pct",
            "eco_taxa_conta_propria_pct", "eco_taxa_grandes_empregadores_pct",
        ],
        "Rendimento e Capacidade Fiscal": [
            "eco_rendimento_bruto_per_capita_e", "eco_irs_per_capita_e",
            "eco_taxa_esforco_irs_pct", "eco_ipc_base100",
            "eco_tcma_rendimento_bruto_pct", "eco_proporcao_pc_pct",
        ],
        "Empresarialidade": [
            "eco_taxa_natalidade_emp_pct", "eco_taxa_mortalidade_emp_pct",
            "eco_taxa_sobrevivencia_1ano_pct", "eco_vn_per_capita_e",
            # eco_estrutura_vn_* removidas: o ficheiro VN não tem desagregação CAE municipal
        ],
    }

    rows = []
    for subcluster, metricas in SUBCLUSTERS.items():
        df_sub = df[df["metrica_codigo"].isin(metricas)].copy()
        if df_sub.empty:
            continue
        idx_last = df_sub.groupby(["codigo_ine", "metrica_codigo"])["ano"].idxmax()
        df_last  = df_sub.loc[idx_last]
        # Agregar por município: média dos valor_normalizado de cada métrica
        scores_mun = (
            df_last[df_last["valor_normalizado"].notna()]
            .groupby("codigo_ine")
            .agg(valor_normalizado=("valor_normalizado", "mean"),
                 ano=("ano", "max"))
            .reset_index()
        )
        for _, r in scores_mun.iterrows():
            mun_id   = mun_map.get(str(r["codigo_ine"]))
            val_n    = float(r["valor_normalizado"]) if pd.notna(r["valor_normalizado"]) else None
            ano      = int(r["ano"]) if pd.notna(r["ano"]) else None
            tempo_id = tempo_map.get(ano)
            if mun_id is None or val_n is None:
                continue
            rows.append((mun_id, tempo_id, "Economia", subcluster,
                         round(val_n, 4), round(val_n * 100, 2)))

    if dry_run:
        print(f"   [DRY-RUN] {len(rows)} scores prontos")
        return len(rows)

    if not rows:
        return 0

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, UPSERT_SCORES_SQL, rows, page_size=200)
    conn.commit()
    return len(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False):
    print(f"\n=== LOAD · Cluster 5 — Economia {'[DRY-RUN]' if dry_run else ''} ===\n")

    df = pd.read_parquet(STAGING_DIR / "eco_transformed.parquet")
    print(f"   Registos a carregar: {len(df)}")
    print(f"   Métricas únicas:     {df['metrica_codigo'].nunique()}")
    print(f"   Municípios:          {df['codigo_ine'].nunique()}")

    conn = conectar()
    print("   Ligação PostgreSQL: OK")

    mun_map, metrica_map, tempo_map = carregar_lookup(conn)
    print(f"   Lookup carregado: {len(mun_map)} mun · {len(metrica_map)} métricas eco · {len(tempo_map)} anos")

    # Verificar métricas sem ID (schema não foi corrido?)
    metricas_sem_id = set(df["metrica_codigo"].unique()) - set(metrica_map.keys())
    if metricas_sem_id:
        print(f"\n  ⚠  {len(metricas_sem_id)} métricas sem ID em dim_metrica:")
        for m in sorted(metricas_sem_id):
            print(f"     · {m}")
        print("     → Correr schema_economia.sql antes deste script!\n")

    print("\n[ fact_indicadores ]")
    n_ok, n_ign = upsert_indicadores(conn, df, mun_map, metrica_map, tempo_map, dry_run)
    print(f"   Upsert: {n_ok} rows · {n_ign} ignoradas")

    print("\n[ fact_scores ]")
    n_scores = calcular_e_inserir_scores(conn, df, mun_map, tempo_map, dry_run)
    print(f"   Scores: {n_scores} rows")

    conn.close()

    print(f"\n✓ Load concluído {'(dry-run — nada foi escrito)' if dry_run else ''}")
    print(f"   fact_indicadores: {n_ok} upserts")
    print(f"   fact_scores:      {n_scores} upserts")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Simula o load sem escrever na base de dados")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
