import json
import sys
from datetime import datetime

import pandas as pd

from etl.utils import (
    STAGING_DIR,
    aviso, erro, resetar_log,
    get_avisos, get_erros,
    imprimir_resumo_validacao,
)

N_MUN = 11

_METRICAS_ESPERADAS = {
    # 5.1 Emprego
    "eco_taxa_emprego_pct",
    "eco_estrutura_agricultura_pct", "eco_estrutura_industria_pct",
    "eco_estrutura_servicos_pct",
    "eco_taxa_conta_propria_pct", "eco_taxa_grandes_empregadores_pct",
    # 5.2 Rendimento
    "eco_rendimento_bruto_per_capita_e", "eco_irs_per_capita_e",
    "eco_taxa_esforco_irs_pct", "eco_ipc_base100",
    "eco_tcma_rendimento_bruto_pct", "eco_proporcao_pc_pct",
    # 5.3 Empresarialidade
    "eco_empresas_nascidas_n", "eco_empresas_mortas_n",
    "eco_saldo_empresarial_n", "eco_vn_per_capita_e",
}

_FICHEIROS_STAGING = [
    ("eco_emprego_conta_outrem.parquet",    "Emprego CO"),
    ("eco_emprego_censos_bruto.parquet",    "Censos bruto"),
    ("eco_rendimento_bruto.parquet",        "Rendimento bruto"),
    ("eco_irs_liquidado.parquet",           "IRS"),
    ("eco_poder_compra_per_capita.parquet", "IPC per capita"),
    ("eco_proporcao_poder_compra.parquet",  "Proporção PC"),
    ("eco_empresas_nascidas.parquet",       "Nascidas"),
    ("eco_empresas_mortas.parquet",         "Mortas"),
    ("eco_empresas_sobreviventes.parquet",  "Sobreviventes"),
    ("eco_volume_negocios_total.parquet",   "VN total"),
    ("eco_volume_negocios_sectores.parquet","VN sectores"),
]


def _check(cond: bool, msg: str, fatal: bool = True) -> None:
    if not cond:
        erro(msg) if fatal else aviso(msg)


def validar_staging() -> None:
    print("\n[ STAGING — ficheiros extraídos ]")
    for fname, label in _FICHEIROS_STAGING:
        p = STAGING_DIR / fname
        if not p.exists():
            erro(f"Ficheiro em falta: {fname}"); continue
        df    = pd.read_parquet(p)
        n_mun = df["codigo_ine"].nunique() if "codigo_ine" in df.columns else 0
        print(f"   {label:<30} {len(df):>6} reg · {n_mun}/{N_MUN} mun")
        _check(n_mun > 0, f"{fname} — nenhum município reconhecido")
        _check(n_mun == N_MUN, f"{fname} — {n_mun}/{N_MUN} municípios", fatal=False)


def validar_transformed() -> None:
    p = STAGING_DIR / "eco_transformed.parquet"
    if not p.exists():
        erro("eco_transformed.parquet não encontrado — correr transform primeiro"); return

    df = pd.read_parquet(p)
    print("\n[ TRANSFORMED ]")
    print(f"   Total registos:  {len(df)}")
    print(f"   Municípios:      {df['codigo_ine'].nunique()}")
    print(f"   Métricas:        {df['metrica_codigo'].nunique()}")

    presentes = set(df["metrica_codigo"].unique())
    for m in sorted(_METRICAS_ESPERADAS - presentes):
        aviso(f"Métrica esperada ausente: {m}")
    for m in sorted(presentes - _METRICAS_ESPERADAS):
        aviso(f"Métrica extra inesperada: {m}")

    for metrica in ["eco_taxa_natalidade_emp_pct", "eco_taxa_mortalidade_emp_pct"]:
        vals = df[df["metrica_codigo"] == metrica]["valor"].dropna()
        if not vals.empty:
            fora = vals[(vals < 0) | (vals > 100)]
            _check(fora.empty, f"{metrica}: {len(fora)} valores fora de [0,100]%", fatal=False)
            if vals.max() > 50:
                _check(False, f"{metrica}: max {vals.max():.1f}% — verificar denominador", fatal=False)

    sobr = df[df["metrica_codigo"] == "eco_taxa_sobrevivencia_1ano_pct"]["valor"].dropna()
    if not sobr.empty:
        _check(
            ((sobr < 0) | (sobr > 150)).empty,
            f"eco_taxa_sobrevivencia_1ano_pct: valores fora de [0,150]%", fatal=False,
        )

    taxa = df[df["metrica_codigo"] == "eco_taxa_esforco_irs_pct"]["valor"].dropna()
    if not taxa.empty:
        _check(taxa.max() <= 100, f"Taxa esforço IRS > 100% (max={taxa.max():.1f}%)", fatal=False)

    rpc = df[df["metrica_codigo"] == "eco_rendimento_bruto_per_capita_e"]["valor"].dropna()
    if not rpc.empty:
        _check(rpc.min() >= 5_000,  f"Rendimento p.c. mín suspeito: {rpc.min():.0f}€", fatal=False)
        _check(rpc.max() <= 60_000, f"Rendimento p.c. máx suspeito: {rpc.max():.0f}€", fatal=False)

    irs = df[df["metrica_codigo"] == "eco_irs_per_capita_e"]["valor"].dropna()
    if not irs.empty:
        _check(irs.min() >= 200,   f"IRS p.c. mín suspeito: {irs.min():.0f}€", fatal=False)
        _check(irs.max() <= 15_000, f"IRS p.c. máx suspeito: {irs.max():.0f}€", fatal=False)

    anos_vn   = df[df["metrica_codigo"] == "eco_vn_per_capita_e"]["ano"].dropna().astype(int)
    anos_rend = df[df["metrica_codigo"] == "eco_rendimento_bruto_per_capita_e"]["ano"].dropna().astype(int)
    if not anos_vn.empty and not anos_rend.empty:
        diff = anos_rend.max() - anos_vn.max()
        if diff > 3:
            _check(False,
                   f"Desfasamento temporal: VN até {anos_vn.max()}, Rendimento até {anos_rend.max()} "
                   f"({diff} anos) — comparações inter-métricas a interpretar com cautela",
                   fatal=False)

    set_cols = ["eco_estrutura_agricultura_pct", "eco_estrutura_industria_pct", "eco_estrutura_servicos_pct"]
    df_set = df[df["metrica_codigo"].isin(set_cols)]
    if not df_set.empty:
        pivot = df_set.pivot_table(index="codigo_ine", columns="metrica_codigo", values="valor", aggfunc="first")
        soma  = pivot.reindex(columns=set_cols).sum(axis=1).dropna()
        fora  = ((soma < 90) | (soma > 110)).sum()
        _check(fora == 0, f"Estrutura sectorial soma fora de [90,110]% em {fora} municípios", fatal=False)

    n_ipc = df[df["metrica_codigo"] == "eco_ipc_base100"]["codigo_ine"].nunique()
    _check(n_ipc == N_MUN, f"IPC per capita apenas em {n_ipc}/{N_MUN} municípios", fatal=False)

    print("\n   Detalhe por métrica:")
    for m, g in df.groupby("metrica_codigo"):
        anos = sorted(g["ano"].dropna().astype(int).unique())
        n_m  = g["codigo_ine"].nunique()
        suf  = "..." if len(anos) > 3 else ""
        print(f"     {m:<45} {len(g):>5} reg · {n_m}/{N_MUN} mun · anos {anos[:3]}{suf}")


def main() -> None:
    resetar_log()
    print("\n=== VALIDATE · Cluster 5 — Economia ===")

    validar_staging()
    validar_transformed()

    p = STAGING_DIR / "eco_transformed.parquet"
    stats = {}
    n_rows = n_met = n_mun = 0
    if p.exists():
        df     = pd.read_parquet(p)
        n_rows = len(df)
        n_met  = int(df["metrica_codigo"].nunique())
        n_mun  = int(df["codigo_ine"].nunique())
        stats  = (
            df.groupby("metrica_codigo")["valor"]
            .agg(["count", "min", "max", "mean"])
            .round(2)
            .reset_index()
            .to_dict(orient="records")
        )

    report_path = STAGING_DIR / "eco_quality_report.json"
    report = {
        "timestamp":    datetime.now().isoformat(),
        "cluster":      "Economia",
        "total_rows":   n_rows,
        "n_metricas":   n_met,
        "n_municipios": n_mun,
        "avisos":       get_avisos(),
        "erros":        get_erros(),
        "stats":        stats,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "─" * 60)
    if get_erros():
        print(f"  ✗ {len(get_erros())} erro(s) — corrigir antes do load")
        for e in get_erros():
            print(f"   ✗ {e}")
        print(f"  Relatório: {report_path}")
        sys.exit(1)
    else:
        imprimir_resumo_validacao(report_path)


if __name__ == "__main__":
    main()
