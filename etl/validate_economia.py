import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime

STAGING_DIR = Path("data/staging")
N_MUN = 11

AVISOS: list[str] = []
ERROS:  list[str] = []


def aviso(msg: str) -> None:
    print(f"  ⚠  {msg}")
    AVISOS.append(msg)


def erro(msg: str) -> None:
    print(f"  ✗  {msg}")
    ERROS.append(msg)


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def check(cond: bool, msg: str, fatal: bool = True) -> None:
    if not cond:
        erro(msg) if fatal else aviso(msg)


METRICAS_ESPERADAS = {
    # 5.1 Emprego e Estrutura (6)
    "eco_taxa_emprego_pct",
    "eco_estrutura_agricultura_pct",
    "eco_estrutura_industria_pct",
    "eco_estrutura_servicos_pct",
    "eco_taxa_conta_propria_pct",
    "eco_taxa_grandes_empregadores_pct",
    # 5.2 Rendimento e Capacidade Fiscal (6)
    "eco_rendimento_bruto_per_capita_e",
    "eco_irs_per_capita_e",
    "eco_taxa_esforco_irs_pct",
    "eco_ipc_base100",
    "eco_tcma_rendimento_bruto_pct",
    "eco_proporcao_pc_pct",
    # 5.3 Empresarialidade (4)
    "eco_taxa_natalidade_emp_pct",
    "eco_taxa_mortalidade_emp_pct",
    "eco_taxa_sobrevivencia_1ano_pct",
    "eco_vn_per_capita_e",
}

def validar_staging():
    ficheiros = [
        ("eco_emprego_conta_outrem.parquet",       "Emprego CO"),
        ("eco_emprego_censos_bruto.parquet",        "Censos bruto"),
        ("eco_rendimento_bruto.parquet",            "Rendimento bruto"),
        ("eco_irs_liquidado.parquet",               "IRS"),
        ("eco_poder_compra_per_capita.parquet",     "IPC per capita"),
        ("eco_proporcao_poder_compra.parquet",      "Proporção PC"),
        ("eco_empresas_nascidas.parquet",           "Nascidas"),
        ("eco_empresas_mortas.parquet",             "Mortas"),
        ("eco_empresas_sobreviventes.parquet",      "Sobreviventes"),
        ("eco_volume_negocios_total.parquet",       "VN total"),
        ("eco_volume_negocios_sectores.parquet",    "VN sectores"),
    ]
    print("\n[ STAGING — ficheiros extraídos ]")
    for fname, label in ficheiros:
        p = STAGING_DIR / fname
        if not p.exists():
            ERROS.append(f"Ficheiro em falta: {fname}")
            continue
        df = pd.read_parquet(p)
        n_mun = df["codigo_ine"].nunique() if "codigo_ine" in df.columns else 0
        print(f"   {label:<30} {len(df):>6} reg · {n_mun}/{N_MUN} mun")
        check(n_mun > 0, f"{fname} — nenhum município reconhecido")
        check(n_mun == N_MUN, f"{fname} — {n_mun}/{N_MUN} municípios", fatal=False)

def validar_transformed():
    p = STAGING_DIR / "eco_transformed.parquet"
    if not p.exists():
        ERROS.append("eco_transformed.parquet não encontrado — correr transform primeiro")
        return
    df = pd.read_parquet(p)

    print("\n[ TRANSFORMED ]")
    print(f"   Total registos:  {len(df)}")
    print(f"   Municípios:      {df['codigo_ine'].nunique()}")
    print(f"   Métricas:        {df['metrica_codigo'].nunique()}")

    presentes = set(df["metrica_codigo"].unique())
    ausentes  = METRICAS_ESPERADAS - presentes
    extras    = presentes - METRICAS_ESPERADAS
    for m in sorted(ausentes):
        AVISOS.append(f"Métrica esperada ausente: {m}")
    for m in sorted(extras):
        AVISOS.append(f"Métrica extra inesperada: {m}")

   
    for metrica in ["eco_taxa_natalidade_emp_pct", "eco_taxa_mortalidade_emp_pct"]:
        vals = df[df["metrica_codigo"] == metrica]["valor"].dropna()
        if not vals.empty:
            fora = vals[(vals < 0) | (vals > 100)]
            check(fora.empty, f"{metrica}: {len(fora)} valores fora de [0,100]%", fatal=False)
            if vals.max() > 50:
                check(False, f"{metrica}: valor máximo {vals.max():.1f}% — verificar denominador (flag_estimado?)", fatal=False)

    sobr = df[df["metrica_codigo"] == "eco_taxa_sobrevivencia_1ano_pct"]["valor"].dropna()
    if not sobr.empty:
        fora_sobr = sobr[(sobr < 0) | (sobr > 150)]
        check(fora_sobr.empty, f"eco_taxa_sobrevivencia_1ano_pct: {len(fora_sobr)} valores fora de [0,150]%", fatal=False)

   
    taxa = df[df["metrica_codigo"] == "eco_taxa_esforco_irs_pct"]["valor"].dropna()
    if not taxa.empty:
        check(taxa.max() <= 100, f"Taxa esforço IRS > 100% (max={taxa.max():.1f}%)", fatal=False)

    rpc = df[df["metrica_codigo"] == "eco_rendimento_bruto_per_capita_e"]["valor"].dropna()
    if not rpc.empty:
        check(rpc.min() >= 5_000,  f"Rendimento p.c. mínimo suspeito: {rpc.min():.0f}€ — verificar conversão k€→€", fatal=False)
        check(rpc.max() <= 60_000, f"Rendimento p.c. máximo suspeito: {rpc.max():.0f}€ — verificar conversão k€→€", fatal=False)

   
    ipc_v = df[df["metrica_codigo"] == "eco_irs_per_capita_e"]["valor"].dropna()
    if not ipc_v.empty:
        check(ipc_v.min() >= 200,  f"IRS p.c. mínimo suspeito: {ipc_v.min():.0f}€ — verificar conversão k€→€", fatal=False)
        check(ipc_v.max() <= 15_000, f"IRS p.c. máximo suspeito: {ipc_v.max():.0f}€ — verificar conversão k€→€", fatal=False)

  
    anos_vn   = df[df["metrica_codigo"] == "eco_vn_per_capita_e"]["ano"].dropna().astype(int)
    anos_rend = df[df["metrica_codigo"] == "eco_rendimento_bruto_per_capita_e"]["ano"].dropna().astype(int)
    if not anos_vn.empty and not anos_rend.empty:
        diff_anos = anos_rend.max() - anos_vn.max()
        if diff_anos > 3:
            check(False,
                  f"Desfasamento temporal: VN até {anos_vn.max()}, Rendimento até {anos_rend.max()} "
                  f"({diff_anos} anos de diferença) — comparações inter-métricas devem ser interpretadas com cautela",
                  fatal=False)

    
    set_cols = ["eco_estrutura_agricultura_pct","eco_estrutura_industria_pct","eco_estrutura_servicos_pct"]
    df_set = df[df["metrica_codigo"].isin(set_cols)]
    if not df_set.empty:
        pivot = df_set.pivot_table(index="codigo_ine", columns="metrica_codigo", values="valor", aggfunc="first")
        pivot = pivot.reindex(columns=set_cols)
        soma_set = pivot.sum(axis=1).dropna()
        fora = ((soma_set < 90) | (soma_set > 110)).sum()
        check(fora == 0, f"Estrutura sectorial soma fora de [90,110]% em {fora} municípios", fatal=False)

    
    n_ipc = df[df["metrica_codigo"] == "eco_ipc_base100"]["codigo_ine"].nunique()
    check(n_ipc == N_MUN, f"IPC per capita apenas em {n_ipc}/{N_MUN} municípios", fatal=False)

    
    print("\n   Detalhe por métrica:")
    for m, g in df.groupby("metrica_codigo"):
        anos = sorted(g["ano"].dropna().astype(int).unique())
        n_m  = g["codigo_ine"].nunique()
        suf  = "..." if len(anos) > 3 else ""
        print(f"     {m:<45} {len(g):>5} reg · {n_m}/{N_MUN} mun · anos {anos[:3]}{suf}")


def main():
    print("\n=== VALIDATE · Cluster 5 — Economia ===")
    validar_staging()
    validar_transformed()

   
    p = STAGING_DIR / "eco_transformed.parquet"
    stats = {}
    if p.exists():
        df = pd.read_parquet(p)
        stats = (
            df.groupby("metrica_codigo")["valor"]
            .agg(["count", "min", "max", "mean"])
            .round(2)
            .reset_index()
            .to_dict(orient="records")
        )
        n_rows = len(df)
        n_met  = int(df["metrica_codigo"].nunique())
        n_mun  = int(df["codigo_ine"].nunique())
    else:
        n_rows = n_met = n_mun = 0

    report = {
        "timestamp":    datetime.now().isoformat(),
        "cluster":      "Economia",
        "total_rows":   n_rows,
        "n_metricas":   n_met,
        "n_municipios": n_mun,
        "avisos":       AVISOS,
        "erros":        ERROS,
        "stats":        stats,
    }
    report_path = STAGING_DIR / "eco_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "─" * 60)
    if ERROS:
        print(f"  ✗ {len(ERROS)} erro(s) — corrigir antes do load")
        for e in ERROS:
            print(f"   ✗ {e}")
        print(f"  Relatório: {report_path}")
        sys.exit(1)
    elif AVISOS:
        print(f"  ⚠ {len(AVISOS)} aviso(s) — pode prosseguir com cautela")
        for a in AVISOS:
            print(f"   · {a}")
    else:
        print("  ✓ Sem problemas — pronto para load")
    print(f"  Relatório: {report_path}")


if __name__ == "__main__":
    main()
