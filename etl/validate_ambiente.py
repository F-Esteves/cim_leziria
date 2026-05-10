import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

STAGING_DIR = Path("data/staging")

MUNICIPIOS = {
    "1403": "Almeirim",        "1404": "Alpiarça",
    "1103": "Azambuja",        "1405": "Benavente",
    "1406": "Cartaxo",         "1407": "Chamusca",
    "1409": "Coruche",         "1412": "Golegã",
    "1414": "Rio Maior",       "1415": "Salvaterra de Magos",
    "1416": "Santarém",
}

METRICAS_PCT_ILIMITADAS = {
    "amb_var_consumo_anual_pct",   
    "amb_tcma_consumo_pct",        
    "amb_evolucao_valorizacao_pp", 
}

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


def check_cobertura(df: pd.DataFrame, label: str) -> None:
    presentes = set(df["codigo_ine"].astype(str).unique())
    ausentes  = set(MUNICIPIOS.keys()) - presentes
    if ausentes:
        nomes = [MUNICIPIOS[c] for c in sorted(ausentes)]
        aviso(f"{label}: municípios em falta → {nomes}")
    else:
        ok(f"{label}: todos os 11 municípios presentes")


def check_nulos(df: pd.DataFrame, label: str) -> dict:
    result = {}
    for metrica, grupo in df.groupby("metrica_codigo"):
        pct = grupo["valor"].isna().mean() * 100
        result[metrica] = round(pct, 1)
        if pct > 50:
            aviso(f"{label} · {metrica}: {pct:.0f}% nulos")
    return result


def check_outliers(df: pd.DataFrame, label: str) -> list:
    suspeitos = []
    for (metrica, ano), grupo in df.groupby(["metrica_codigo", "ano"]):
        vals = grupo["valor"].dropna()
        if len(vals) < 4:
            continue
        std = vals.std()
        if std == 0:
            continue
        z = np.abs((vals - vals.mean()) / std)
        for idx in z[z > 3].index:
            row = grupo.loc[idx]
            msg = f"{label} · {metrica} · {row['nome']} · {ano}: {row['valor']:.2f} (Z>3)"
            aviso(msg)
            suspeitos.append(msg)
    return suspeitos


def check_scores(df: pd.DataFrame, label: str) -> None:
    df_num = df[df["valor_normalizado"].notna()]
    fora   = df_num[(df_num["valor_normalizado"] < 0) | (df_num["valor_normalizado"] > 1)]
    if not fora.empty:
        erro(f"{label}: {len(fora)} scores fora de [0,1]")
    else:
        ok(f"{label}: scores em [0,1]")


def check_valores_positivos(df: pd.DataFrame) -> None:
    """Consumos e totais de resíduos devem ser positivos."""
    metricas_positivas = [
        "amb_consumo_total_kwh", "amb_consumo_bt_kwh", "amb_consumo_at_kwh",
        "amb_n_cpes_total", "amb_residuos_total_ton",
    ]
    for m in metricas_positivas:
        sub = df[df["metrica_codigo"] == m]
        if sub.empty:
            aviso(f"{m}: sem dados")
            continue
        negativos = sub[sub["valor"] < 0]
        if not negativos.empty:
            erro(f"{m}: {len(negativos)} valores negativos")
        else:
            ok(f"{m}: valores positivos ✓")


def check_percentagens(df: pd.DataFrame) -> None:
  
    pct_bounded = [
        c for c in df["metrica_codigo"].unique()
        if c.endswith("_pct") and c not in METRICAS_PCT_ILIMITADAS
    ]
    for m in pct_bounded:
        sub  = df[df["metrica_codigo"] == m]
        fora = sub[(sub["valor"] < 0) | (sub["valor"] > 100)]
        if not fora.empty:
            erro(f"{m}: {len(fora)} valores fora de [0,100]%")
        else:
            ok(f"{m}: valores em [0,100]%")

    
    for m in METRICAS_PCT_ILIMITADAS:
        sub = df[df["metrica_codigo"] == m]
        if sub.empty:
            continue
        vals = sub["valor"].dropna()
        print(f"     ℹ  {m}: [{vals.min():.1f}%, {vals.max():.1f}%] (ilimitada por natureza)")


def check_anos_completos(df: pd.DataFrame) -> None:
    anos_incompletos = {2020, 2025}
    metricas_anuais  = ["amb_consumo_total_kwh", "amb_consumo_bt_kwh", "amb_consumo_at_kwh"]
    for m in metricas_anuais:
        sub = df[df["metrica_codigo"] == m]
        presentes = set(sub["ano"].astype(int).unique())
        incompletos_presentes = presentes & anos_incompletos
        if incompletos_presentes:
            aviso(f"{m}: anos incompletos presentes {sorted(incompletos_presentes)} — TCMA e variações anuais podem estar distorcidos")
        else:
            ok(f"{m}: sem anos incompletos")


def main() -> None:
    print("\n=== VALIDATE · Cluster 2 — Ambiente ===\n")

    path = STAGING_DIR / "amb_transformed.parquet"
    if not path.exists():
        erro("amb_transformed.parquet não encontrado — corre transform primeiro")
        return

    df = pd.read_parquet(path)
    print(f"  Carregados {len(df)} registos\n")

    print("[ Cobertura municipal ]")
    check_cobertura(df, "Ambiente")

    print("\n[ Nulos por métrica ]")
    nulos = check_nulos(df, "Ambiente")

    print("\n[ Outliers (Z-score > 3) ]")
    outliers = check_outliers(df, "Ambiente")

    print("\n[ Scores normalizados ]")
    check_scores(df, "Ambiente")

    print("\n[ Valores positivos ]")
    check_valores_positivos(df)

    print("\n[ Percentagens bounded ]")
    check_percentagens(df)

    print("\n[ Anos incompletos em métricas anuais ]")
    check_anos_completos(df)

    print("\n[ Estatísticas por métrica ]")
    stats = (
        df.groupby("metrica_codigo")["valor"]
        .agg(["count", "min", "max", "mean"])
        .round(2)
    )
    print(stats.to_string())

    report = {
        "timestamp":    datetime.now().isoformat(),
        "cluster":      "Ambiente",
        "total_rows":   len(df),
        "n_metricas":   int(df["metrica_codigo"].nunique()),
        "n_municipios": int(df["codigo_ine"].nunique()),
        "avisos":       AVISOS,
        "erros":        ERROS,
        "nulos_pct":    nulos,
        "outliers":     outliers,
        "stats":        stats.reset_index().to_dict(orient="records"),
    }

    report_path = STAGING_DIR / "amb_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    if ERROS:
        print(f"  ✗ {len(ERROS)} erro(s) — corrigir antes do load")
    elif AVISOS:
        print(f"  ⚠ {len(AVISOS)} aviso(s) — pode prosseguir")
    else:
        print("  ✓ Sem problemas — pronto para load")
    print(f"  Relatório: data/staging/amb_quality_report.json")


if __name__ == "__main__":
    main()