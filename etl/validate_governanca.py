import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

STAGING_DIR = Path("data/staging")

MUNICIPIOS = {
    "1403": "Almeirim",
    "1404": "Alpiarça",
    "1103": "Azambuja",
    "1405": "Benavente",
    "1406": "Cartaxo",
    "1407": "Chamusca",
    "1409": "Coruche",
    "1412": "Golegã",
    "1414": "Rio Maior",
    "1415": "Salvaterra de Magos",
    "1416": "Santarém",
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


# ── Checks ─────────────────────────────────────────────────────

def check_cobertura(df: pd.DataFrame, label: str) -> None:
    ausentes = set(MUNICIPIOS.keys()) - set(df["codigo_ine"].astype(str).unique())
    if ausentes:
        nomes = [MUNICIPIOS[c] for c in sorted(ausentes)]
        aviso(f"{label}: municípios sem dados → {nomes}")
    else:
        ok(f"{label}: todos os 11 municípios presentes")


def check_nulos(df: pd.DataFrame, label: str) -> dict:
    result = {}
    for metrica, grupo in df.groupby("metrica_codigo"):
        pct = grupo["valor"].isna().mean() * 100
        result[metrica] = round(pct, 1)
        if pct > 50:
            aviso(f"{label} · {metrica}: {pct:.0f}% nulos")
        elif pct > 0:
            print(f"     ℹ  {metrica}: {pct:.0f}% nulos")
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
            row = df.loc[idx]
            msg = f"{label} · {metrica} · {row['nome']} · {ano}: {row['valor']:.2f}"
            aviso(msg)
            suspeitos.append(msg)
    return suspeitos


def check_scores(df: pd.DataFrame, label: str) -> None:
   
    df_num = df[df["valor_normalizado"].notna()]
    fora   = df_num[(df_num["valor_normalizado"] < 0) | (df_num["valor_normalizado"] > 1)]
    if not fora.empty:
        erro(f"{label}: {len(fora)} scores fora de [0,1]")
    else:
        ok(f"{label}: todos os scores normalizados em [0,1]")


def check_tv_unidade(df: pd.DataFrame) -> None:
    tem_abs    = "gov_tv_assinantes_abs" in df["metrica_codigo"].values
    tem_100hab = "gov_tv_100hab"         in df["metrica_codigo"].values
    if tem_100hab and tem_abs:
        ok("TV: gov_tv_assinantes_abs (legacy) + gov_tv_100hab (normalizado) presentes")
    elif tem_100hab:
        ok("TV: gov_tv_100hab presente")
    elif tem_abs:
        aviso("TV: apenas gov_tv_assinantes_abs — gov_tv_100hab não calculado "
              "(soc_censos_2021.parquet ausente durante o transform?)")
    else:
        aviso("TV: nenhuma métrica de TV encontrada")


def check_partido_vencedor() -> None:
    path = STAGING_DIR / "gov_partido_vencedor.parquet"
    if not path.exists():
        aviso("gov_partido_vencedor.parquet não encontrado")
        return

    df = pd.read_parquet(path)
    ausentes = set(MUNICIPIOS.keys()) - set(df["codigo_ine"].astype(str).unique())
    if ausentes:
        nomes = [MUNICIPIOS[c] for c in sorted(ausentes)]
        aviso(f"Partido vencedor: municípios em falta → {nomes}")
    else:
        ok("Partido vencedor: todos os 11 municípios presentes")

    print("\n  Resultados autárquicas 2025:")
    for _, r in df.sort_values("nome").iterrows():
        print(f"    {r['nome']:25s} → {str(r['partido']):40s} [{r['categoria']}]")


def check_digital_anos(df: pd.DataFrame) -> None:
    metricas_digitais = [
        "gov_banda_larga_100hab",
        "gov_telefone_100hab",
        "gov_tv_assinantes_abs",
        "gov_tv_100hab",
    ]
    for metrica in metricas_digitais:
        sub = df[df["metrica_codigo"] == metrica]
        if sub.empty:
            
            if metrica == "gov_tv_100hab" and \
               "gov_tv_assinantes_abs" in df["metrica_codigo"].values:
                aviso(f"{metrica}: não calculado — soc_censos_2021.parquet estava ausente")
            continue
        anos_por_mun = sub.groupby("codigo_ine")["ano"].nunique()
        incompletos = anos_por_mun[anos_por_mun < 4].index.tolist()
        if incompletos:
            nomes = [MUNICIPIOS.get(c, c) for c in incompletos]
            aviso(f"{metrica}: menos de 4 anos em {nomes}")
        else:
            ok(f"{metrica}: 4 anos por município")


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    print("\n=== VALIDATE · Cluster 1 — Governança ===\n")

    path = STAGING_DIR / "gov_transformed.parquet"
    if not path.exists():
        erro("gov_transformed.parquet não encontrado")
        return

    df = pd.read_parquet(path)
    print(f"  Carregados {len(df)} registos\n")

    print("[ Cobertura municipal ]")
    check_cobertura(df, "Governança")

    print("\n[ Nulos por métrica ]")
    nulos = check_nulos(df, "Governança")

    print("\n[ Outliers ]")
    outliers = check_outliers(df, "Governança")

    print("\n[ Scores normalizados ]")
    check_scores(df, "Governança")

    print("\n[ Intervalos lógicos — métricas % ]")
    for metrica in [m for m in df["metrica_codigo"].unique() if m.endswith("_pct")]:
        vals = df[df["metrica_codigo"] == metrica]["valor"].dropna()
        fora = vals[(vals < 0) | (vals > 100)]
        if not fora.empty:
            erro(f"{metrica}: {len(fora)} valores fora de [0,100]")
        else:
            ok(f"{metrica}: intervalo válido")

    print("\n[ Unidade TV ]")
    check_tv_unidade(df)

    print("\n[ Série temporal digital ]")
    check_digital_anos(df)

    print("\n[ Partido vencedor ]")
    check_partido_vencedor()

    print("\n[ Estatísticas por métrica (métricas numéricas) ]")
    df_num = df[df["valor"].notna() & ~df["metrica_codigo"].isin(
        {"gov_partido_vencedor_cm", "gov_tv_assinantes_abs"}
    )]
    stats = df_num.groupby("metrica_codigo")["valor"].agg(
        ["count", "min", "max", "mean", "std"]
    ).round(3)
    print(stats.to_string())

    report = {
        "timestamp":    datetime.now().isoformat(),
        "cluster":      "Governança",
        "total_rows":   len(df),
        "n_metricas":   int(df["metrica_codigo"].nunique()),
        "n_municipios": int(df["codigo_ine"].nunique()),
        "avisos":       AVISOS,
        "erros":        ERROS,
        "nulos_pct":    nulos,
        "outliers":     outliers,
    }

    report_path = STAGING_DIR / "gov_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    if ERROS:
        print(f"  ✗ {len(ERROS)} erro(s) — corrigir antes do load")
    elif AVISOS:
        print(f"  ⚠ {len(AVISOS)} aviso(s) — pode prosseguir com cautela")
    else:
        print("  ✓ Sem problemas — pronto para load")
    print(f"  Relatório: data/staging/gov_quality_report.json")


if __name__ == "__main__":
    main()