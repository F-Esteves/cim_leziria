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

# Métricas de variação/crescimento — ilimitadas por natureza
METRICAS_PCT_ILIMITADAS = {
    "mob_evolucao_veiculos_pp",
    "mob_ve_crescimento_pct",
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
            row = df.loc[idx]
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


def check_percentagens(df: pd.DataFrame) -> None:
    """Percentagens bounded [0,100]: só as taxas/proporções, não as variações."""
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


def check_valores_positivos(df: pd.DataFrame) -> None:
    metricas_positivas = [
        "mob_registo_total_1000hab", "mob_registo_ligeiros_1000hab",
        "mob_ve_total",
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


def check_serie_veiculos(df: pd.DataFrame) -> None:
    """Veículos devem ter 4 anos (2021-2024) por município."""
    for metrica in ["mob_registo_total_1000hab", "mob_registo_ligeiros_1000hab"]:
        sub = df[df["metrica_codigo"] == metrica]
        if sub.empty:
            aviso(f"{metrica}: sem dados")
            continue
        anos_por_mun = sub.groupby("codigo_ine")["ano"].nunique()
        incompletos  = anos_por_mun[anos_por_mun < 4].index.tolist()
        if incompletos:
            nomes = [MUNICIPIOS.get(c, c) for c in incompletos]
            aviso(f"{metrica}: menos de 4 anos em {nomes}")
        else:
            ok(f"{metrica}: 4 anos por município")


def check_pontos_ve_consistencia(df: pd.DataFrame) -> None:
    pub  = df[df["metrica_codigo"] == "mob_ve_publicos_pct"]["valor"]
    priv = df[df["metrica_codigo"] == "mob_ve_privados_pct"]["valor"]
    if not pub.empty and not priv.empty:
        soma = pub.values + priv.values
        fora = soma[(soma < 99.0) | (soma > 101.0)]
        if len(fora) > 0:
            aviso(f"mob_ve_publicos_pct + privados_pct: {len(fora)} municípios com soma ≠ 100%")
        else:
            ok("mob_ve: público + privado = 100% ✓")

    semi  = df[df["metrica_codigo"] == "mob_ve_semirrapidos_pct"]["valor"]
    rap   = df[df["metrica_codigo"] == "mob_ve_rapidos_pct"]["valor"]
    if not semi.empty and not rap.empty:
        ok("mob_ve: métricas de tipo de ponto presentes ✓")


def main() -> None:
    print("\n=== VALIDATE · Cluster 3 — Mobilidade ===\n")

    path = STAGING_DIR / "mob_transformed.parquet"
    if not path.exists():
        erro("mob_transformed.parquet não encontrado — corre transform primeiro")
        return

    df = pd.read_parquet(path)
    print(f"  Carregados {len(df)} registos\n")

    print("[ Cobertura municipal ]")
    check_cobertura(df, "Mobilidade")

    print("\n[ Nulos por métrica ]")
    nulos = check_nulos(df, "Mobilidade")

    print("\n[ Outliers (Z-score > 3) ]")
    outliers = check_outliers(df, "Mobilidade")

    print("\n[ Scores normalizados ]")
    check_scores(df, "Mobilidade")

    print("\n[ Valores positivos ]")
    check_valores_positivos(df)

    print("\n[ Percentagens bounded ]")
    check_percentagens(df)

    print("\n[ Série temporal veículos ]")
    check_serie_veiculos(df)

    print("\n[ Consistência pontos VE ]")
    check_pontos_ve_consistencia(df)

    print("\n[ Estatísticas por métrica ]")
    stats = (
        df.groupby("metrica_codigo")["valor"]
        .agg(["count", "min", "max", "mean"])
        .round(3)
    )
    print(stats.to_string())

    report = {
        "timestamp":    datetime.now().isoformat(),
        "cluster":      "Mobilidade",
        "total_rows":   len(df),
        "n_metricas":   int(df["metrica_codigo"].nunique()),
        "n_municipios": int(df["codigo_ine"].nunique()),
        "avisos":       AVISOS,
        "erros":        ERROS,
        "nulos_pct":    nulos,
        "outliers":     outliers,
        "stats":        stats.reset_index().to_dict(orient="records"),
    }

    report_path = STAGING_DIR / "mob_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    if ERROS:
        print(f"  ✗ {len(ERROS)} erro(s) — corrigir antes do load")
    elif AVISOS:
        print(f"  ⚠ {len(AVISOS)} aviso(s) — pode prosseguir")
    else:
        print("  ✓ Sem problemas — pronto para load")
    print(f"  Relatório: data/staging/mob_quality_report.json")


if __name__ == "__main__":
    main()
