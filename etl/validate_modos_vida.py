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

METRICAS_ULS = {
    "mdv_utentes_csp", "mdv_pct_utentes_mdf",
    "mdv_consultas_presenciais", "mdv_consultas_total",
}


COBERTURA_ESPERADA: dict[str, int] = {
    "mdv_acidentes_vitimas_1000hab": 10,  
}


METRICAS_PCT_ILIMITADAS = {
    "mdv_evolucao_criminalidade_pp",
    "mdv_evolucao_dormidas_pp",
}


METRICAS_SEM_NORMALIZACAO = {
    "mdv_utentes_csp", "mdv_consultas_presenciais", "mdv_consultas_total",
    "mdv_alojamentos_total", "mdv_alojamentos_familares",
    "mdv_alojamentos_uso_sazonal", "mdv_alojamentos_vagos",
    "mdv_ensino_superior_n",
    "mdv_consultas_por_hab",  
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


def check_cobertura(df: pd.DataFrame) -> None:
    for metrica in sorted(df["metrica_codigo"].unique()):
        n_esp = COBERTURA_ESPERADA.get(metrica, 11)
        sub   = df[df["metrica_codigo"] == metrica]
        n_mun = sub["codigo_ine"].nunique()
        ausentes = set(MUNICIPIOS.keys()) - set(sub["codigo_ine"].astype(str).unique())
        if n_mun < n_esp:
            aviso(f"{metrica}: {n_mun}/{n_esp} municípios "
                  f"(ausentes: {[MUNICIPIOS[c] for c in sorted(ausentes)]})")
        elif n_mun == 11:
            ok(f"{metrica}: 11/11 municípios ✓")
        else:
            ok(f"{metrica}: {n_mun}/{n_esp} municípios ✓ (cobertura esperada)")


def check_nulos(df: pd.DataFrame) -> dict:
    result = {}
    for metrica, grupo in df.groupby("metrica_codigo"):
        pct = grupo["valor"].isna().mean() * 100
        result[metrica] = round(pct, 1)
        if pct > 50:
            aviso(f"{metrica}: {pct:.0f}% nulos")
    return result


def check_outliers(df: pd.DataFrame) -> list:
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
            msg = f"{metrica} · {row['nome']} · {ano}: {row['valor']:.2f} (Z>3)"
            aviso(msg)
            suspeitos.append(msg)
    return suspeitos


def check_scores(df: pd.DataFrame) -> None:
    df_num = df[df["valor_normalizado"].notna()]
    fora   = df_num[(df_num["valor_normalizado"] < 0) | (df_num["valor_normalizado"] > 1)]
    if not fora.empty:
        erro(f"{len(fora)} scores fora de [0,1]")
    else:
        ok("Todos os scores normalizados em [0,1]")


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
        print(f"     ℹ  {m}: [{vals.min():.2f}, {vals.max():.2f}] (ilimitada por natureza)")


def check_valores_positivos(df: pd.DataFrame) -> None:
    metricas_pos = [
        "mdv_hab_medico", "mdv_dormidas_100hab",
        "mdv_acidentes_vitimas_1000hab", "mdv_criminalidade_total",
    ]
    for m in metricas_pos:
        sub = df[df["metrica_codigo"] == m]
        if sub.empty:
            aviso(f"{m}: sem dados")
            continue
        neg = sub[sub["valor"] < 0]
        if not neg.empty:
            erro(f"{m}: {len(neg)} valores negativos")
        else:
            ok(f"{m}: valores positivos ✓")


def check_anos_csp(df: pd.DataFrame) -> None:
    """CSP têm dados de 2015-2025 mas só interessa 2021-2025 para o dashboard."""
    for m in METRICAS_ULS:
        sub  = df[df["metrica_codigo"] == m]
        anos = sorted(sub["ano"].unique())
        if not anos:
            aviso(f"{m}: sem dados")
        elif max(anos) < 2024:
            aviso(f"{m}: dados mais recentes em {max(anos)} (esperado ≥ 2024)")
        else:
            ok(f"{m}: anos {anos[0]}–{anos[-1]} ✓")


def main() -> None:
    print("\n=== VALIDATE · Cluster 4 — Modos de Vida ===\n")

    path = STAGING_DIR / "mdv_transformed.parquet"
    if not path.exists():
        erro("mdv_transformed.parquet não encontrado — corre transform primeiro")
        return

    df = pd.read_parquet(path)
    print(f"  Carregados {len(df)} registos · "
          f"{df['metrica_codigo'].nunique()} métricas · "
          f"{df['codigo_ine'].nunique()} municípios\n")

    print("[ Cobertura municipal por métrica ]")
    check_cobertura(df)

    print("\n[ Nulos por métrica ]")
    nulos = check_nulos(df)

    print("\n[ Outliers (Z-score > 3) ]")
    outliers = check_outliers(df)

    print("\n[ Scores normalizados ]")
    check_scores(df)

    print("\n[ Percentagens bounded ]")
    check_percentagens(df)

    print("\n[ Valores positivos ]")
    check_valores_positivos(df)

    print("\n[ Séries temporais CSP ]")
    check_anos_csp(df)

    print("\n[ Estatísticas por métrica ]")
    stats = (
        df.groupby("metrica_codigo")["valor"]
        .agg(["count", "min", "max", "mean"])
        .round(3)
    )
    print(stats.to_string())

    report = {
        "timestamp":    datetime.now().isoformat(),
        "cluster":      "Modos de Vida",
        "total_rows":   len(df),
        "n_metricas":   int(df["metrica_codigo"].nunique()),
        "n_municipios": int(df["codigo_ine"].nunique()),
        "avisos":       AVISOS,
        "erros":        ERROS,
        "nulos_pct":    nulos,
        "outliers":     outliers,
        "stats":        stats.reset_index().to_dict(orient="records"),
    }

    report_path = STAGING_DIR / "mdv_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    if ERROS:
        print(f"  ✗ {len(ERROS)} erro(s) — corrigir antes do load")
    elif AVISOS:
        print(f"  ⚠ {len(AVISOS)} aviso(s) — pode prosseguir")
    else:
        print("  ✓ Sem problemas — pronto para load")
    print(f"  Relatório: data/staging/mdv_quality_report.json")


if __name__ == "__main__":
    main()
