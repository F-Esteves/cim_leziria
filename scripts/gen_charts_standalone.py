import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.patheffects as pe
matplotlib.use("Agg")
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import numpy as np
import warnings
import json
import re
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
STAGING = BASE_DIR / "data" / "staging"
OUT = BASE_DIR / "reports" / "charts_standalone"
GEOJSON = BASE_DIR / "data" / "ContinenteConcelhos.geojson"
MUNICIPIO_REF = "Santar\xe9m"

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
"font.family": "Calibri",
"font.size": 11,
"axes.edgecolor": "#D9D9D9",
"axes.grid": True,
"grid.color": "#EDEDED",
"grid.linewidth": 0.6,
"axes.spines.top": False,
"axes.spines.right": False,
"figure.facecolor": "white",
"axes.titlecolor": "#1F2A44",
"text.color": "#262626",
"axes.labelcolor": "#262626",
"xtick.color": "#404040",
"ytick.color": "#404040",
})

MUNICIPIOS_CIM = [
"Almeirim", "Alpiar\xe7a", "Azambuja", "Benavente", "Cartaxo",
"Chamusca", "Coruche", "Goleg\xe3", "Rio Maior",
"Salvaterra de Magos", "Santar\xe9m",
]

municipios_cim_upper = [m.upper() for m in MUNICIPIOS_CIM]

gdf_base = gpd.read_file(GEOJSON)
gdf_base = gdf_base.to_crs(epsg=3763)
gdf_base["codigo_ine"] = gdf_base["DICO"].astype(int)

gdf_cim = gdf_base[gdf_base["Concelho"].str.upper().isin(municipios_cim_upper)].copy()

narrativas = {}
resumo_indicadores = []  # dados estruturados p/ a tabela-s\xedntese (sem\xe1foro) do relat\xf3rio

def carregar(cluster):
    df = pd.read_parquet(f"{STAGING}/{cluster}_transformed.parquet")
    df["codigo_ine"] = pd.to_numeric(df["codigo_ine"], errors="coerce")
    return df

def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{nome}.png", dpi=145, bbox_inches="tight")
    plt.close(fig)

def choropleth_fig(df, metrica, ano, cmap="Blues", title="", figsize=(6, 5.6)):
    fig, ax = plt.subplots(figsize=figsize)
    dados = df[(df["metrica_codigo"] == metrica) & (df["ano"] == ano) &
               (df["nome"] != "Portugal") & (df["nome"] != "Lez\xedria do Tejo")]
    dados = dados[["codigo_ine", "valor"]].dropna(subset=["codigo_ine"])
    gdf = gdf_cim.merge(dados, on="codigo_ine", how="left")

    gdf.plot(column="valor", cmap=cmap, edgecolor="#707070", linewidth=0.7, ax=ax,
              legend=True, legend_kwds={"shrink": 0.65, "label": ""},
              missing_kwds={"color": "#EEEEEE"})

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight="bold", loc="center", pad=12)
    return fig, ax

def bar_fig(labels, series_dict, colors, title="", fmt="{:.1f}", figsize=(7, 4.3), ylabel=""):
    fig, ax = plt.subplots(figsize=figsize)
    n_series = len(series_dict)
    x = np.arange(len(labels))
    width = 0.8 / n_series
    for i, (nome_serie, valores) in enumerate(series_dict.items()):
        pos = x - 0.4 + width/2 + i*width
        bars = ax.bar(pos, valores, width=width*0.9, label=nome_serie, color=colors[i % len(colors)])
        for b, v in zip(bars, valores):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width()/2, b.get_height()),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.2)
    if n_series > 1:
        ax.legend(fontsize=9, loc="upper center", ncol=min(n_series, 4), frameon=True,
                  facecolor="white", framealpha=0.85, edgecolor="none")
    return fig, ax

def linha_fig(x, y, color, title="", fmt="{:.1f}", fill=True, figsize=(7, 4.3), ylabel="", label=None):
    fig, ax = plt.subplots(figsize=figsize)
    x = list(x)
    ax.plot(x, y, marker="o", color=color, linewidth=2.4, markersize=6, label=label)
    if fill:
        ax.fill_between(x, y, min(y)*0.95 if min(y) > 0 else min(y)*1.05, alpha=0.12, color=color)
    for xi, yi in zip(x, y):
        ax.annotate(fmt.format(yi), (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(xi)) if float(xi).is_integer() else str(xi) for xi in x], fontsize=9.5)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.15)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5)
    return fig, ax

def multilinha_fig(series_list, title="", figsize=(7, 4.3), ylabel="", fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=figsize)
    all_x = set()
    for x, y, color, label in series_list:
        ax.plot(x, y, marker="o", color=color, linewidth=2.2, markersize=5.5, label=label)
        all_x.update(x)
    ax.set_xticks(sorted(all_x))
    ax.set_xticklabels([str(int(v)) for v in sorted(all_x)], fontsize=9.5)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.2)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5)
    ax.legend(fontsize=9, loc="upper center", ncol=len(series_list), frameon=True,
              facecolor="white", framealpha=0.85, edgecolor="none")
    return fig, ax

def barh_fig(labels, valores, color, title="", fmt="{:.0f}", figsize=(7, 4.6)):
    fig, ax = plt.subplots(figsize=figsize)
    order = np.argsort(valores)
    labels_o = [labels[i] for i in order]
    valores_o = [valores[i] for i in order]
    bars = ax.barh(labels_o, valores_o, color=color)
    for b, v in zip(bars, valores_o):
        ax.annotate(fmt.format(v), (b.get_width(), b.get_y() + b.get_height()/2),
                    textcoords="offset points", xytext=(5, 0), va="center", fontsize=9)
    ax.set_xlim(min(0, min(valores_o)*1.1), max(valores_o) * 1.15)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    return fig, ax

def _limites_eixo(valores_o, extra=None, frac=0.18):
    """Calcula limites do eixo X com margem proporcional ao INTERVALO dos
    dados, em vez de multiplicar o valor m\xe1ximo por uma constante (ex.:
    xmax = max(valores)*1.2) \u2014 essa conta inverte-se quando os valores s\xe3o
    negativos (ex.: -34*1.22 = -41.5, que \xe9 MENOR que -34, cortando a barra
    em vez de lhe dar margem). Funciona com qualquer combina\xe7\xe3o de sinais."""
    todos = list(valores_o) + ([extra] if extra is not None else [])
    lo, hi = min(todos), max(todos)
    rng = (hi - lo) if hi != lo else (abs(hi) if hi != 0 else 1)
    pad = rng * frac
    xmin = min(0, lo - pad * 0.3)
    xmax = hi + pad
    return xmin, xmax


def barh_ref_fig(labels, valores, cim_valor, color, title="", fmt="{:.0f}", figsize=(7, 4.8), cim_label="Lez\xedria do Tejo"):
    fig, ax = plt.subplots(figsize=figsize)
    order = np.argsort(valores)
    labels_o = [labels[i] for i in order]
    valores_o = [valores[i] for i in order]
    bars = ax.barh(labels_o, valores_o, color=color)
    for b, v in zip(bars, valores_o):
        ax.annotate(fmt.format(v), (b.get_width(), b.get_y() + b.get_height()/2),
                    textcoords="offset points", xytext=(5, 0), va="center", fontsize=9)
    ax.axvline(cim_valor, color="#333333", linestyle="--", linewidth=1.8, zorder=5)
    xmin, xmax = _limites_eixo(valores_o, extra=cim_valor, frac=0.18)
    ax.set_xlim(xmin, xmax)
    ax.annotate(f"{cim_label}: {fmt.format(cim_valor)}", (cim_valor, len(labels_o) - 1 + 0.55),
                ha="center", va="bottom", fontsize=8.5, color="#333333", fontweight="bold",
                annotation_clip=False)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=22)
    return fig, ax

def barh_ref_grid_fig(paineis, ncols=2, figsize=(11, 8.5), cim_label="CIM", mostrar_linha=True):
    nrows = int(np.ceil(len(paineis) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()
    for ax, p in zip(axes, paineis):
        labels, valores, cim_valor, color = p["labels"], p["valores"], p["cim_valor"], p["color"]
        fmt = p.get("fmt", "{:.0f}")
        mostrar = p.get("mostrar_linha", mostrar_linha)
        order = np.argsort(valores)
        labels_o = [labels[i] for i in order]
        valores_o = [valores[i] for i in order]
        bars = ax.barh(labels_o, valores_o, color=color, zorder=3)
        for b, v in zip(bars, valores_o):
            ax.annotate(fmt.format(v), (b.get_width(), b.get_y() + b.get_height()/2),
                        textcoords="offset points", xytext=(5, 0), va="center", fontsize=9.5,
                        zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
        if mostrar:
            ax.axvline(cim_valor, color="#555555", linestyle="--", linewidth=1.2, zorder=2)
            xmin, xmax = _limites_eixo(valores_o, extra=cim_valor, frac=0.22)
        else:
            xmin, xmax = _limites_eixo(valores_o, frac=0.22)
        ax.set_xlim(xmin, xmax)
        rotulo = p.get("cim_label", cim_label)
        titulo = p.get("title", "")
        if mostrar:
            titulo = f"{titulo} \xb7 {rotulo}: {fmt.format(cim_valor)}"
        ax.set_title(titulo, fontsize=11.5, fontweight="bold", pad=10)
        ax.tick_params(labelsize=9.5)
    for j in range(len(paineis), len(axes)):
        axes[j].axis("off")
    fig.tight_layout(pad=1.6)
    return fig, axes

def _texto_contraste(cor_hex):
    c = mcolors.to_rgb(cor_hex)
    luminancia = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    return "#262626" if luminancia > 0.62 else "white"

def barh_stacked100_fig(labels, series_dict, colors, title="", figsize=(8, 5.5)):
    fig, ax = plt.subplots(figsize=figsize)
    n = len(labels)
    y = np.arange(n)
    esquerda = np.zeros(n)
    nomes_series = list(series_dict.keys())
    for i, nome_serie in enumerate(nomes_series):
        valores = np.array(series_dict[nome_serie])
        cor = colors[i % len(colors)]
        cor_texto = _texto_contraste(cor)
        ax.barh(y, valores, left=esquerda, color=cor, height=0.65, label=nome_serie)
        for j, (v, l) in enumerate(zip(valores, esquerda)):
            if v > 6:
                ax.annotate(f"{v:.0f}%", (l + v/2, j), ha="center", va="center", fontsize=9,
                            color=cor_texto, fontweight="bold")
        esquerda += valores
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("%", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=38)
    ax.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(nomes_series),
              frameon=True, facecolor="white", framealpha=0.85, edgecolor="none")
    ax.grid(axis="y", visible=False)
    return fig, ax

def small_multiples_fig(dados_por_municipio, title="", ylabel="", fmt="{:.0f}", color="#7B1E3A", figsize=(11, 8), destacar=None):
    municipios = list(dados_por_municipio.keys())
    todos_y = [v for (x, y) in dados_por_municipio.values() for v in y]
    ymin, ymax = min(todos_y), max(todos_y)
    margem = (ymax - ymin) * 0.15 if ymax > ymin else 1
    fig, axes = plt.subplots(3, 4, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    for i, mun in enumerate(municipios):
        ax = axes[i]
        x, y = dados_por_municipio[mun]
        cor_linha = "#1F4E79" if mun == destacar else color
        ax.plot(x, y, marker="o", color=cor_linha, linewidth=2.2 if mun == destacar else 1.8, markersize=3.5)
        ax.fill_between(x, y, ymin - margem, alpha=0.18 if mun == destacar else 0.12, color=cor_linha)
        ax.set_title(mun, fontsize=9.5, fontweight="bold", pad=4, color=cor_linha)
        if mun == destacar:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#1F4E79")
                spine.set_linewidth(1.6)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
        ax.set_ylim(ymin - margem, ymax + margem)
        ax.grid(True, alpha=0.4)
    for j in range(len(municipios), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    if ylabel:
        fig.text(0.02, 0.5, ylabel, va="center", rotation="vertical", fontsize=9.5)
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    return fig, axes

def kpis_row_fig(kpis, figsize=None, card_width=2.4, card_height=1.9):
    if figsize is None:
        # +0.35" extra por cartao a mais, para compensar o espaco "roubado" pelo
        # wspace entre cartoes (sem isto, o texto do cabecalho corta nas margens
        # quando ha 2+ cartoes lado a lado, mesmo com card_width "suficiente"
        # para 1 cartao isolado).
        figsize = (card_width * len(kpis) + 0.35 * (len(kpis) - 1), card_height)
    fig, axes = plt.subplots(1, len(kpis), figsize=figsize)
    if len(kpis) == 1:
        axes = [axes]
    for ax, (valor, label, sublabel, color) in zip(axes, kpis):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        box = FancyBboxPatch((0.03, 0.03), 0.94, 0.94, transform=ax.transAxes,
                             boxstyle="round,pad=0.0,rounding_size=0.09",
                             linewidth=1.4, edgecolor=color, zorder=1)
        box.set_facecolor(mcolors.to_rgba(color, alpha=0.07))
        box.set_edgecolor(color)
        ax.add_patch(box)
        header = FancyBboxPatch((0.03, 0.72), 0.94, 0.25, transform=ax.transAxes,
                                boxstyle="round,pad=0.0,rounding_size=0.09",
                                linewidth=0, facecolor=color, zorder=2)
        ax.add_patch(header)
        ax.add_patch(plt.Rectangle((0.03, 0.72), 0.94, 0.10, transform=ax.transAxes,
                                   linewidth=0, facecolor=color, zorder=2))
        ax.text(0.5, 0.845, label, ha="center", va="center", fontsize=9, fontweight="bold",
                color="white", transform=ax.transAxes, zorder=3, linespacing=1.25)
        ax.text(0.5, 0.42, valor, ha="center", va="center", fontsize=21, fontweight="bold",
                color=color, transform=ax.transAxes, zorder=3)
        if sublabel:
            ax.text(0.5, 0.12, sublabel, ha="center", va="center", fontsize=8.5,
                    color="#767171", transform=ax.transAxes, zorder=3)
    fig.subplots_adjust(wspace=0.12)
    return fig, axes

def donuts_row_fig(donuts, figsize=(8, 3.6)):
    fig, axes = plt.subplots(1, len(donuts), figsize=figsize)
    if len(donuts) == 1:
        axes = [axes]
    for ax, (valor_pct, color, title) in zip(axes, donuts):
        ax.pie([valor_pct, 100 - valor_pct], colors=[color, "#F2D68A"], startangle=90, counterclock=False,
               wedgeprops=dict(width=0.35))
        ax.text(0, 0, f"{valor_pct:.1f}%", ha="center", va="center", fontsize=15, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
    return fig, axes

def valor_grupo(df, grupo, ano=None, municipio_col="nome"):
    d = df if ano is None else df[df["ano"] == ano]
    if grupo == "Portugal":
        sub = d[d[municipio_col] == "Portugal"]
        return sub["valor"].mean() if len(sub) else float("nan")
    if grupo in ("Lez\xedria do Tejo", "CIM"):
        oficial = d[d[municipio_col] == "Lez\xedria do Tejo"]
        if len(oficial):
            return oficial["valor"].mean()
        sub = d[~d[municipio_col].isin(["Portugal", "Lez\xedria do Tejo"])]
        return sub["valor"].mean() if len(sub) else float("nan")
    sub = d[d[municipio_col] == grupo]
    return sub["valor"].mean() if len(sub) else float("nan")

def evolucao_cim(df, municipio_col="nome"):
    anos = sorted(df["ano"].unique())
    valores = [valor_grupo(df, "Lez\xedria do Tejo", ano, municipio_col) for ano in anos]
    return pd.DataFrame({"ano": anos, "valor": valores})

def so_milhares(texto):
    return re.sub(r"(?<=\d),(?=\d)", " ", texto)

from narrativa_engine import gerar_narrativa, avaliar_indicador

print("A gerar gr\xe1ficos individuais (v3 - narrativas autom\xe1ticas)...")

# ═══════════════════════════════════════════════════════════════
# SOCIEDADE
# ═══════════════════════════════════════════════════════════════
soc = carregar("soc")

df_pop = soc[soc["metrica_codigo"] == "soc_pop_total_cim"]
df_dens = soc[soc["metrica_codigo"] == "soc_densidade_pop"]
df_var = soc[soc["metrica_codigo"] == "soc_variacao_populacional_anual"]
df_estr = soc[soc["metrica_codigo"] == "soc_pct_pop_estrangeira"]
df_saldo = soc[soc["metrica_codigo"] == "soc_saldo_natural"]
df_saldo_ac = soc[soc["metrica_codigo"] == "soc_saldo_natural_acumulado"]
df_nat = soc[soc["metrica_codigo"] == "soc_tx_natalidade"]
df_mort_soc = soc[soc["metrica_codigo"] == "soc_tx_mortalidade"]

ultimo_ano_soc = df_pop["ano"].max()
primeiro_ano_soc = df_pop[df_pop["nome"] != "Portugal"]["ano"].min()

fig, ax = choropleth_fig(df_pop, "soc_pop_total_cim", ultimo_ano_soc, cmap="Oranges", title=f"Popula\xe7\xe3o Residente ({int(ultimo_ano_soc)})")
salvar(fig, "soc_01_mapa_populacao")

pop_ini = df_pop[(~df_pop["nome"].isin(["Portugal", "Lez\xedria do Tejo"])) & (df_pop["ano"]==primeiro_ano_soc)]["valor"].sum()
pop_fim = df_pop[(~df_pop["nome"].isin(["Portugal", "Lez\xedria do Tejo"])) & (df_pop["ano"]==ultimo_ano_soc)]["valor"].sum()
cresc_pop = (pop_fim - pop_ini) / pop_ini * 100
mun_maior = df_pop[(~df_pop["nome"].isin(["Portugal", "Lez\xedria do Tejo"])) & (df_pop["ano"]==ultimo_ano_soc)].sort_values("valor", ascending=False).iloc[0]

# soc_01 é um MAPA (distribui\xe7\xe3o por munic\xedpio no ano mais recente) — a legenda
# descreve o que o mapa mostra (quem é o maior), não a evolu\xe7\xe3o temporal da CIM
# (essa fica reservada para soc_02, o gráfico de evolu\xe7\xe3o), para as duas legendas
# não saírem idênticas.
pop_fim_fmt = f"{pop_fim:,.0f}".replace(",", " ")
mun_maior_valor_fmt = f"{mun_maior['valor']:,.0f}".replace(",", " ")
narrativas["soc_01"] = (
    f"Em {int(ultimo_ano_soc)}, a CIM Lez\xedria do Tejo contava com {pop_fim_fmt} habitantes, "
    f"distribu\xeddos de forma heterog\xe9nea pelos 11 munic\xedpios. {mun_maior['nome']} destaca-se "
    f"como o mais populoso, com {mun_maior_valor_fmt} habitantes."
)

pop_cim = df_pop[~df_pop["nome"].isin(["Portugal", "Lez\xedria do Tejo"])].groupby("ano")["valor"].sum().reset_index()
fig, ax = linha_fig(pop_cim["ano"], pop_cim["valor"], "#1F4E79", title="Evolu\xe7\xe3o da Popula\xe7\xe3o Total \u2014 CIM", fmt="{:.0f}")
salvar(fig, "soc_02_evolucao_populacao")

narrativas["soc_02"] = gerar_narrativa(
    chave="soc_pop_total_cim",
    valor_atual=pop_fim,
    valor_anterior=pop_ini,
    contexto={"sujeito": "a popula\xe7\xe3o da CIM", "ano_inicial": primeiro_ano_soc, "ano_final": ultimo_ano_soc},
)
resumo_indicadores.append({
    "cluster": "Sociedade", "nome": "Popula\xe7\xe3o da CIM", "valor_atual": pop_fim, "unidade": " hab.",
    **avaliar_indicador("soc_pop_total_cim", pop_fim, pop_ini),
})

fig, ax = choropleth_fig(df_dens, "soc_densidade_pop", ultimo_ano_soc, cmap="Oranges", title=f"Densidade Populacional ({int(ultimo_ano_soc)})")
salvar(fig, "soc_03_mapa_densidade")

dens_max = df_dens[df_dens["ano"]==ultimo_ano_soc].sort_values("valor", ascending=False).iloc[0]
dens_min = df_dens[df_dens["ano"]==ultimo_ano_soc].sort_values("valor", ascending=True).iloc[0]

narrativas["soc_03"] = gerar_narrativa(
    chave="soc_densidade_pop",
    valor_atual=dens_max["valor"],
    valor_anterior=None,
    contexto={"sujeito": f"{dens_max['nome']}", "ano_inicial": ultimo_ano_soc, "ano_final": ultimo_ano_soc},
    unidade=" hab./km²",
)

var_ultimo = df_var[(df_var["ano"]==ultimo_ano_soc) & (df_var["nome"] != "Portugal")].sort_values("valor", ascending=True)
fig, ax = barh_fig(var_ultimo["nome"].tolist(), var_ultimo["valor"].tolist(), "#BF9270", title=f"Varia\xe7\xe3o Populacional por Munic\xedpio ({int(ultimo_ano_soc)})")
salvar(fig, "soc_04_variacao_populacional")

n_positivos = (var_ultimo["valor"] > 0).sum()

var_max_row = var_ultimo.loc[var_ultimo["valor"].idxmax()]
var_min_row = var_ultimo.loc[var_ultimo["valor"].idxmin()]
narrativas["soc_04"] = (
    f"Em {int(ultimo_ano_soc)}, {n_positivos} dos 11 munic\xedpios da CIM registaram varia\xe7\xe3o populacional "
    f"positiva. {var_max_row['nome']} teve o maior crescimento (+{var_max_row['valor']:.0f} habitantes), "
    f"enquanto {var_min_row['nome']} teve {'a maior quebra' if var_min_row['valor'] < 0 else 'o crescimento mais baixo'} "
    f"({var_min_row['valor']:+.0f} habitantes)."
)

ultimo_ano_estr = df_estr["ano"].max()
fig, ax = choropleth_fig(df_estr, "soc_pct_pop_estrangeira", ultimo_ano_estr, cmap="YlOrBr", title=f"Popula\xe7\xe3o Estrangeira \u2014 % ({int(ultimo_ano_estr)})")
salvar(fig, "soc_05_mapa_populacao_estrangeira")

estr_cim_ultimo = df_estr[~df_estr["nome"].isin(["Portugal", "Lez\xedria do Tejo"])].groupby("ano")["valor"].mean()

narrativas["soc_05"] = gerar_narrativa(
    chave="soc_pct_pop_estrangeira",
    valor_atual=estr_cim_ultimo.iloc[-1],
    valor_anterior=estr_cim_ultimo.iloc[0],
    contexto={"sujeito": "a propor\xe7\xe3o de popula\xe7\xe3o estrangeira na CIM", "ano_inicial": estr_cim_ultimo.index[0], "ano_final": estr_cim_ultimo.index[-1]},
)

fig, ax = linha_fig(estr_cim_ultimo.index, estr_cim_ultimo.values, "#8B5E3C", title="Evolu\xe7\xe3o da Popula\xe7\xe3o Estrangeira \u2014 CIM", fmt="{:.1f}%")
salvar(fig, "soc_06_evolucao_pop_estrangeira")

var_estr = estr_cim_ultimo.iloc[-1] - estr_cim_ultimo.iloc[0]

narrativas["soc_06"] = gerar_narrativa(
    chave="soc_pct_pop_estrangeira",
    valor_atual=estr_cim_ultimo.iloc[-1],
    valor_anterior=estr_cim_ultimo.iloc[0],
    contexto={"sujeito": "o peso da popula\xe7\xe3o estrangeira na CIM", "ano_inicial": estr_cim_ultimo.index[0], "ano_final": estr_cim_ultimo.index[-1]},
)

# Ranking por munic\xedpio (a evolu\xe7\xe3o acima s\xf3 mostra o total da CIM) \u2014
# consistente com o resto do relat\xf3rio, que compara sempre os 11 munic\xedpios.
estr_ultimo_dados = df_estr[(df_estr["ano"] == ultimo_ano_estr) & (~df_estr["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))].sort_values("nome")
estr_cim_valor = valor_grupo(df_estr, "Lez\xedria do Tejo", ultimo_ano_estr)
fig, ax = barh_ref_fig(estr_ultimo_dados["nome"].tolist(), estr_ultimo_dados["valor"].tolist(), estr_cim_valor,
                        "#BF8F3F", title=f"Popula\xe7\xe3o Estrangeira \u2014 % por Munic\xedpio ({int(ultimo_ano_estr)})", fmt="{:.1f}%")
salvar(fig, "soc_06b_ranking_pop_estrangeira")

estr_max = estr_ultimo_dados.loc[estr_ultimo_dados["valor"].idxmax()]
estr_min = estr_ultimo_dados.loc[estr_ultimo_dados["valor"].idxmin()]
narrativas["soc_06b"] = (
    f"Em {int(ultimo_ano_estr)}, o peso da popula\xe7\xe3o estrangeira varia muito entre munic\xedpios: "
    f"{estr_max['valor']:.1f}% em {estr_max['nome']}, face a apenas {estr_min['valor']:.1f}% em {estr_min['nome']} "
    f"\u2014 uma diferen\xe7a de {estr_max['valor'] - estr_min['valor']:.1f} pontos percentuais. A m\xe9dia da CIM \xe9 de {estr_cim_valor:.1f}%."
)

saldo_cim = df_saldo[df_saldo["nome"] != "Portugal"].groupby("ano")["valor"].sum()
fig, ax = bar_fig(saldo_cim.index.astype(str).tolist(), {"": saldo_cim.values}, ["#BF9270"], title="Saldo Natural Anual \u2014 CIM", fmt="{:.0f}")
if ax.get_legend():
    ax.get_legend().remove()
salvar(fig, "soc_07_saldo_natural")

saldo_ac_cim = df_saldo_ac[(df_saldo_ac["nome"] != "Portugal") & (df_saldo_ac["ano"]==df_saldo_ac["ano"].max())]["valor"].sum()

ano_saldo_ini = saldo_cim.index[0]
ano_saldo_fim = saldo_cim.index[-1]
narrativas["soc_07"] = (
    f"O saldo natural (nascimentos menos \xf3bitos) da CIM \xe9 negativo em todos os anos analisados "
    f"({int(ano_saldo_ini)}\u2013{int(ano_saldo_fim)}), passando de {saldo_cim.iloc[0]:.0f} em {int(ano_saldo_ini)} "
    f"para {saldo_cim.iloc[-1]:.0f} em {int(ano_saldo_fim)} \u2014 "
    f"{'uma ligeira melhoria' if saldo_cim.iloc[-1] > saldo_cim.iloc[0] else 'um agravamento'}, mas ainda "
    f"claramente negativo. O acumulado do per\xedodo \xe9 de {saldo_ac_cim:.0f}. O crescimento populacional geral "
    f"da CIM deve-se, portanto, sobretudo a saldo migrat\xf3rio positivo, n\xe3o a crescimento natural."
)

# Ranking por munic\xedpio no \xfaltimo ano dispon\xedvel
ultimo_ano_saldo = df_saldo["ano"].max()
saldo_ultimo_dados = df_saldo[(df_saldo["ano"] == ultimo_ano_saldo) & (~df_saldo["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))].sort_values("nome")
saldo_cim_valor = valor_grupo(df_saldo, "Lez\xedria do Tejo", ultimo_ano_saldo)
fig, ax = barh_ref_fig(saldo_ultimo_dados["nome"].tolist(), saldo_ultimo_dados["valor"].tolist(), saldo_cim_valor,
                        "#BF9270", title=f"Saldo Natural por Munic\xedpio ({int(ultimo_ano_saldo)})", fmt="{:.0f}", cim_label="CIM (m\xe9dia)")
salvar(fig, "soc_07b_ranking_saldo_natural")

saldo_max = saldo_ultimo_dados.loc[saldo_ultimo_dados["valor"].idxmax()]
saldo_min = saldo_ultimo_dados.loc[saldo_ultimo_dados["valor"].idxmin()]
n_negativos = int((saldo_ultimo_dados["valor"] < 0).sum())
narrativas["soc_07b"] = (
    f"Em {int(ultimo_ano_saldo)}, {n_negativos} dos 11 munic\xedpios da CIM t\xeam saldo natural negativo (mais \xf3bitos "
    f"que nascimentos). {saldo_max['nome']} tem o saldo mais favor\xe1vel ({saldo_max['valor']:.0f}), enquanto "
    f"{saldo_min['nome']} tem o mais desfavor\xe1vel ({saldo_min['valor']:.0f})."
)

nat_cim = df_nat[df_nat["nome"] != "Portugal"].groupby("ano")["valor"].mean()
mort_cim = df_mort_soc[df_mort_soc["nome"] != "Portugal"].groupby("ano")["valor"].mean()
fig, ax = multilinha_fig([
    (nat_cim.index, nat_cim.values, "#D9B48F", "Natalidade"),
    (mort_cim.index, mort_cim.values, "#8B5E3C", "Mortalidade"),
], title="Taxa de Natalidade e Mortalidade (\u2030) \u2014 CIM")
salvar(fig, "soc_08_natalidade_mortalidade")

narrativas["soc_08"] = gerar_narrativa(
    chave="soc_tx_mortalidade",
    valor_atual=mort_cim.iloc[-1],
    valor_anterior=mort_cim.iloc[0],
    contexto={"sujeito": "a taxa de mortalidade na CIM", "ano_inicial": nat_cim.index[0], "ano_final": nat_cim.index[-1]},
)

# Ranking por munic\xedpio: natalidade e mortalidade lado a lado
ultimo_ano_nat = df_nat["ano"].max()
nat_dados = df_nat[(df_nat["ano"] == ultimo_ano_nat) & (~df_nat["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))].sort_values("nome")
mort_dados = df_mort_soc[(df_mort_soc["ano"] == ultimo_ano_nat) & (~df_mort_soc["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))].sort_values("nome")
nat_cim_valor = valor_grupo(df_nat, "Lez\xedria do Tejo", ultimo_ano_nat)
mort_cim_valor = valor_grupo(df_mort_soc, "Lez\xedria do Tejo", ultimo_ano_nat)

paineis_nat_mort = [
    dict(labels=nat_dados["nome"].tolist(), valores=nat_dados["valor"].tolist(), cim_valor=nat_cim_valor,
         color="#D9B48F", title="Taxa de Natalidade (\u2030)", fmt="{:.1f}"),
    dict(labels=mort_dados["nome"].tolist(), valores=mort_dados["valor"].tolist(), cim_valor=mort_cim_valor,
         color="#8B5E3C", title="Taxa de Mortalidade (\u2030)", fmt="{:.1f}"),
]
fig, axes = barh_ref_grid_fig(paineis_nat_mort, ncols=2, figsize=(12, 5.6))
salvar(fig, "soc_08b_ranking_natalidade_mortalidade")

nat_max = nat_dados.loc[nat_dados["valor"].idxmax()]
mort_max = mort_dados.loc[mort_dados["valor"].idxmax()]
nat_idx = nat_dados.set_index("nome")["valor"]
mort_idx = mort_dados.set_index("nome")["valor"]
n_mort_supera_nat = int((mort_idx.reindex(nat_idx.index) > nat_idx).sum())
if n_mort_supera_nat == 11:
    frase_padrao = "Em todos os 11 munic\xedpios a mortalidade supera a natalidade"
elif n_mort_supera_nat == 0:
    frase_padrao = "Em nenhum munic\xedpio a mortalidade supera a natalidade"
else:
    frase_padrao = f"Em {n_mort_supera_nat} dos 11 munic\xedpios a mortalidade supera a natalidade"
narrativas["soc_08b"] = (
    f"Em {int(ultimo_ano_nat)}, {nat_max['nome']} tem a taxa de natalidade mais alta da CIM ({nat_max['valor']:.1f}\u2030), "
    f"e {mort_max['nome']} tem a taxa de mortalidade mais alta ({mort_max['valor']:.1f}\u2030). {frase_padrao}, "
    f"um padr\xe3o consistente com o envelhecimento demogr\xe1fico da regi\xe3o."
)

print("\u2713 Sociedade (8 gr\xe1ficos)")

# ═══════════════════════════════════════════════════════════════
# INTRODU\xe7\xc3O
# ═══════════════════════════════════════════════════════════════
municipios_pop = df_pop[(df_pop["ano"]==ultimo_ano_soc) & (~df_pop["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))].sort_values("valor", ascending=False)
municipios_dens = df_dens[df_dens["ano"]==ultimo_ano_soc].set_index("nome")["valor"]

maior_mun = municipios_pop.iloc[0]
menor_mun = municipios_pop.iloc[-1]
mun_mais_denso = municipios_dens.idxmax()
mun_menos_denso = municipios_dens.idxmin()
pop_total_cim = df_pop[(df_pop["nome"]=="Lez\xedria do Tejo") & (df_pop["ano"]==ultimo_ano_soc)]["valor"].values[0]

def fmt_milhar(v):
    return f"{v:,.0f}".replace(",", " ")

narrativas["intro"] = (
f"A Comunidade Intermunicipal (CIM) da Lez\xedria do Tejo integra 11 munic\xedpios do distrito de Santar\xe9m: "
f"Almeirim, Alpiar\xe7a, Azambuja, Benavente, Cartaxo, Chamusca, Coruche, Goleg\xe3, Rio Maior, Salvaterra de Magos "
f"e Santar\xe9m \u2014 sede da comunidade e o munic\xedpio mais populoso, com {fmt_milhar(maior_mun['valor'])} habitantes, "
f"seguido por Benavente. Em {int(ultimo_ano_soc)}, a CIM contava com {fmt_milhar(pop_total_cim)} habitantes no total, "
f"distribu\xeddos de forma muito heterog\xe9nea: de {fmt_milhar(maior_mun['valor'])} habitantes em {maior_mun['nome']} a "
f"apenas {fmt_milhar(menor_mun['valor'])} em {menor_mun['nome']}, uma diferen\xe7a de escala de mais de 12 vezes entre "
f"o maior e o menor munic\xedpio. Esta heterogeneidade repete-se na densidade populacional: {mun_mais_denso} \xe9 o "
f"munic\xedpio mais denso ({municipios_dens[mun_mais_denso]:.0f} hab./km\xb2), com um perfil claramente mais urbano, "
f"enquanto {mun_menos_denso} \xe9 o menos denso ({municipios_dens[mun_menos_denso]:.0f} hab./km\xb2), refletindo a sua "
f"vocac\xe3o rural e agr\xedcola. Este relat\xf3rio percorre os seis eixos de an\xe1lise da CIM \u2014 Governan\xe7a, Ambiente, "
f"Mobilidade, Modos de Vida, Economia e Sociedade \u2014 sempre que poss\xedvel comparando os 11 munic\xedpios entre si, "
f"e n\xe3o apenas cada um isoladamente face \xe0 m\xe9dia regional."
)

print("\u2713 Introdu\xe7\xe3o gerada")

tabela_municipios = []
for _, row in municipios_pop.iterrows():
    tabela_municipios.append({
        "municipio": row["nome"],
        "populacao": int(row["valor"]),
        "densidade": round(float(municipios_dens.get(row["nome"], 0)), 1)
    })
with open(f"{OUT}/tabela_municipios.json", "w", encoding="utf-8") as f:
    json.dump({"ano": int(ultimo_ano_soc), "municipios": tabela_municipios}, f, ensure_ascii=False, indent=2)
print("\u2713 Tabela de munic\xedpios gerada")

# ═══════════════════════════════════════════════════════════════
# GOVERNAN\xc7A
# ═══════════════════════════════════════════════════════════════
gov = carregar("gov")

eleicoes_cfg = [
    ("aut", "Aut\xe1rquicas", "gov_abstencao_aut_pct", "gov_participacao_aut_pct"),
    ("ar", "Legislativas (AR)", "gov_abstencao_ar_pct", "gov_participacao_ar_pct"),
    ("pres", "Presidenciais", "gov_abstencao_pres_pct", "gov_participacao_pres_pct"),
]

idx_gov = 1
for codigo, nome_eleicao, cod_abst, cod_part in eleicoes_cfg:
    tipo, cod_metrica, label = "Absten\xe7\xe3o", cod_abst, "Taxa de Absten\xe7\xe3o"
    df_m = gov[gov["metrica_codigo"] == cod_metrica].copy()
    anos = sorted(df_m["ano"].unique())
    ultimo_ano = max(anos)

    fig, ax = choropleth_fig(df_m, cod_metrica, ultimo_ano, cmap="Purples",
                              title=f"{label} \u2014 {nome_eleicao} ({int(ultimo_ano)})")
    chave_mapa = f"gov_{idx_gov:02d}_mapa_{tipo.lower()}_{codigo}"
    salvar(fig, chave_mapa)

    cim_media = evolucao_cim(df_m)
    ultimo_dados = df_m[(df_m["ano"] == ultimo_ano) &
                        (~df_m["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))].sort_values("nome")

    cim_valor_ultimo = cim_media["valor"].iloc[-1]
    fig, ax = barh_ref_fig(ultimo_dados["nome"].tolist(), ultimo_dados["valor"].tolist(),
                            cim_valor_ultimo, "#1F2A54",
                            title=f"{label} \u2014 {nome_eleicao} ({int(ultimo_ano)})", fmt="{:.1f}%")
    chave_grafico = f"gov_{idx_gov:02d}_evolucao_{tipo.lower()}_{codigo}"
    salvar(fig, chave_grafico)

    max_row = cim_media.loc[cim_media["valor"].idxmax()]
    narrativas[chave_mapa] = gerar_narrativa(
        chave=cod_abst,
        valor_atual=cim_media["valor"].iloc[-1],
        valor_anterior=cim_media["valor"].iloc[0],
        contexto={"sujeito": f"a {label.lower()} m\xe9dia na CIM", "ano_inicial": anos[0], "ano_final": ultimo_ano},
    )
    if codigo == "aut":
        resumo_indicadores.append({
            "cluster": "Governan\xe7a", "nome": "Taxa de Absten\xe7\xe3o (Aut\xe1rquicas)",
            "valor_atual": cim_media["valor"].iloc[-1], "unidade": "%",
            **avaliar_indicador(cod_abst, cim_media["valor"].iloc[-1], cim_media["valor"].iloc[0]),
        })

    mun_max = ultimo_dados.loc[ultimo_dados["valor"].idxmax()]
    mun_min = ultimo_dados.loc[ultimo_dados["valor"].idxmin()]
    # Nota: este é um gráfico de RANKING (compara municípios entre si no MESMO
    # ano), não uma série temporal — por isso não passa por gerar_narrativa()
    # (que usa vocabulário de tendência: "subiu"/"desceu"). Usar esse motor aqui
    # produziria frases como "teve uma subida de 51.5%", sugerindo erradamente
    # uma evolução no tempo quando é só a amplitude entre municípios.
    narrativas[chave_grafico] = (
        f"Em {int(ultimo_ano)}, a {label.lower()} nas elei\xe7\xf5es {nome_eleicao.lower()} variou entre "
        f"{mun_min['valor']:.1f}% em {mun_min['nome']} e {mun_max['valor']:.1f}% em {mun_max['nome']}, "
        f"uma amplitude de {mun_max['valor'] - mun_min['valor']:.1f} pontos percentuais entre os 11 munic\xedpios. "
        f"A m\xe9dia da CIM foi de {cim_valor_ultimo:.1f}%."
    )

    idx_gov += 1

print("\u2713 Governan\xe7a - Elei\xe7\xf5es (6 gr\xe1ficos)")

df_bl = gov[gov["metrica_codigo"] == "gov_banda_larga_100hab"]
df_tel = gov[gov["metrica_codigo"] == "gov_telefone_100hab"]
df_tv = gov[gov["metrica_codigo"] == "gov_tv_100hab"]

anos_gov = sorted(df_bl[df_bl["nome"] != "Portugal"]["ano"].unique())
cim_bl = df_bl[df_bl["nome"] != "Portugal"].groupby("ano")["valor"].mean()
cim_tel = df_tel[df_tel["nome"] != "Portugal"].groupby("ano")["valor"].mean()
cim_tv = df_tv[df_tv["nome"] != "Portugal"].groupby("ano")["valor"].mean()

fig, ax = multilinha_fig([
    (anos_gov, cim_bl.values, "#8FAADC", "Banda Larga"),
    (anos_gov, cim_tel.values, "#4472C4", "Telefone"),
    (anos_gov, cim_tv.values, "#1F2A54", "TV"),
], title="Acessos a Servi\xe7os de Telecomunica\xe7\xf5es /100hab \u2014 CIM", fmt="{:.0f}%")
salvar(fig, "gov_13_telecom_evolucao")

crescimento = ((cim_bl.values[-1] / cim_bl.values[0]) ** (1/(len(cim_bl)-1)) - 1) * 100

narrativas["gov_13"] = (
    f"Entre {anos_gov[0]} e {anos_gov[-1]}, o acesso a Telefone foi o mais generalizado na CIM ({cim_tel.values[-1]:.0f}% "
    f"em {anos_gov[-1]}), seguido pela Banda Larga ({cim_bl.values[-1]:.0f}%) e pela TV por subscri\xe7\xe3o "
    f"({cim_tv.values[-1]:.0f}%). A Banda Larga foi a que mais cresceu no per\xedodo, a um ritmo m\xe9dio de "
    f"{crescimento:.1f}% ao ano."
)

fig, axes = kpis_row_fig([
    (f"{crescimento:.2f}%", "Taxa de Crescimento M\xe9dio\nBanda Larga", f"{anos_gov[0]}-{anos_gov[-1]}", "#1F4E79"),
    (f"{cim_bl.values[-1]:.1f}%", "\xcdndice de Acessibilidade\nBanda Larga", str(anos_gov[-1]), "#1F4E79"),
])
salvar(fig, "gov_14_telecom_kpis")

narrativas["gov_14"] = gerar_narrativa(
    chave="gov_banda_larga_100hab",
    valor_atual=cim_bl.values[-1],
    valor_anterior=cim_bl.values[0],
    contexto={"sujeito": "a transi\xe7\xe3o digital da regi\xe3o", "ano_inicial": anos_gov[0], "ano_final": anos_gov[-1]},
)

print("\u2713 Governan\xe7a - Telecomunica\xe7\xf5es (2 gr\xe1ficos)")

# ═══════════════════════════════════════════════════════════════
# AMBIENTE \u2014 Energia
# ═══════════════════════════════════════════════════════════════
amb = carregar("amb")

df_cons_1k = amb[amb["metrica_codigo"] == "amb_consumo_total_1k_hab"]
df_var_cons = amb[amb["metrica_codigo"] == "amb_var_consumo_anual_pct"]
df_smart = amb[amb["metrica_codigo"] == "amb_pct_contadores_smart"]
df_bt_1k = amb[amb["metrica_codigo"] == "amb_consumo_bt_1k_hab"]
df_at_1k = amb[amb["metrica_codigo"] == "amb_consumo_at_1k_hab"]
df_n_cpes = amb[amb["metrica_codigo"] == "amb_n_cpes_total"]
df_acc = amb[amb["metrica_codigo"] == "amb_membros_acc"]
df_aterro = amb[amb["metrica_codigo"] == "amb_taxa_aterro_pct"]
df_recic = amb[amb["metrica_codigo"] == "amb_taxa_reciclagem_pct"]
df_valor = amb[amb["metrica_codigo"] == "amb_taxa_valorizacao_pct"]

# 1. Mapa Varia\xe7\xe3o Anual do Consumo
ultimo_ano_var = df_var_cons["ano"].max()
fig, ax = choropleth_fig(df_var_cons, "amb_var_consumo_anual_pct", ultimo_ano_var, cmap="Greens", title=f"Varia\xe7\xe3o Anual do Consumo ({int(ultimo_ano_var)})")
salvar(fig, "amb_01_mapa_variacao_consumo")

var_max = df_var_cons[df_var_cons["ano"]==ultimo_ano_var].sort_values("valor", ascending=False).iloc[0]

narrativas["amb_01"] = gerar_narrativa(
    chave="amb_var_consumo_anual_pct",
    valor_atual=var_max["valor"],
    valor_anterior=None,
    contexto={"sujeito": f"{var_max['nome']}", "ano_inicial": ultimo_ano_var, "ano_final": ultimo_ano_var},
    unidade="%",
)
var_cim_valor = valor_grupo(df_var_cons, "Lez\xedria do Tejo", ultimo_ano_var)
resumo_indicadores.append({
    "cluster": "Ambiente", "nome": "Varia\xe7\xe3o Anual do Consumo (CIM)", "valor_atual": var_cim_valor, "unidade": "%",
    **avaliar_indicador("amb_var_consumo_anual_pct", var_cim_valor, None),
})

# 2. KPI Total Contadores no Munic\xedpio
n_cpes_ultimo_ano = df_n_cpes["ano"].max()
n_cpes_mun = df_n_cpes[(df_n_cpes["nome"]==MUNICIPIO_REF) & (df_n_cpes["ano"]==n_cpes_ultimo_ano)]["valor"].values[0]
fig, axes = kpis_row_fig([
    (f"{n_cpes_mun:,.0f}".replace(",", " "), f"Total de Contadores\nem {MUNICIPIO_REF}", str(int(n_cpes_ultimo_ano)), "#548235"),
])
salvar(fig, "amb_02_kpi_contadores")

narrativas["amb_02"] = gerar_narrativa(
    chave="amb_n_cpes_total",
    valor_atual=n_cpes_mun,
    valor_anterior=None,
    contexto={"sujeito": f"o n\xfamero de contadores em {MUNICIPIO_REF}", "ano_inicial": n_cpes_ultimo_ano, "ano_final": n_cpes_ultimo_ano},
)

# 3. Consumo Total de Eletricidade Anual (por 1000 hab) \u2014 evolu\xe7\xe3o CIM
cons_cim_1k = df_cons_1k[~df_cons_1k["nome"].isin(["Portugal", "Lez\xedria do Tejo"])].groupby("ano")["valor"].mean()
fig, ax = linha_fig(cons_cim_1k.index, cons_cim_1k.values / 1000, "#548235", title="Consumo Total de Eletricidade Anual (por 1000hab, m\xe9dia CIM)", fmt="{:.0f}K")
salvar(fig, "amb_03_consumo_energia")

var_cons_1k = (cons_cim_1k.values[-1] - cons_cim_1k.values[0]) / cons_cim_1k.values[0] * 100

narrativas["amb_03"] = gerar_narrativa(
    chave="amb_consumo_total_1k_hab",
    valor_atual=cons_cim_1k.values[-1],
    valor_anterior=cons_cim_1k.values[0],
    contexto={"sujeito": "o consumo de eletricidade por 1000 habitantes na CIM", "ano_inicial": cons_cim_1k.index[0], "ano_final": cons_cim_1k.index[-1]},
)

# 4. % Contadores Inteligentes
smart_cim = df_smart[df_smart["nome"] != "Portugal"].groupby("ano")["valor"].mean()
fig, ax = linha_fig(smart_cim.index, smart_cim.values, "#548235", title="% Contadores Inteligentes \u2014 CIM", fmt="{:.0f}%")
salvar(fig, "amb_04_contadores_inteligentes")

narrativas["amb_04"] = gerar_narrativa(
    chave="amb_pct_contadores_smart",
    valor_atual=smart_cim.values[-1],
    valor_anterior=smart_cim.values[0],
    contexto={"sujeito": "a ado\xe7\xe3o de contadores inteligentes na CIM", "ano_inicial": smart_cim.index[0], "ano_final": smart_cim.index[-1]},
)
resumo_indicadores.append({
    "cluster": "Ambiente", "nome": "Contadores Inteligentes", "valor_atual": smart_cim.values[-1], "unidade": "%",
    **avaliar_indicador("amb_pct_contadores_smart", smart_cim.values[-1], smart_cim.values[0]),
})

# 5. Consumo de Eletricidade em Baixa Tens\xe3o \u2014 ranking dos 11 munic\xedpios + linha CIM
ultimo_ano_bt = df_bt_1k["ano"].max()
bt_cim = valor_grupo(df_bt_1k, "Lez\xedria do Tejo", ultimo_ano_bt)
bt_mun = df_bt_1k[(df_bt_1k["nome"]==MUNICIPIO_REF) & (df_bt_1k["ano"]==ultimo_ano_bt)]["valor"].mean()
at_cim = valor_grupo(df_at_1k, "Lez\xedria do Tejo", ultimo_ano_bt)
at_mun = df_at_1k[(df_at_1k["nome"]==MUNICIPIO_REF) & (df_at_1k["ano"]==ultimo_ano_bt)]["valor"].mean()

bt_dados = df_bt_1k[(df_bt_1k["ano"]==ultimo_ano_bt) & (~df_bt_1k["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))]
fig, ax = barh_ref_fig(bt_dados["nome"].tolist(), (bt_dados["valor"]/1000).tolist(), bt_cim/1000, "#A9D18E",
                        title=f"Consumo em Baixa Tens\xe3o por 1000hab, por Munic\xedpio ({int(ultimo_ano_bt)})", fmt="{:.0f}K")
salvar(fig, "amb_05_consumo_bt_at")

bt_max = bt_dados.sort_values("valor", ascending=False).iloc[0]

narrativas["amb_05"] = gerar_narrativa(
    chave="amb_consumo_bt_1k_hab",
    valor_atual=bt_max["valor"],
    valor_anterior=None,
    contexto={"sujeito": f"{bt_max['nome']}", "ano_inicial": ultimo_ano_bt, "ano_final": ultimo_ano_bt},
    unidade=" kWh/1000 hab.",
)

# 6. Membros Comunidades de Energia
acc_ultimo_ano = df_acc["ano"].max()
acc_dados = df_acc[df_acc["ano"] == acc_ultimo_ano].sort_values("valor", ascending=True)
fig, ax = barh_fig(acc_dados["nome"].tolist(), acc_dados["valor"].tolist(), "#548235", title=f"N.\xba de Membros em Comunidades de Energia ({int(acc_ultimo_ano)})")
salvar(fig, "amb_06_comunidades_energia")

mun_lider_acc = acc_dados.iloc[-1]

narrativas["amb_06"] = gerar_narrativa(
    chave="amb_membros_acc",
    valor_atual=mun_lider_acc["valor"],
    valor_anterior=None,
    contexto={"sujeito": f"{mun_lider_acc['nome']}", "ano_inicial": acc_ultimo_ano, "ano_final": acc_ultimo_ano},
    unidade=" membros",
)

print("\u2713 Ambiente - Energia (6 gr\xe1ficos)")

# ═══════════════════════════════════════════════════════════════
# AMBIENTE \u2014 Res\xedduos
# ═══════════════════════════════════════════════════════════════
ultimo_ano_res = df_aterro["ano"].max()
fig, ax = choropleth_fig(df_aterro, "amb_taxa_aterro_pct", ultimo_ano_res, cmap="Greens", title=f"Taxa de Deposi\xe7\xe3o em Aterro ({int(ultimo_ano_res)})")
salvar(fig, "amb_07_mapa_aterro")

aterro_max = df_aterro[df_aterro["ano"]==ultimo_ano_res].sort_values("valor", ascending=False).iloc[0]

narrativas["amb_07"] = gerar_narrativa(
    chave="amb_taxa_aterro_pct",
    valor_atual=aterro_max["valor"],
    valor_anterior=None,
    contexto={"sujeito": f"{aterro_max['nome']}", "ano_inicial": ultimo_ano_res, "ano_final": ultimo_ano_res},
)
resumo_indicadores.append({
    "cluster": "Ambiente", "nome": "Taxa de Aterro (pior munic\xedpio)", "valor_atual": aterro_max["valor"], "unidade": "%",
    **avaliar_indicador("amb_taxa_aterro_pct", aterro_max["valor"], None),
})

municipios_res = sorted(df_aterro[df_aterro["nome"] != "Portugal"]["nome"].unique())
aterro_all = [valor_grupo(df_aterro, m, ultimo_ano_res) for m in municipios_res]
valor_all = [valor_grupo(df_valor, m, ultimo_ano_res) for m in municipios_res]
recic_all = [valor_grupo(df_recic, m, ultimo_ano_res) for m in municipios_res]

ordem_r = sorted(range(len(municipios_res)), key=lambda i: -valor_all[i])
labels_r = [municipios_res[i] for i in ordem_r]
aterro_ord = [aterro_all[i] for i in ordem_r]
valor_ord = [valor_all[i] for i in ordem_r]
recic_ord = [recic_all[i] for i in ordem_r]

fig, ax = barh_stacked100_fig(labels_r, {"Valoriza\xe7\xe3o": valor_ord, "Aterro": aterro_ord},
                               ["#375623", "#8C6244"], title=f"Destino dos Res\xedduos, por Munic\xedpio ({int(ultimo_ano_res)})")
salvar(fig, "amb_08_destino_residuos")

aterro_cim = valor_grupo(df_aterro, "Lez\xedria do Tejo", ultimo_ano_res)
aterro_mun = valor_grupo(df_aterro, MUNICIPIO_REF, ultimo_ano_res)

narrativas["amb_08"] = gerar_narrativa(
    chave="amb_taxa_aterro_pct",
    valor_atual=aterro_mun,
    valor_anterior=None,
    contexto={"sujeito": f"a deposi\xe7\xe3o em aterro em {MUNICIPIO_REF}", "ano_inicial": ultimo_ano_res, "ano_final": ultimo_ano_res},
)

df_valor_mun = df_valor[df_valor["nome"] != "Portugal"].sort_values("ano")
municipios_ordem = df_valor_mun[df_valor_mun["ano"]==ultimo_ano_res].sort_values("valor", ascending=False)["nome"].tolist()
valores_ordem = [df_valor_mun[(df_valor_mun["nome"]==m) & (df_valor_mun["ano"]==ultimo_ano_res)]["valor"].values[0] for m in municipios_ordem]
fig, ax = plt.subplots(figsize=(8, 4.3))
ax.plot(range(len(municipios_ordem)), valores_ordem, marker="o", color="#375623", linewidth=2.2, markersize=6)
ax.fill_between(range(len(municipios_ordem)), valores_ordem, 0, alpha=0.12, color="#375623")
for i, v in enumerate(valores_ordem):
    ax.annotate(f"{v:.0f}%", (i, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
ax.set_xticks(range(len(municipios_ordem)))
ax.set_xticklabels(municipios_ordem, rotation=40, ha="right", fontsize=9)
ax.set_title(f"Taxa de Valoriza\xe7\xe3o de Res\xedduos por Munic\xedpio ({int(ultimo_ano_res)})", fontsize=12, fontweight="bold", pad=12)
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax * 1.2)
salvar(fig, "amb_09_valorizacao_municipio")

narrativas["amb_09"] = gerar_narrativa(
    chave="amb_taxa_valorizacao_pct",
    valor_atual=valores_ordem[0],
    valor_anterior=None,
    contexto={"sujeito": f"{municipios_ordem[0]}", "ano_inicial": ultimo_ano_res, "ano_final": ultimo_ano_res},
)

print("\u2713 Ambiente - Res\xedduos (3 gr\xe1ficos)")

# ═══════════════════════════════════════════════════════════════
# MOBILIDADE \u2014 Parque Autom\xf3vel
# ═══════════════════════════════════════════════════════════════
mob = carregar("mob")

df_registo_total = mob[mob["metrica_codigo"] == "mob_registo_total_1000hab"]
df_registo_total_pct = mob[mob["metrica_codigo"] == "mob_registo_total_pct_cim"]
df_lig = mob[mob["metrica_codigo"] == "mob_registo_ligeiros_1000hab"]
df_pes = mob[mob["metrica_codigo"] == "mob_registo_pesados_1000hab"]
df_tra = mob[mob["metrica_codigo"] == "mob_registo_tratores_1000hab"]
df_ve_rap = mob[mob["metrica_codigo"] == "mob_ve_rapidos_pct"]
df_ve_semi = mob[mob["metrica_codigo"] == "mob_ve_semirrapidos_pct"]
df_ve_priv = mob[mob["metrica_codigo"] == "mob_ve_privados_pct"]
df_ve_pub = mob[mob["metrica_codigo"] == "mob_ve_publicos_pct"]

ultimo_ano_mob = df_registo_total["ano"].max()

# 1. Mapa Registo de Ve\xedculos Novos
fig, ax = choropleth_fig(df_registo_total, "mob_registo_total_1000hab", ultimo_ano_mob, cmap="Purples", title=f"Registo de Ve\xedculos Novos /1000hab ({int(ultimo_ano_mob)})")
salvar(fig, "mob_01_mapa_veiculos")

reg_max = df_registo_total[(df_registo_total["ano"]==ultimo_ano_mob) & (df_registo_total["nome"] != "Lez\xedria do Tejo")].sort_values("valor", ascending=False).iloc[0]
reg_min_mob = df_registo_total[(df_registo_total["ano"]==ultimo_ano_mob) & (df_registo_total["nome"] != "Lez\xedria do Tejo")].sort_values("valor", ascending=True).iloc[0]

narrativas["mob_01"] = gerar_narrativa(
    chave="mob_registo_total_1000hab",
    valor_atual=reg_max["valor"],
    valor_anterior=None,
    contexto={"sujeito": f"{reg_max['nome']}", "ano_inicial": ultimo_ano_mob, "ano_final": ultimo_ano_mob},
    unidade=" registos/1000 hab.",
)

# 2. Registo de Ve\xedculos Ligeiros \u2014 ranking dos 11 munic\xedpios + linha CIM
lig_cim = df_lig[(df_lig["nome"]=="Lez\xedria do Tejo") & (df_lig["ano"]==ultimo_ano_mob)]["valor"].values[0]
lig_mun = df_lig[(df_lig["nome"]==MUNICIPIO_REF) & (df_lig["ano"]==ultimo_ano_mob)]["valor"].values[0]
pes_cim = df_pes[(df_pes["nome"]=="Lez\xedria do Tejo") & (df_pes["ano"]==ultimo_ano_mob)]["valor"].values[0]
pes_mun = df_pes[(df_pes["nome"]==MUNICIPIO_REF) & (df_pes["ano"]==ultimo_ano_mob)]["valor"].values[0]
tra_cim = df_tra[(df_tra["nome"]=="Lez\xedria do Tejo") & (df_tra["ano"]==ultimo_ano_mob)]["valor"].values[0]
tra_mun = df_tra[(df_tra["nome"]==MUNICIPIO_REF) & (df_tra["ano"]==ultimo_ano_mob)]["valor"].values[0]

lig_dados = df_lig[(df_lig["ano"]==ultimo_ano_mob) & (~df_lig["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))]
tra_dados = df_tra[(df_tra["ano"]==ultimo_ano_mob) & (~df_tra["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))]
fig, ax = barh_ref_fig(lig_dados["nome"].tolist(), lig_dados["valor"].tolist(), lig_cim, "#4B2E83",
                        title=f"Registo de Ve\xedculos Ligeiros /1000hab, por Munic\xedpio ({int(ultimo_ano_mob)})", fmt="{:.2f}")
salvar(fig, "mob_02_registos_por_tipo")

tra_max = tra_dados.sort_values("valor", ascending=False).iloc[0]

narrativas["mob_02"] = gerar_narrativa(
    chave="mob_registo_ligeiros_1000hab",
    valor_atual=lig_cim,
    valor_anterior=None,
    contexto={"sujeito": "o registo de ve\xedculos ligeiros na CIM", "ano_inicial": ultimo_ano_mob, "ano_final": ultimo_ano_mob},
    unidade=" registos/1000 hab.",
)

# 3. Evolu\xe7\xe3o do peso do registo total (% CIM)
reg_pct_mun = df_registo_total_pct[df_registo_total_pct["nome"]==MUNICIPIO_REF].sort_values("ano")
fig, ax = linha_fig(reg_pct_mun["ano"], reg_pct_mun["valor"], "#8064A2", title=f"Evolu\xe7\xe3o do Registo de Ve\xedculos \u2014 {MUNICIPIO_REF} (% da CIM)", fmt="{:.1f}%")
salvar(fig, "mob_03_evolucao_veiculos")

narrativas["mob_03"] = gerar_narrativa(
    chave="mob_registo_total_pct_cim",
    valor_atual=reg_pct_mun["valor"].iloc[-1],
    valor_anterior=reg_pct_mun["valor"].iloc[0],
    contexto={"sujeito": f"o peso de {MUNICIPIO_REF} no registo total de ve\xedculos da CIM", "ano_inicial": reg_pct_mun["ano"].iloc[0], "ano_final": reg_pct_mun["ano"].iloc[-1]},
)
resumo_indicadores.append({
    "cluster": "Mobilidade", "nome": f"Peso de {MUNICIPIO_REF} no Registo de Ve\xedculos", "valor_atual": reg_pct_mun["valor"].iloc[-1], "unidade": "%",
    **avaliar_indicador("mob_registo_total_pct_cim", reg_pct_mun["valor"].iloc[-1], reg_pct_mun["valor"].iloc[0]),
})

# 4. Pontos de Carregamento por tipo
ultimo_ano_carreg = df_ve_rap["ano"].max()
def match_lez(df, ano):
    d = df[df["ano"]==ano]
    sub = d[d["nome"].str.contains("Lez", na=False)]
    if len(sub): return sub["valor"].values[0]
    return d[d["nome"] != "Portugal"]["valor"].mean()

def nome_limpo(s):
    return s.split(": ")[-1] if ": " in s else s

rap_cim = match_lez(df_ve_rap, ultimo_ano_carreg)
semi_cim = match_lez(df_ve_semi, ultimo_ano_carreg)

municipios_carreg = sorted(df_ve_rap[df_ve_rap["ano"]==ultimo_ano_carreg]["nome"].apply(nome_limpo).unique())
rap_all, semi_all = [], []
for m in municipios_carreg:
    r = df_ve_rap[(df_ve_rap["ano"]==ultimo_ano_carreg) & (df_ve_rap["nome"].apply(nome_limpo)==m)]["valor"]
    s = df_ve_semi[(df_ve_semi["ano"]==ultimo_ano_carreg) & (df_ve_semi["nome"].apply(nome_limpo)==m)]["valor"]
    rap_all.append(r.values[0] if len(r) else 0)
    semi_all.append(s.values[0] if len(s) else 0)

ordem_c = sorted(range(len(municipios_carreg)), key=lambda i: -rap_all[i])
labels_c = [municipios_carreg[i] for i in ordem_c]
rap_ord = [rap_all[i] for i in ordem_c]
semi_ord = [semi_all[i] for i in ordem_c]

fig, ax = barh_stacked100_fig(labels_c, {"R\xe1pidos/Ultrarr\xe1pidos": rap_ord, "Semirr\xe1pidos": semi_ord},
                               ["#5C4187", "#B8A2D9"], title=f"Pontos de Carregamento El\xe9trico, por Munic\xedpio ({int(ultimo_ano_carreg)})")
salvar(fig, "mob_04_pontos_carregamento")

narrativas["mob_04"] = gerar_narrativa(
    chave="mob_ve_rapidos_pct",
    valor_atual=rap_ord[0],
    valor_anterior=None,
    contexto={"sujeito": f"{labels_c[0]}", "ano_inicial": ultimo_ano_carreg, "ano_final": ultimo_ano_carreg},
    unidade="%",
)

priv_all, pub_all = [], []
for m in municipios_carreg:
    p1 = df_ve_priv[(df_ve_priv["ano"]==ultimo_ano_carreg) & (df_ve_priv["nome"].apply(nome_limpo)==m)]["valor"]
    p2 = df_ve_pub[(df_ve_pub["ano"]==ultimo_ano_carreg) & (df_ve_pub["nome"].apply(nome_limpo)==m)]["valor"]
    priv_all.append(p1.values[0] if len(p1) else 0)
    pub_all.append(p2.values[0] if len(p2) else 0)

ordem_p = sorted(range(len(municipios_carreg)), key=lambda i: -priv_all[i])
labels_p = [municipios_carreg[i] for i in ordem_p]
priv_ord = [priv_all[i] for i in ordem_p]
pub_ord = [pub_all[i] for i in ordem_p]

fig, ax = barh_stacked100_fig(labels_p, {"Privado": priv_ord, "P\xfablico": pub_ord},
                               ["#5C4187", "#B8A2D9"], title=f"Carregamento Privado vs P\xfablico, por Munic\xedpio ({int(ultimo_ano_carreg)})")
salvar(fig, "mob_05_privado_publico")

narrativas["mob_05"] = gerar_narrativa(
    chave="mob_ve_publicos_pct",
    valor_atual=pub_ord[-1],
    valor_anterior=None,
    contexto={"sujeito": f"{labels_p[-1]}", "ano_inicial": ultimo_ano_carreg, "ano_final": ultimo_ano_carreg},
    unidade="%",
)

print("\u2713 Mobilidade - Parque Autom\xf3vel (5 gr\xe1ficos)")

# ═══════════════════════════════════════════════════════════════
# MODOS DE VIDA \u2014 Sa\xfade
# ═══════════════════════════════════════════════════════════════
mdv = carregar("mdv")

df_hm = mdv[mdv["metrica_codigo"] == "mdv_hab_medico"]
df_hf = mdv[mdv["metrica_codigo"] == "mdv_hab_farmaceutico"]
df_medicos = mdv[mdv["metrica_codigo"] == "mdv_medicos"]
df_enfermeiros = mdv[mdv["metrica_codigo"] == "mdv_enfermeiros"]
df_farmaceuticos = mdv[mdv["metrica_codigo"] == "mdv_farmaceuticos"]
df_dentistas = mdv[mdv["metrica_codigo"] == "mdv_dentistas"]
df_utentes = mdv[mdv["metrica_codigo"] == "mdv_utentes_csp"]
df_consultas = mdv[mdv["metrica_codigo"] == "mdv_consultas_total"]
df_consultas_p = mdv[mdv["metrica_codigo"] == "mdv_consultas_presenciais"]

ultimo_ano_saude = df_hm["ano"].max()

# 1. Mapa Habitantes por M\xe9dico
fig, ax = choropleth_fig(df_hm, "mdv_hab_medico", ultimo_ano_saude, cmap="Oranges", title=f"Habitantes por M\xe9dico ({int(ultimo_ano_saude)})")
salvar(fig, "mdv_01_mapa_hab_medico")

hm_max = df_hm[df_hm["ano"]==ultimo_ano_saude].sort_values("valor", ascending=False).iloc[0]
hm_min = df_hm[df_hm["ano"]==ultimo_ano_saude].sort_values("valor", ascending=True).iloc[0]

narrativas["mdv_01"] = gerar_narrativa(
    chave="mdv_hab_medico",
    valor_atual=hm_max["valor"],
    valor_anterior=None,
    contexto={"sujeito": f"{hm_max['nome']}", "ano_inicial": ultimo_ano_saude, "ano_final": ultimo_ano_saude},
    unidade=" hab./médico",
)

# 2. Profissionais de sa\xfade \u2014 n\xfameros absolutos
def dados_absolutos(df, ano):
    d = df[(df["ano"] == ano) & (~df["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))].set_index("nome")["valor"]
    return d.dropna()

medicos_abs = dados_absolutos(df_medicos, ultimo_ano_saude)
farm_abs = dados_absolutos(df_farmaceuticos, ultimo_ano_saude)
dent_abs = dados_absolutos(df_dentistas, ultimo_ano_saude)
enf_abs = dados_absolutos(df_enfermeiros, ultimo_ano_saude)

paineis_saude = [
    dict(labels=medicos_abs.index.tolist(), valores=medicos_abs.values.tolist(),
         cim_valor=medicos_abs.median(), color="#C55A11", title="N.\xba de M\xe9dicos", fmt="{:.0f}"),
    dict(labels=farm_abs.index.tolist(), valores=farm_abs.values.tolist(),
         cim_valor=farm_abs.median(), color="#C55A11", title="N.\xba de Farmac\xeauticos", fmt="{:.0f}"),
    dict(labels=dent_abs.index.tolist(), valores=dent_abs.values.tolist(),
         cim_valor=dent_abs.median(), color="#C55A11", title="N.\xba de Dentistas", fmt="{:.0f}"),
    dict(labels=enf_abs.index.tolist(), valores=enf_abs.values.tolist(),
         cim_valor=enf_abs.median(), color="#C55A11", title="N.\xba de Enfermeiros", fmt="{:.0f}"),
]

fig, axes = barh_ref_grid_fig(paineis_saude, ncols=2, figsize=(12, 9), mostrar_linha=False)
salvar(fig, "mdv_02_kpis_profissionais")

medicos_mun = df_medicos[(df_medicos["nome"]==MUNICIPIO_REF) & (df_medicos["ano"]==ultimo_ano_saude)]["valor"].values[0]
farm_mun = df_farmaceuticos[(df_farmaceuticos["nome"]==MUNICIPIO_REF) & (df_farmaceuticos["ano"]==ultimo_ano_saude)]["valor"].values[0]
dent_mun = df_dentistas[(df_dentistas["nome"]==MUNICIPIO_REF) & (df_dentistas["ano"]==ultimo_ano_saude)]["valor"].values[0]
enf_mun = df_enfermeiros[(df_enfermeiros["nome"]==MUNICIPIO_REF) & (df_enfermeiros["ano"]==ultimo_ano_saude)]["valor"].values[0]

medicos_min_mun = medicos_abs.idxmin()
narrativas["mdv_02"] = (
    f"Em n\xfamero absoluto de profissionais, {MUNICIPIO_REF} lidera nas quatro categorias \u2014 natural, j\xe1 que "
    f"\xe9 o munic\xedpio mais populoso da CIM \u2014 com {medicos_mun:.0f} m\xe9dicos, {enf_mun:.0f} enfermeiros, "
    f"{farm_mun:.0f} farmac\xeauticos e {dent_mun:.0f} dentistas em {int(ultimo_ano_saude)}. No extremo oposto, "
    f"{medicos_min_mun} tem apenas {medicos_abs.min():.0f} m\xe9dicos e {enf_abs.idxmin()} tem {enf_abs.min():.0f} "
    f"enfermeiros. Esta leitura mostra a dimens\xe3o real de cada equipa de sa\xfade local; para uma compara\xe7\xe3o "
    f"da cobertura ajustada \xe0 popula\xe7\xe3o, ver os r\xe1cios de habitantes por m\xe9dico/farmac\xeautico mais \xe0 "
    f"frente nesta sec\xe7\xe3o."
)

# 3. Utentes inscritos CSP na CIM
ultimo_ano_utentes = df_utentes["ano"].max()
utentes_cim = df_utentes[df_utentes["ano"]==ultimo_ano_utentes]["valor"].iloc[0]
fig, axes = kpis_row_fig([
    (f"{utentes_cim:,.0f}".replace(",", " "), "Utentes Inscritos no CSP\nna CIM", str(int(ultimo_ano_utentes)), "#C55A11"),
])
salvar(fig, "mdv_03_kpi_utentes")

pct_cobertura_csp = utentes_cim / pop_fim * 100
utentes_cim_fmt = f"{utentes_cim:,.0f}".replace(",", " ")
narrativas["mdv_03"] = (
    f"Em {int(ultimo_ano_utentes)}, a CIM tinha {utentes_cim_fmt} utentes inscritos nos Cuidados de Sa\xfade "
    f"Prim\xe1rios (CSP) \u2014 o equivalente a {pct_cobertura_csp:.0f}% da popula\xe7\xe3o residente "
    f"({int(ultimo_ano_soc)})."
)

# 4. Habitantes por M\xe9dico e por Farmac\xeautico
hm_cim = valor_grupo(df_hm, "Lez\xedria do Tejo", ultimo_ano_saude)
hf_cim = valor_grupo(df_hf, "Lez\xedria do Tejo", ultimo_ano_saude)
hm_dados = df_hm[(df_hm["ano"]==ultimo_ano_saude) & (~df_hm["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))].sort_values("nome")
hf_dados = df_hf[(df_hf["ano"]==ultimo_ano_saude) & (~df_hf["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))].sort_values("nome")
hm_mun = df_hm[(df_hm["nome"]==MUNICIPIO_REF) & (df_hm["ano"]==ultimo_ano_saude)]["valor"].values[0]
hf_mun = df_hf[(df_hf["nome"]==MUNICIPIO_REF) & (df_hf["ano"]==ultimo_ano_saude)]["valor"].values[0]

paineis_hab = [
    dict(labels=hm_dados["nome"].tolist(), valores=hm_dados["valor"].tolist(), cim_valor=hm_cim,
         color="#C55A11", title="Habitantes por M\xe9dico", fmt="{:.0f}", cim_label="CIM"),
    dict(labels=hf_dados["nome"].tolist(), valores=hf_dados["valor"].tolist(), cim_valor=hf_cim,
         color="#C55A11", title="Habitantes por Farmac\xeautico", fmt="{:.0f}", cim_label="CIM"),
]

fig, axes = barh_ref_grid_fig(paineis_hab, ncols=2, figsize=(12, 5.6))
salvar(fig, "mdv_04_hab_farmaceuticos_medicos")

hm_max_mun = hm_dados.loc[hm_dados["valor"].idxmax()]
hf_max_mun = hf_dados.loc[hf_dados["valor"].idxmax()]
narrativas["mdv_04"] = (
    f"Em {MUNICIPIO_REF}, h\xe1 {hm_mun:.0f} habitantes por m\xe9dico e {hf_mun:.0f} habitantes por farmac\xeautico "
    f"(m\xe9dia da CIM: {hm_cim:.0f} e {hf_cim:.0f}, respetivamente). A pior cobertura m\xe9dica \xe9 em "
    f"{hm_max_mun['nome']} ({hm_max_mun['valor']:.0f} hab./m\xe9dico) e a pior cobertura farmac\xeautica \xe9 em "
    f"{hf_max_mun['nome']} ({hf_max_mun['valor']:.0f} hab./farmac\xeautico)."
)

# 5. Consultas CSP na CIM
consultas_total_cim = df_consultas[df_consultas["nome"] != "Portugal"].groupby("ano")["valor"].sum()
consultas_p_cim = df_consultas_p[df_consultas_p["nome"] != "Portugal"].groupby("ano")["valor"].sum()
fig, ax = multilinha_fig([
    (consultas_total_cim.index, consultas_total_cim.values/1e3, "#C55A11", "Consultas Totais"),
    (consultas_p_cim.index, consultas_p_cim.values/1e3, "#F4B183", "Consultas Presenciais"),
], title="Consultas CSP na CIM (milhares)", fmt="{:.0f}")
salvar(fig, "mdv_05_consultas_csp")

var_consultas = (consultas_total_cim.values[-1] - consultas_total_cim.values[0]) / consultas_total_cim.values[0] * 100

narrativas["mdv_05"] = gerar_narrativa(
    chave="mdv_consultas_total",
    valor_atual=consultas_total_cim.values[-1],
    valor_anterior=consultas_total_cim.values[0],
    contexto={"sujeito": "o n\xfamero de consultas totais nos CSP da CIM", "ano_inicial": consultas_total_cim.index[0], "ano_final": consultas_total_cim.index[-1]},
)

print("\u2713 Modos de Vida - Sa\xfade (5 gr\xe1ficos)")

# ═══════════════════════════════════════════════════════════════
# MODOS DE VIDA \u2014 Seguran\xe7a
# ═══════════════════════════════════════════════════════════════
df_acid = mdv[mdv["metrica_codigo"] == "mdv_acidentes_vitimas_1000hab"]
df_feridos = mdv[mdv["metrica_codigo"] == "mdv_feridos_acidentes"]
df_mortos = mdv[mdv["metrica_codigo"] == "mdv_mortos_acidentes"]
df_crim_total = mdv[mdv["metrica_codigo"] == "mdv_criminalidade_total"]
df_crim_patrim = mdv[mdv["metrica_codigo"] == "mdv_criminalidade_patrimonio"]
df_crim_integ = mdv[mdv["metrica_codigo"] == "mdv_criminalidade_integridade_fisica"]

ultimo_ano_seg = df_acid["ano"].max()

# 1. Acidentes de Via\xe7\xe3o
feridos_mun = df_feridos[(df_feridos["nome"] != "Portugal") & (df_feridos["ano"]==ultimo_ano_seg)].groupby("nome")["valor"].sum().sort_values(ascending=False)
fig, ax1 = plt.subplots(figsize=(8, 4.6))
ax1b = ax1.twinx()
ax1.bar(feridos_mun.index, feridos_mun.values, color="#F4B183", label="Feridos")
mortos_mun = df_mortos[(df_mortos["nome"] != "Portugal") & (df_mortos["ano"]==ultimo_ano_seg)].set_index("nome")["valor"].reindex(feridos_mun.index).fillna(0)
ax1b.plot(feridos_mun.index, mortos_mun.values, color="#C00000", marker="o", linewidth=2)
ax1.set_xticklabels(feridos_mun.index, rotation=40, ha="right", fontsize=9)
ax1.set_title(f"Acidentes de Via\xe7\xe3o com V\xedtimas ({int(ultimo_ano_seg)})", fontsize=12, fontweight="bold", pad=12)
ax1.set_ylabel("N.\xba de Feridos", fontsize=9.5)
ax1b.set_ylabel("N.\xba de Mortos", fontsize=9.5)
salvar(fig, "mdv_06_acidentes_viacao")

mun_mais_feridos = feridos_mun.index[0]

# Vis\xe3o Zero 2030: meta = 50% do valor de 2019 do pr\xf3prio munic\xedpio. A s\xe9rie
# de dados s\xf3 come\xe7a em 2021 (n\xe3o h\xe1 2019 dispon\xedvel) \u2014 workaround: usa-se o
# primeiro ano dispon\xedvel (2021) como base provis\xf3ria, deixando isso expl\xedcito
# no texto em vez de apresentar como se fosse a meta oficial exata.
feridos_baseline_ano = df_feridos["ano"].min()
feridos_baseline_valor = df_feridos[(df_feridos["nome"]==mun_mais_feridos) & (df_feridos["ano"]==feridos_baseline_ano)]["valor"].sum()
meta_feridos_aprox = 0.5 * feridos_baseline_valor if feridos_baseline_valor > 0 else None

narrativas["mdv_06"] = gerar_narrativa(
    chave="mdv_feridos_acidentes",
    valor_atual=feridos_mun.iloc[0],
    valor_anterior=None,
    contexto={"sujeito": f"{mun_mais_feridos}", "ano_inicial": ultimo_ano_seg, "ano_final": ultimo_ano_seg},
    unidade=" feridos",
    meta_valor_override=meta_feridos_aprox,
)
if meta_feridos_aprox is not None and int(feridos_baseline_ano) != 2019:
    narrativas["mdv_06"] += (
        f" Nota: a meta da Vis\xe3o Zero 2030 usa oficialmente 2019 como ano-base, mas a s\xe9rie dispon\xedvel "
        f"come\xe7a em {int(feridos_baseline_ano)} \u2014 o valor de refer\xeancia usado aqui \xe9 uma aproxima\xe7\xe3o "
        f"(50% do valor de {int(feridos_baseline_ano)}), n\xe3o o alvo oficial exato."
    )

resumo_indicadores.append({
    "cluster": "Modos de Vida", "nome": f"Feridos em Acidentes ({mun_mais_feridos})",
    "valor_atual": feridos_mun.iloc[0], "unidade": " feridos",
    **avaliar_indicador("mdv_feridos_acidentes", feridos_mun.iloc[0], None, meta_valor_override=meta_feridos_aprox),
})

# 2. Taxa de Criminalidade Total \u2014 evolu\xe7\xe3o
crim_total_mun = df_crim_total[df_crim_total["nome"]==MUNICIPIO_REF].sort_values("ano")
fig, ax = linha_fig(crim_total_mun["ano"], crim_total_mun["valor"], "#C55A11", title=f"Taxa de Criminalidade Total \u2014 {MUNICIPIO_REF} (\u2030)", fmt="{:.0f}")
salvar(fig, "mdv_07_criminalidade_evolucao")

var_crim_mun = crim_total_mun["valor"].iloc[-1] - crim_total_mun["valor"].iloc[0]
crim_cim_evol = evolucao_cim(df_crim_total)

narrativas["mdv_07"] = gerar_narrativa(
    chave="mdv_criminalidade_total",
    valor_atual=crim_total_mun["valor"].iloc[-1],
    valor_anterior=crim_total_mun["valor"].iloc[0],
    contexto={"sujeito": f"a taxa de criminalidade total em {MUNICIPIO_REF}", "ano_inicial": crim_total_mun["ano"].iloc[0], "ano_final": crim_total_mun["ano"].iloc[-1]},
)
resumo_indicadores.append({
    "cluster": "Modos de Vida", "nome": f"Taxa de Criminalidade ({MUNICIPIO_REF})",
    "valor_atual": crim_total_mun["valor"].iloc[-1], "unidade": "\u2030",
    **avaliar_indicador("mdv_criminalidade_total", crim_total_mun["valor"].iloc[-1], crim_total_mun["valor"].iloc[0]),
})

# 3. Criminalidade Total \u2014 ranking
patrim_cim = valor_grupo(df_crim_patrim, "Lez\xedria do Tejo", ultimo_ano_seg)
patrim_mun = valor_grupo(df_crim_patrim, MUNICIPIO_REF, ultimo_ano_seg)
total_cim = valor_grupo(df_crim_total, "Lez\xedria do Tejo", ultimo_ano_seg)
integ_cim = valor_grupo(df_crim_integ, "Lez\xedria do Tejo", ultimo_ano_seg)
integ_mun = valor_grupo(df_crim_integ, MUNICIPIO_REF, ultimo_ano_seg)

crim_total_dados = df_crim_total[(df_crim_total["ano"]==ultimo_ano_seg) & (~df_crim_total["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))]
fig, ax = barh_ref_fig(crim_total_dados["nome"].tolist(), crim_total_dados["valor"].tolist(), total_cim, "#C55A11",
                        title=f"Taxa de Criminalidade Total por Munic\xedpio ({int(ultimo_ano_seg)}, \u2030)", fmt="{:.1f}")
salvar(fig, "mdv_08_criminalidade_tipo")

crim_max = crim_total_dados.sort_values("valor", ascending=False).iloc[0]
crim_min = crim_total_dados.sort_values("valor", ascending=True).iloc[0]

narrativas["mdv_08"] = gerar_narrativa(
    chave="mdv_criminalidade_total",
    valor_atual=crim_max["valor"],
    valor_anterior=None,
    contexto={"sujeito": f"{crim_max['nome']}", "ano_inicial": ultimo_ano_seg, "ano_final": ultimo_ano_seg},
    unidade="‰",
)

print("\u2713 Modos de Vida - Seguran\xe7a (3 gr\xe1ficos)")

# ═══════════════════════════════════════════════════════════════
# MODOS DE VIDA \u2014 Educa\xe7\xe3o
# ═══════════════════════════════════════════════════════════════
df_sem_esc = mdv[mdv["metrica_codigo"] == "mdv_sem_escolaridade_pct"]
df_pre = mdv[mdv["metrica_codigo"] == "mdv_ensino_matriculados_pre_escolar_n"]
df_c1 = mdv[mdv["metrica_codigo"] == "mdv_ensino_matriculados_basico_1ciclo_n"]
df_c2 = mdv[mdv["metrica_codigo"] == "mdv_ensino_matriculados_basico_2ciclo_n"]
df_c3 = mdv[mdv["metrica_codigo"] == "mdv_ensino_matriculados_basico_3ciclo_n"]
df_sec = mdv[mdv["metrica_codigo"] == "mdv_ensino_secundario_orientado_n"]
df_sup = mdv[mdv["metrica_codigo"] == "mdv_ensino_superior_inscritos_n"]
df_trans_h = mdv[mdv["metrica_codigo"] == "mdv_tx_transicao_conclusao_h_pct"]
df_trans_m = mdv[mdv["metrica_codigo"] == "mdv_tx_transicao_conclusao_m_pct"]

ultimo_ano_edu = df_sem_esc["ano"].max()
ultimo_ano_edu_serie = df_pre["ano"].max()

fig, ax = choropleth_fig(df_sem_esc, "mdv_sem_escolaridade_pct", ultimo_ano_edu, cmap="Oranges", title=f"Pop. Sem N\xedvel de Escolaridade ({int(ultimo_ano_edu)})")
salvar(fig, "mdv_09_mapa_sem_escolaridade")

sem_esc_max = df_sem_esc[df_sem_esc["ano"]==ultimo_ano_edu].sort_values("valor", ascending=False).iloc[0]
sem_esc_min = df_sem_esc[df_sem_esc["ano"]==ultimo_ano_edu].sort_values("valor", ascending=True).iloc[0]

narrativas["mdv_09"] = gerar_narrativa(
    chave="mdv_sem_escolaridade_pct",
    valor_atual=sem_esc_max["valor"],
    valor_anterior=None,
    contexto={"sujeito": f"{sem_esc_max['nome']}", "ano_inicial": ultimo_ano_edu, "ano_final": ultimo_ano_edu},
    unidade="%",
)

def kpi_ensino(df, ano, municipio):
    return int(df[(df["nome"]==municipio) & (df["ano"]==ano)]["valor"].sum())

niveis = [
    ("Pr\xe9-Escolar", df_pre), ("1.\xba Ciclo", df_c1), ("2.\xba Ciclo", df_c2),
    ("3.\xba Ciclo", df_c3), ("Secund\xe1rio", df_sec), ("Ens. Superior", df_sup),
]

valores_niveis = [kpi_ensino(df, ultimo_ano_edu_serie, MUNICIPIO_REF) for _, df in niveis]

municipios_edu = sorted(df_pre[(df_pre["ano"]==ultimo_ano_edu_serie) & (~df_pre["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))]["nome"].unique())
tabela_niveis_mun = {nome_nivel: [] for nome_nivel, _ in niveis}
totais_mun = []
for mun in municipios_edu:
    vals = [kpi_ensino(df, ultimo_ano_edu_serie, mun) for _, df in niveis]
    total = sum(vals) or 1
    totais_mun.append(total)
    for (nome_nivel, _), v in zip(niveis, vals):
        tabela_niveis_mun[nome_nivel].append(v / total * 100)

cores_niveis = ["#F4B183", "#E8926B", "#C55A11", "#9C3D0A", "#6D2906", "#3D1603"]
fig, ax = barh_stacked100_fig(municipios_edu, tabela_niveis_mun, cores_niveis,
                               title=f"Distribui\xe7\xe3o de Alunos Matriculados por N\xedvel de Ensino ({int(ultimo_ano_edu_serie)})",
                               figsize=(9, 5.8))
salvar(fig, "mdv_10_kpis_niveis_ensino")

nivel_maior = niveis[valores_niveis.index(max(valores_niveis))][0]
pre_pct_range = tabela_niveis_mun["Pr\xe9-Escolar"]
mun_mais_pre = municipios_edu[pre_pct_range.index(max(pre_pct_range))]
mun_menos_pre = municipios_edu[pre_pct_range.index(min(pre_pct_range))]

nomes_niveis = [n for n, _ in niveis]
idx_maior = valores_niveis.index(max(valores_niveis))
narrativas["mdv_10"] = (
    f"Em {MUNICIPIO_REF} ({int(ultimo_ano_edu_serie)}), a distribui\xe7\xe3o de alunos matriculados por n\xedvel de "
    f"ensino \xe9: {', '.join(f'{n} ({v:.0f})' for n, v in zip(nomes_niveis, valores_niveis))}. O n\xedvel com mais "
    f"alunos \xe9 o {nomes_niveis[idx_maior]}. Entre os 11 munic\xedpios, o peso do Pr\xe9-Escolar no total de "
    f"matriculados varia entre {min(pre_pct_range):.0f}% em {mun_menos_pre} e {max(pre_pct_range):.0f}% em "
    f"{mun_mais_pre}, refletindo perfis demogr\xe1ficos distintos."
)

trans_h_v = df_trans_h[(df_trans_h["nome"]==MUNICIPIO_REF) & (df_trans_h["ano"]==ultimo_ano_edu_serie)]["valor"].mean()
trans_m_v = df_trans_m[(df_trans_m["nome"]==MUNICIPIO_REF) & (df_trans_m["ano"]==ultimo_ano_edu_serie)]["valor"].mean()
fig, axes = donuts_row_fig([
    (trans_h_v, "#E8B33D", f"Transi\xe7\xe3o/Reten\xe7\xe3o\nHomens ({int(ultimo_ano_edu_serie)})"),
    (trans_m_v, "#E8B33D", f"Transi\xe7\xe3o/Reten\xe7\xe3o\nMulheres ({int(ultimo_ano_edu_serie)})"),
])
salvar(fig, "mdv_11_transicao_retencao")

narrativas["mdv_11"] = gerar_narrativa(
    chave="mdv_tx_transicao_conclusao_h_pct",
    valor_atual=trans_h_v,
    valor_anterior=trans_m_v,
    contexto={"sujeito": f"a taxa de transi\xe7\xe3o/conclus\xe3o no ensino b\xe1sico em {MUNICIPIO_REF}", "ano_inicial": ultimo_ano_edu_serie, "ano_final": ultimo_ano_edu_serie},
)

print("\u2713 Modos de Vida - Educa\xe7\xe3o (3 gr\xe1ficos)")

# ═══════════════════════════════════════════════════════════════
# MODOS DE VIDA \u2014 Turismo
# ═══════════════════════════════════════════════════════════════
df_dorm = mdv[mdv["metrica_codigo"] == "mdv_dormidas_100hab"]
df_vagos = mdv[mdv["metrica_codigo"] == "mdv_alojamentos_vagos_pct"]
df_sazonal = mdv[mdv["metrica_codigo"] == "mdv_alojamentos_sazonal_pct"]

ultimo_ano_tur = df_dorm["ano"].max()
ultimo_ano_aloj = df_vagos["ano"].max()
vagos_cim = df_vagos[(df_vagos["nome"] != "Portugal") & (df_vagos["ano"]==ultimo_ano_aloj)]["valor"].mean()
sazonal_cim = df_sazonal[(df_sazonal["nome"] != "Portugal") & (df_sazonal["ano"]==ultimo_ano_aloj)]["valor"].mean()

fig, axes = kpis_row_fig([
    (f"{vagos_cim:.0f}%", "Taxa de Alojamentos\nVagos da CIM", str(int(ultimo_ano_aloj)), "#C55A11"),
    (f"{sazonal_cim:.0f}%", "Taxa de Uso Sazonal\nda CIM", str(int(ultimo_ano_aloj)), "#C55A11"),
])
salvar(fig, "mdv_12_kpis_turismo")

narrativas["mdv_12"] = gerar_narrativa(
    chave="mdv_alojamentos_vagos_pct",
    valor_atual=vagos_cim,
    valor_anterior=None,
    contexto={"sujeito": "a taxa de alojamentos vagos da CIM", "ano_inicial": ultimo_ano_aloj, "ano_final": ultimo_ano_aloj},
    unidade="%",
)

fig, ax = choropleth_fig(df_dorm, "mdv_dormidas_100hab", ultimo_ano_tur, cmap="Oranges", title=f"Dormidas /100hab ({int(ultimo_ano_tur)})")
salvar(fig, "mdv_13_mapa_dormidas")

dorm_max = df_dorm[df_dorm["ano"]==ultimo_ano_tur].sort_values("valor", ascending=False).iloc[0]
dorm_cim_media = df_dorm[(df_dorm["ano"]==ultimo_ano_tur) & (df_dorm["nome"] != "Portugal")]["valor"].mean()

narrativas["mdv_13"] = gerar_narrativa(
    chave="mdv_dormidas_100hab",
    valor_atual=dorm_max["valor"],
    valor_anterior=None,
    contexto={"sujeito": f"{dorm_max['nome']}", "ano_inicial": ultimo_ano_tur, "ano_final": ultimo_ano_tur},
    unidade=" dormidas/100 hab.",
)

vagos_mun = df_vagos[(df_vagos["ano"]==ultimo_ano_aloj) & (df_vagos["nome"] != "Portugal")].set_index("nome")["valor"]
fig, ax = barh_fig(vagos_mun.index.tolist(), vagos_mun.values.tolist(), "#F4B183", title=f"Alojamentos Vagos por Munic\xedpio \u2014 {int(ultimo_ano_aloj)}")
salvar(fig, "mdv_14_alojamentos_vagos")

narrativas["mdv_14"] = gerar_narrativa(
    chave="mdv_alojamentos_vagos_pct",
    valor_atual=vagos_mun.max(),
    valor_anterior=None,
    contexto={"sujeito": f"{vagos_mun.idxmax()}", "ano_inicial": ultimo_ano_aloj, "ano_final": ultimo_ano_aloj},
    unidade="%",
)

print("\u2713 Modos de Vida - Turismo (3 gr\xe1ficos)")

# ═══════════════════════════════════════════════════════════════
# ECONOMIA \u2014 Emprego
# ═══════════════════════════════════════════════════════════════
eco = carregar("eco")

df_tconta = eco[eco["metrica_codigo"] == "eco_taxa_conta_propria_pct"]
df_tempreg = eco[eco["metrica_codigo"] == "eco_taxa_grandes_empregadores_pct"]
df_temprego = eco[eco["metrica_codigo"] == "eco_taxa_emprego_pct"]
df_est_agri = eco[eco["metrica_codigo"] == "eco_estrutura_agricultura_pct"]
df_est_ind = eco[eco["metrica_codigo"] == "eco_estrutura_industria_pct"]
df_est_serv = eco[eco["metrica_codigo"] == "eco_estrutura_servicos_pct"]
df_nasc = eco[eco["metrica_codigo"] == "eco_empresas_nascidas_n"]
df_mort = eco[eco["metrica_codigo"] == "eco_empresas_mortas_n"]

ultimo_ano_emp = df_temprego["ano"].max()
tconta_cim = df_tconta[df_tconta["nome"] != "Portugal"]["valor"].mean()
tempreg_cim = df_tempreg[df_tempreg["nome"] != "Portugal"]["valor"].mean()

fig, axes = kpis_row_fig([
    (f"{tconta_cim:.2f}%", "Taxa de Trabalho\npor Conta Pr\xf3pria", str(int(ultimo_ano_emp)), "#C0504D"),
    (f"{tempreg_cim:.2f}%", "Taxa de\nEmpregadores", str(int(ultimo_ano_emp)), "#C0504D"),
])
salvar(fig, "eco_01_kpis_emprego")

narrativas["eco_01"] = gerar_narrativa(
    chave="eco_taxa_conta_propria_pct",
    valor_atual=tconta_cim,
    valor_anterior=None,
    contexto={"sujeito": "a taxa de trabalho por conta pr\xf3pria na CIM", "ano_inicial": ultimo_ano_emp, "ano_final": ultimo_ano_emp},
    unidade="%",
)

fig, ax = choropleth_fig(df_temprego, "eco_taxa_emprego_pct", ultimo_ano_emp, cmap="Reds", title=f"Taxa de Emprego ({int(ultimo_ano_emp)})")
salvar(fig, "eco_02_mapa_taxa_emprego")

temp_max = df_temprego[df_temprego["ano"]==ultimo_ano_emp].sort_values("valor", ascending=False).iloc[0]

narrativas["eco_02"] = gerar_narrativa(
    chave="eco_taxa_emprego_pct",
    valor_atual=temp_max["valor"],
    valor_anterior=None,
    contexto={"sujeito": f"{temp_max['nome']}", "ano_inicial": ultimo_ano_emp, "ano_final": ultimo_ano_emp},
)

municipios_est = sorted(df_est_agri[df_est_agri["nome"] != "Portugal"]["nome"].unique())
agri_v_all = [valor_grupo(df_est_agri, m) for m in municipios_est]
ind_v_all = [valor_grupo(df_est_ind, m) for m in municipios_est]
serv_v_all = [valor_grupo(df_est_serv, m) for m in municipios_est]
agri_v = [valor_grupo(df_est_agri, "Lez\xedria do Tejo"), valor_grupo(df_est_agri, MUNICIPIO_REF)]
ind_v = [valor_grupo(df_est_ind, "Lez\xedria do Tejo"), valor_grupo(df_est_ind, MUNICIPIO_REF)]
serv_v = [valor_grupo(df_est_serv, "Lez\xedria do Tejo"), valor_grupo(df_est_serv, MUNICIPIO_REF)]

ordem = sorted(range(len(municipios_est)), key=lambda i: -serv_v_all[i])
labels_ord = [municipios_est[i] for i in ordem]
agri_ord = [agri_v_all[i] for i in ordem]
ind_ord = [ind_v_all[i] for i in ordem]
serv_ord = [serv_v_all[i] for i in ordem]

fig, ax = barh_stacked100_fig(labels_ord, {"Agricultura": agri_ord, "Ind\xfastria": ind_ord, "Servi\xe7os": serv_ord},
                               ["#F2C4C1", "#C0504D", "#772C2A"], title="Estrutura Setorial do Emprego, por Munic\xedpio")
salvar(fig, "eco_03_estrutura_setorial_emprego")

narrativas["eco_03"] = (
    f"Na CIM, o setor dos Servi\xe7os domina a estrutura do emprego ({serv_v[0]:.0f}%), seguido pela Ind\xfastria "
    f"({ind_v[0]:.0f}%) e pela Agricultura ({agri_v[0]:.0f}%). Em {MUNICIPIO_REF}, os Servi\xe7os representam "
    f"{serv_v[1]:.0f}% do emprego, a Ind\xfastria {ind_v[1]:.0f}% e a Agricultura {agri_v[1]:.0f}%."
)

nasc_mun = df_nasc[df_nasc["nome"]==MUNICIPIO_REF].sort_values("ano")
mort_mun = df_mort[df_mort["nome"]==MUNICIPIO_REF].sort_values("ano")

municipios_nasc = sorted(df_nasc[df_nasc["nome"] != "Portugal"]["nome"].unique())
municipios_nasc = [m for m in municipios_nasc if m != MUNICIPIO_REF] + [MUNICIPIO_REF]
dados_nasc = {}
for m in municipios_nasc:
    sub = df_nasc[df_nasc["nome"]==m].sort_values("ano")
    dados_nasc[m] = (sub["ano"].tolist(), sub["valor"].tolist())

fig, axes = small_multiples_fig(dados_nasc, "Empresas Nascidas por Munic\xedpio (evolu\xe7\xe3o anual)",
                                 ylabel="N.\xba de empresas", color="#C0504D", destacar=MUNICIPIO_REF, figsize=(11, 8))
salvar(fig, "eco_04_dinamica_empresarial")

saldo_empresas = nasc_mun["valor"].iloc[-1] - mort_mun["valor"].iloc[-1]
crescimentos = {m: (dados_nasc[m][1][-1] - dados_nasc[m][1][0]) for m in dados_nasc}
mun_maior_cresc = max(crescimentos, key=crescimentos.get)

narrativas["eco_04"] = gerar_narrativa(
    chave="eco_empresas_nascidas_n",
    valor_atual=nasc_mun["valor"].iloc[-1],
    valor_anterior=nasc_mun["valor"].iloc[0],
    contexto={"sujeito": f"o n\xfamero de empresas nascidas em {MUNICIPIO_REF}", "ano_inicial": nasc_mun["ano"].iloc[0], "ano_final": nasc_mun["ano"].iloc[-1]},
)
resumo_indicadores.append({
    "cluster": "Economia", "nome": f"Empresas Nascidas ({MUNICIPIO_REF})",
    "valor_atual": nasc_mun["valor"].iloc[-1], "unidade": "",
    **avaliar_indicador("eco_empresas_nascidas_n", nasc_mun["valor"].iloc[-1], nasc_mun["valor"].iloc[0]),
})

print("\u2713 Economia - Emprego (4 gr\xe1ficos)")

# ═══════════════════════════════════════════════════════════════
# ECONOMIA \u2014 Rendimento
# ═══════════════════════════════════════════════════════════════
df_rend = eco[eco["metrica_codigo"] == "eco_rendimento_bruto_per_capita_e"]
df_irs = eco[eco["metrica_codigo"] == "eco_irs_per_capita_e"]
df_vn = eco[eco["metrica_codigo"] == "eco_vn_per_capita_e"]
df_ipc = eco[eco["metrica_codigo"] == "eco_ipc_base100"]
df_vn_agri = eco[eco["metrica_codigo"] == "eco_estrutura_vn_agricultura_pct"]
df_vn_ind = eco[eco["metrica_codigo"] == "eco_estrutura_vn_industria_pct"]
df_vn_serv = eco[eco["metrica_codigo"] == "eco_estrutura_vn_servicos_pct"]

ultimo_ano_rend = df_rend["ano"].max()
fig, ax = choropleth_fig(df_rend, "eco_rendimento_bruto_per_capita_e", ultimo_ano_rend, cmap="Reds", title=f"Rendimento Bruto per Capita \u2014 \u20ac ({int(ultimo_ano_rend)})")
salvar(fig, "eco_05_mapa_rendimento")

rend_max = df_rend[(df_rend["ano"]==ultimo_ano_rend) & (df_rend["nome"] != "Portugal")].sort_values("valor", ascending=False).iloc[0]
rend_min = df_rend[(df_rend["ano"]==ultimo_ano_rend) & (df_rend["nome"] != "Portugal")].sort_values("valor", ascending=True).iloc[0]
rend_cim_media = valor_grupo(df_rend, "Lez\xedria do Tejo", ultimo_ano_rend)
rend_ord_desc = df_rend[(df_rend["ano"]==ultimo_ano_rend) & (~df_rend["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))].sort_values("valor", ascending=False)
n_acima = int((rend_ord_desc["valor"] > rend_cim_media).sum())

narrativas["eco_05"] = gerar_narrativa(
    chave="eco_rendimento_bruto_per_capita_e",
    valor_atual=rend_max["valor"],
    valor_anterior=None,
    contexto={"sujeito": f"{rend_max['nome']}", "ano_inicial": ultimo_ano_rend, "ano_final": ultimo_ano_rend},
    unidade="€",
)

rend_ultimo = df_rend[(df_rend["ano"]==ultimo_ano_rend) & (df_rend["nome"] != "Portugal")].set_index("nome")["valor"]
irs_ultimo = df_irs[(df_irs["ano"]==ultimo_ano_rend) & (df_irs["nome"] != "Portugal")].set_index("nome")["valor"]
comuns = rend_ultimo.index.intersection(irs_ultimo.index)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(rend_ultimo[comuns], irs_ultimo[comuns], s=200, color="#D99795", edgecolor="#C0504D", alpha=0.85)
for m in comuns:
    ax.annotate(m, (rend_ultimo[m], irs_ultimo[m]), fontsize=8.5, ha="center", va="center")
ax.set_title(f"Rendimento per Capita vs IRS per Capita ({int(ultimo_ano_rend)})", fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Rendimento per capita (\u20ac)", fontsize=9.5)
ax.set_ylabel("IRS per capita (\u20ac)", fontsize=9.5)
salvar(fig, "eco_06_scatter_rendimento_irs")

narrativas["eco_06"] = gerar_narrativa(
    chave="eco_rendimento_bruto_per_capita_e",
    valor_atual=rend_ultimo[comuns].mean(),
    valor_anterior=None,
    contexto={"sujeito": "o rendimento per capita na CIM", "ano_inicial": ultimo_ano_rend, "ano_final": ultimo_ano_rend},
    unidade="€",
)

vn_ultimo = df_vn[df_vn["ano"]==df_vn["ano"].max()].sort_values("valor", ascending=False)
fig, ax = barh_fig(vn_ultimo["nome"].tolist(), vn_ultimo["valor"].tolist(), "#D99795", title=f"Volume de Neg\xf3cios (\u20ac/hab) \u2014 {int(df_vn['ano'].max())}")
salvar(fig, "eco_07_volume_negocios")

vn_max = vn_ultimo.iloc[0]

narrativas["eco_07"] = gerar_narrativa(
    chave="eco_vn_per_capita_e",
    valor_atual=vn_max["valor"],
    valor_anterior=None,
    contexto={"sujeito": f"{vn_max['nome']}", "ano_inicial": df_vn["ano"].max(), "ano_final": df_vn["ano"].max()},
    unidade="€",
)

ultimo_ano_vn_est = df_vn_agri["ano"].max()
municipios_vn = sorted(df_vn_agri[df_vn_agri["nome"] != "Portugal"]["nome"].unique())
vn_agri_all = [valor_grupo(df_vn_agri, m, ultimo_ano_vn_est) for m in municipios_vn]
vn_ind_all = [valor_grupo(df_vn_ind, m, ultimo_ano_vn_est) for m in municipios_vn]
vn_serv_all = [valor_grupo(df_vn_serv, m, ultimo_ano_vn_est) for m in municipios_vn]

ordem_vn = sorted(range(len(municipios_vn)), key=lambda i: -vn_ind_all[i])
labels_vn = [municipios_vn[i] for i in ordem_vn]
agri_vn_ord = [vn_agri_all[i] for i in ordem_vn]
ind_vn_ord = [vn_ind_all[i] for i in ordem_vn]
serv_vn_ord = [vn_serv_all[i] for i in ordem_vn]

fig, ax = barh_stacked100_fig(labels_vn, {"Agricultura": agri_vn_ord, "Ind\xfastria": ind_vn_ord, "Servi\xe7os": serv_vn_ord},
                               ["#F2C4C1", "#C0504D", "#772C2A"], title=f"Estrutura do Volume de Neg\xf3cios por Setor, por Munic\xedpio ({int(ultimo_ano_vn_est)})")
salvar(fig, "eco_08_estrutura_vn")

vn_ind_cim = valor_grupo(df_vn_ind, "Lez\xedria do Tejo", ultimo_ano_vn_est)
vn_ind_mun = valor_grupo(df_vn_ind, MUNICIPIO_REF, ultimo_ano_vn_est)

narrativas["eco_08"] = (
    f"Em {MUNICIPIO_REF} ({int(ultimo_ano_vn_est)}), a estrutura do Volume de Neg\xf3cios por setor \xe9: "
    f"Ind\xfastria {vn_ind_mun:.0f}%, Servi\xe7os {vn_serv_all[municipios_vn.index(MUNICIPIO_REF)]:.0f}% e "
    f"Agricultura {vn_agri_all[municipios_vn.index(MUNICIPIO_REF)]:.0f}%. Na m\xe9dia da CIM, a Ind\xfastria pesa "
    f"{vn_ind_cim:.0f}%."
)

# Poder de Compra (\xcdndice per capita, base 100 = m\xe9dia nacional) \u2014 pequenos
# m\xfaltiplos com a evolu\xe7\xe3o de cada munic\xedpio, mesmo padr\xe3o do eco_04.
municipios_ipc = sorted(df_ipc[~df_ipc["nome"].isin(["Portugal", "Lez\xedria do Tejo"])]["nome"].unique())
dados_ipc = {}
for m in municipios_ipc:
    sub = df_ipc[df_ipc["nome"]==m].sort_values("ano")
    dados_ipc[m] = (sub["ano"].tolist(), sub["valor"].tolist())

fig, axes = small_multiples_fig(dados_ipc, "Poder de Compra por Munic\xedpio (\xcdndice per capita, base 100 = PT)",
                                 ylabel="\xcdndice (PT=100)", color="#5C4187", destacar=MUNICIPIO_REF, figsize=(11, 8))
salvar(fig, "eco_09_poder_compra")

ultimo_ano_ipc = df_ipc["ano"].max()
ipc_ultimo_dados = df_ipc[(df_ipc["ano"]==ultimo_ano_ipc) & (~df_ipc["nome"].isin(["Portugal", "Lez\xedria do Tejo"]))]
ipc_cim_valor = valor_grupo(df_ipc, "Lez\xedria do Tejo", ultimo_ano_ipc)
ipc_mun_valor = df_ipc[(df_ipc["nome"]==MUNICIPIO_REF) & (df_ipc["ano"]==ultimo_ano_ipc)]["valor"].values[0]
ipc_max = ipc_ultimo_dados.loc[ipc_ultimo_dados["valor"].idxmax()]
ipc_min = ipc_ultimo_dados.loc[ipc_ultimo_dados["valor"].idxmin()]
n_acima_pt = int((ipc_ultimo_dados["valor"] >= 100).sum())

narrativas["eco_09"] = (
    f"Em {int(ultimo_ano_ipc)}, o \xedndice de poder de compra per capita na CIM \xe9 de {ipc_cim_valor:.1f} "
    f"(base 100 = m\xe9dia nacional) \u2014 ou seja, abaixo da m\xe9dia do pa\xeds. {ipc_max['nome']} tem o valor mais alto "
    f"({ipc_max['valor']:.1f}), face a {ipc_min['nome']} ({ipc_min['valor']:.1f}). Apenas {n_acima_pt} dos 11 munic\xedpios "
    f"est\xe3o acima ou ao n\xedvel da m\xe9dia nacional (\xedndice \u2265 100). Em {MUNICIPIO_REF}, o \xedndice \xe9 de {ipc_mun_valor:.1f}."
)

print("\u2713 Economia - Rendimento (4 gr\xe1ficos)")

# -- Grava as narrativas em disco: sem isto, o gerador do relatorio (Word/PDF)
#    nao tem acesso a nenhum dos textos gerados acima. --
with open(f"{OUT}/narrativas.json", "w", encoding="utf-8") as f:
    json.dump(narrativas, f, ensure_ascii=False, indent=2)

def _tipo_serializavel(o):
    """Converte tipos numpy (int64, float64, etc.) para tipos nativos do
    Python antes de gravar em JSON \u2014 o m\xf3dulo json n\xe3o sabe serializar
    tipos numpy diretamente."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Tipo n\xe3o serializ\xe1vel: {type(o)}")

with open(f"{OUT}/resumo_indicadores.json", "w", encoding="utf-8") as f:
    json.dump(resumo_indicadores, f, ensure_ascii=False, indent=2, default=_tipo_serializavel)
print(f"\u2713 Resumo com {len(resumo_indicadores)} indicadores classificados guardado em resumo_indicadores.json")

print("\n\u2713\u2713\u2713 Todas as narrativas foram geradas automaticamente! \u2713\u2713\u2713")