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
OUT = BASE_DIR / "reports" / "charts_standalone_manual"
GEOJSON = BASE_DIR / "data" / "ContinenteConcelhos.geojson"
MUNICIPIO_REF = "Santarém"

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
    "Almeirim", "Alpiarça", "Azambuja", "Benavente", "Cartaxo",
    "Chamusca", "Coruche", "Golegã", "Rio Maior",
    "Salvaterra de Magos", "Santarém",
]
municipios_cim_upper = [m.upper() for m in MUNICIPIOS_CIM]

gdf_base = gpd.read_file(GEOJSON)
gdf_base = gdf_base.to_crs(epsg=3763)
gdf_base["codigo_ine"] = gdf_base["DICO"].astype(int)

gdf_cim = gdf_base[gdf_base["Concelho"].str.upper().isin(municipios_cim_upper)].copy()

narrativas = {}


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
               (df["nome"] != "Portugal") & (df["nome"] != "Lezíria do Tejo")]
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
    dados, em vez de multiplicar o valor máximo por uma constante (ex.:
    xmax = max(valores)*1.2) — essa conta inverte-se quando os valores são
    negativos (ex.: -34*1.22 = -41.5, que é MENOR que -34, cortando a barra
    em vez de lhe dar margem). Funciona com qualquer combinação de sinais."""
    todos = list(valores_o) + ([extra] if extra is not None else [])
    lo, hi = min(todos), max(todos)
    rng = (hi - lo) if hi != lo else (abs(hi) if hi != 0 else 1)
    pad = rng * frac
    xmin = min(0, lo - pad * 0.3)
    xmax = hi + pad
    return xmin, xmax


def barh_ref_fig(labels, valores, cim_valor, color, title="", fmt="{:.0f}", figsize=(7, 4.8), cim_label="Lezíria do Tejo"):
    """Ranking horizontal dos 11 municípios + linha vertical tracejada com a referência da CIM."""
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
    ax.set_title(f"{title}  ·  {cim_label}: {fmt.format(cim_valor)}", fontsize=11.5, fontweight="bold", pad=10)
    return fig, ax


def barh_ref_grid_fig(paineis, ncols=2, figsize=(11, 8.5), cim_label="CIM", mostrar_linha=True):
    """Grelha de pequenos rankings horizontais, um por painel. A média/CIM aparece no
    título de cada painel (evita rótulos a sobrepor barras) e os valores têm um halo
    branco para se manterem legíveis mesmo quando a linha de referência os atravessa.
    'paineis' é uma lista de dicts: {labels, valores, cim_valor, color, title, fmt}.
    Se mostrar_linha=False, não desenha a linha/valor de referência (útil quando o valor
    de referência não é uma comparação justa, ex. contagens absolutas dominadas por um
    município muito maior que os restantes)."""
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
            titulo = f"{titulo}  ·  {rotulo}: {fmt.format(cim_valor)}"
        ax.set_title(titulo, fontsize=11.5, fontweight="bold", pad=10)
        ax.tick_params(labelsize=9.5)
    for j in range(len(paineis), len(axes)):
        axes[j].axis("off")
    fig.tight_layout(pad=1.6)
    return fig, axes


def _texto_contraste(cor_hex):
    """Devolve branco ou cinza-escuro consoante a luminância da cor de fundo,
    para o número dentro da barra ser sempre legível."""
    c = mcolors.to_rgb(cor_hex)
    luminancia = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    return "#262626" if luminancia > 0.62 else "white"


def barh_stacked100_fig(labels, series_dict, colors, title="", figsize=(8, 5.5)):
    """Barra horizontal empilhada a 100% — uma linha por município (+ eventuais linhas de referência
    como Portugal/CIM), dividida nas proporções de series_dict (que devem somar ~100 por linha)."""
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
            if v > 6:  # só anota se houver espaço para o número não ficar cortado
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
    """Grelha 3x4 de mini-gráficos de linha, um por município (+ opcionalmente a CIM), com a mesma escala Y.
    'destacar' é o nome de um painel a evidenciar visualmente (ex: 'Lezíria do Tejo')."""
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


def kpis_row_fig(kpis, figsize=None, card_width=1.9, card_height=1.5):
    """Cartão com cabeçalho colorido (rótulo) + valor em destaque + subtítulo,
    fundo levemente colorido para parecer um cartão real e não uma caixa vazia.
    Proporção do cartão fixa, independentemente do número de cartões na linha."""
    if figsize is None:
        figsize = (card_width * len(kpis), card_height)
    fig, axes = plt.subplots(1, len(kpis), figsize=figsize)
    if len(kpis) == 1:
        axes = [axes]
    for ax, (valor, label, sublabel, color) in zip(axes, kpis):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # caixa exterior com leve fundo colorido (tint) em vez de branco vazio
        box = FancyBboxPatch((0.03, 0.03), 0.94, 0.94, transform=ax.transAxes,
                              boxstyle="round,pad=0.0,rounding_size=0.09",
                              linewidth=1.4, edgecolor=color,
                              zorder=1)
        box.set_facecolor(mcolors.to_rgba(color, alpha=0.07))
        box.set_edgecolor(color)
        ax.add_patch(box)

        # cabeçalho sólido colorido com o rótulo em branco
        header = FancyBboxPatch((0.03, 0.72), 0.94, 0.25, transform=ax.transAxes,
                                 boxstyle="round,pad=0.0,rounding_size=0.09",
                                 linewidth=0, facecolor=color, zorder=2)
        ax.add_patch(header)
        # retângulo reto por baixo do cabeçalho para tapar o arredondamento inferior
        ax.add_patch(plt.Rectangle((0.03, 0.72), 0.94, 0.10, transform=ax.transAxes,
                                    linewidth=0, facecolor=color, zorder=2))

        ax.text(0.5, 0.845, label, ha="center", va="center", fontsize=9, fontweight="bold",
                color="white", transform=ax.transAxes, zorder=3, linespacing=1.25)
        ax.text(0.5, 0.42, valor, ha="center", va="center", fontsize=25, fontweight="bold",
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
    """Devolve o valor de 'Portugal', a linha oficial 'Lezíria do Tejo' (se existir) ou a média
    calculada da CIM como fallback, ou o valor de um município específico."""
    d = df if ano is None else df[df["ano"] == ano]
    if grupo == "Portugal":
        sub = d[d[municipio_col] == "Portugal"]
        return sub["valor"].mean() if len(sub) else float("nan")
    if grupo in ("Lezíria do Tejo", "CIM"):
        oficial = d[d[municipio_col] == "Lezíria do Tejo"]
        if len(oficial):
            return oficial["valor"].mean()
        sub = d[~d[municipio_col].isin(["Portugal", "Lezíria do Tejo"])]
        return sub["valor"].mean() if len(sub) else float("nan")
    sub = d[d[municipio_col] == grupo]
    return sub["valor"].mean() if len(sub) else float("nan")


def evolucao_cim(df, municipio_col="nome"):
    """Série temporal da CIM: usa a linha oficial 'Lezíria do Tejo' por ano se existir,
    caso contrário calcula a média dos municípios (excluindo Portugal e a própria agregação)."""
    anos = sorted(df["ano"].unique())
    valores = [valor_grupo(df, "Lezíria do Tejo", ano, municipio_col) for ano in anos]
    return pd.DataFrame({"ano": anos, "valor": valores})


def so_milhares(texto):
    """Substitui vírgulas de separador de milhar (entre dígitos) por espaço,
    sem tocar em vírgulas de pontuação normal do texto."""
    return re.sub(r"(?<=\d),(?=\d)", " ", texto)


print("A gerar gráficos...")

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

fig, ax = choropleth_fig(df_pop, "soc_pop_total_cim", ultimo_ano_soc, cmap="Oranges", title=f"População Residente ({int(ultimo_ano_soc)})")
salvar(fig, "soc_01_mapa_populacao")

pop_ini = df_pop[(~df_pop["nome"].isin(["Portugal", "Lezíria do Tejo"])) & (df_pop["ano"]==primeiro_ano_soc)]["valor"].sum()
pop_fim = df_pop[(~df_pop["nome"].isin(["Portugal", "Lezíria do Tejo"])) & (df_pop["ano"]==ultimo_ano_soc)]["valor"].sum()
cresc_pop = (pop_fim - pop_ini) / pop_ini * 100
mun_maior = df_pop[(~df_pop["nome"].isin(["Portugal", "Lezíria do Tejo"])) & (df_pop["ano"]==ultimo_ano_soc)].sort_values("valor", ascending=False).iloc[0]
narrativas["soc_01"] = so_milhares(
    f"Em {int(ultimo_ano_soc)}, a CIM Lezíria do Tejo contava com {pop_fim:,.0f} habitantes, distribuídos de forma "
    f"heterogénea pelos 11 municípios. {mun_maior['nome']} destaca-se como o mais populoso, com "
    f"{mun_maior['valor']:,.0f} habitantes."
)


pop_cim = df_pop[~df_pop["nome"].isin(["Portugal", "Lezíria do Tejo"])].groupby("ano")["valor"].sum().reset_index()
fig, ax = linha_fig(pop_cim["ano"], pop_cim["valor"], "#1F4E79", title="Evolução da População Total — CIM", fmt="{:.0f}")
salvar(fig, "soc_02_evolucao_populacao")

narrativas["soc_02"] = (
    f"Entre {int(primeiro_ano_soc)} e {int(ultimo_ano_soc)}, a população da CIM cresceu {cresc_pop:.1f}%, "
    f"contrariando a tendência de despovoamento observada noutras regiões do interior de Portugal."
)

fig, ax = choropleth_fig(df_dens, "soc_densidade_pop", ultimo_ano_soc, cmap="Oranges", title=f"Densidade Populacional ({int(ultimo_ano_soc)})")
salvar(fig, "soc_03_mapa_densidade")

dens_max = df_dens[df_dens["ano"]==ultimo_ano_soc].sort_values("valor", ascending=False).iloc[0]
dens_min = df_dens[df_dens["ano"]==ultimo_ano_soc].sort_values("valor", ascending=True).iloc[0]
narrativas["soc_03"] = (
    f"A densidade populacional varia significativamente entre municípios: {dens_max['nome']} regista a maior densidade "
    f"({dens_max['valor']:.0f} hab./km²), enquanto {dens_min['nome']} é o menos denso ({dens_min['valor']:.0f} hab./km²)."
)

var_ultimo = df_var[(df_var["ano"]==ultimo_ano_soc) & (df_var["nome"] != "Portugal")].sort_values("valor", ascending=True)
fig, ax = barh_fig(var_ultimo["nome"].tolist(), var_ultimo["valor"].tolist(), "#BF9270", title=f"Variação Populacional por Município ({int(ultimo_ano_soc)})")
salvar(fig, "soc_04_variacao_populacional")

n_positivos = (var_ultimo["valor"] > 0).sum()
narrativas["soc_04"] = (
    f"{n_positivos} dos 11 municípios registaram variação populacional positiva em {int(ultimo_ano_soc)}, "
    f"confirmando a tendência geral de crescimento demográfico na CIM."
)

ultimo_ano_estr = df_estr["ano"].max()
fig, ax = choropleth_fig(df_estr, "soc_pct_pop_estrangeira", ultimo_ano_estr, cmap="YlOrBr", title=f"População Estrangeira — % ({int(ultimo_ano_estr)})")
salvar(fig, "soc_05_mapa_populacao_estrangeira")

estr_cim_ultimo = df_estr[~df_estr["nome"].isin(["Portugal", "Lezíria do Tejo"])].groupby("ano")["valor"].mean()
narrativas["soc_05"] = (
    f"A proporção de população estrangeira na CIM tem vindo a aumentar de forma consistente, atingindo "
    f"{estr_cim_ultimo.iloc[-1]:.1f}% em {int(ultimo_ano_estr)}. Este crescimento acompanha o aumento geral da "
    f"população na região (ver secção 6, Sociedade) e pode estar associado a fluxos de imigração laboral ligados "
    f"à agricultura e à logística, setores relevantes na economia da Lezíria do Tejo."
)

fig, ax = linha_fig(estr_cim_ultimo.index, estr_cim_ultimo.values, "#8B5E3C", title="Evolução da População Estrangeira — CIM", fmt="{:.1f}%")
salvar(fig, "soc_06_evolucao_pop_estrangeira")

var_estr = estr_cim_ultimo.iloc[-1] - estr_cim_ultimo.iloc[0]
narrativas["soc_06"] = (
    f"Entre {int(estr_cim_ultimo.index[0])} e {int(estr_cim_ultimo.index[-1])}, o peso da população estrangeira "
    f"na CIM subiu {var_estr:.1f} pontos percentuais, passando de {estr_cim_ultimo.iloc[0]:.1f}% para "
    f"{estr_cim_ultimo.iloc[-1]:.1f}%. É um crescimento acelerado num período curto, que merece ser monitorizado "
    f"em conjunto com indicadores de integração e acesso a serviços públicos."
)

# Ranking por município (a evolução acima só mostra o total da CIM) —
# consistente com o resto do relatório, que compara sempre os 11 municípios.
estr_ultimo_dados = df_estr[(df_estr["ano"] == ultimo_ano_estr) & (~df_estr["nome"].isin(["Portugal", "Lezíria do Tejo"]))].sort_values("nome")
estr_cim_valor = valor_grupo(df_estr, "Lezíria do Tejo", ultimo_ano_estr)
fig, ax = barh_ref_fig(estr_ultimo_dados["nome"].tolist(), estr_ultimo_dados["valor"].tolist(), estr_cim_valor,
                        "#BF8F3F", title=f"População Estrangeira — % por Município ({int(ultimo_ano_estr)})", fmt="{:.1f}%")
salvar(fig, "soc_06b_ranking_pop_estrangeira")

estr_max = estr_ultimo_dados.loc[estr_ultimo_dados["valor"].idxmax()]
estr_min = estr_ultimo_dados.loc[estr_ultimo_dados["valor"].idxmin()]
narrativas["soc_06b"] = (
    f"Em {int(ultimo_ano_estr)}, o peso da população estrangeira varia muito entre municípios: "
    f"{estr_max['valor']:.1f}% em {estr_max['nome']}, face a apenas {estr_min['valor']:.1f}% em {estr_min['nome']} "
    f"— uma diferença de {estr_max['valor'] - estr_min['valor']:.1f} pontos percentuais. A média da CIM é de {estr_cim_valor:.1f}%."
)

saldo_cim = df_saldo[df_saldo["nome"] != "Portugal"].groupby("ano")["valor"].sum()
fig, ax = bar_fig(saldo_cim.index.astype(str).tolist(), {"": saldo_cim.values}, ["#BF9270"], title="Saldo Natural Anual — CIM", fmt="{:.0f}")
ax.get_legend().remove() if ax.get_legend() else None
salvar(fig, "soc_07_saldo_natural")

saldo_ac_cim = df_saldo_ac[(df_saldo_ac["nome"] != "Portugal") & (df_saldo_ac["ano"]==df_saldo_ac["ano"].max())]["valor"].sum()
narrativas["soc_07"] = (
    f"O saldo natural (nascimentos menos óbitos) da CIM é negativo em todos os anos analisados, acumulando "
    f"{saldo_ac_cim:.0f} no período mais recente disponível. O crescimento populacional geral deve-se, portanto, "
    f"sobretudo a saldo migratório positivo."
)

# Ranking por município no último ano disponível
ultimo_ano_saldo = df_saldo["ano"].max()
saldo_ultimo_dados = df_saldo[(df_saldo["ano"] == ultimo_ano_saldo) & (~df_saldo["nome"].isin(["Portugal", "Lezíria do Tejo"]))].sort_values("nome")
saldo_cim_valor = valor_grupo(df_saldo, "Lezíria do Tejo", ultimo_ano_saldo)
fig, ax = barh_ref_fig(saldo_ultimo_dados["nome"].tolist(), saldo_ultimo_dados["valor"].tolist(), saldo_cim_valor,
                        "#BF9270", title=f"Saldo Natural por Município ({int(ultimo_ano_saldo)})", fmt="{:.0f}", cim_label="CIM (média)")
salvar(fig, "soc_07b_ranking_saldo_natural")

saldo_max = saldo_ultimo_dados.loc[saldo_ultimo_dados["valor"].idxmax()]
saldo_min = saldo_ultimo_dados.loc[saldo_ultimo_dados["valor"].idxmin()]
n_negativos = int((saldo_ultimo_dados["valor"] < 0).sum())
narrativas["soc_07b"] = (
    f"Em {int(ultimo_ano_saldo)}, {n_negativos} dos 11 municípios da CIM têm saldo natural negativo (mais óbitos "
    f"que nascimentos). {saldo_max['nome']} tem o saldo mais favorável ({saldo_max['valor']:.0f}), enquanto "
    f"{saldo_min['nome']} tem o mais desfavorável ({saldo_min['valor']:.0f})."
)

nat_cim = df_nat[df_nat["nome"] != "Portugal"].groupby("ano")["valor"].mean()
mort_cim = df_mort_soc[df_mort_soc["nome"] != "Portugal"].groupby("ano")["valor"].mean()
fig, ax = multilinha_fig([
    (nat_cim.index, nat_cim.values, "#D9B48F", "Natalidade"),
    (mort_cim.index, mort_cim.values, "#8B5E3C", "Mortalidade"),
], title="Taxa de Natalidade e Mortalidade (‰) — CIM")
salvar(fig, "soc_08_natalidade_mortalidade")

narrativas["soc_08"] = (
    f"A taxa de mortalidade ({mort_cim.iloc[-1]:.1f}‰) mantém-se consistentemente acima da taxa de natalidade "
    f"({nat_cim.iloc[-1]:.1f}‰), um padrão demográfico comum em regiões do interior com população envelhecida."
)

# Ranking por município: natalidade e mortalidade lado a lado
ultimo_ano_nat = df_nat["ano"].max()
nat_dados = df_nat[(df_nat["ano"] == ultimo_ano_nat) & (~df_nat["nome"].isin(["Portugal", "Lezíria do Tejo"]))].sort_values("nome")
mort_dados = df_mort_soc[(df_mort_soc["ano"] == ultimo_ano_nat) & (~df_mort_soc["nome"].isin(["Portugal", "Lezíria do Tejo"]))].sort_values("nome")
nat_cim_valor = valor_grupo(df_nat, "Lezíria do Tejo", ultimo_ano_nat)
mort_cim_valor = valor_grupo(df_mort_soc, "Lezíria do Tejo", ultimo_ano_nat)

paineis_nat_mort = [
    dict(labels=nat_dados["nome"].tolist(), valores=nat_dados["valor"].tolist(), cim_valor=nat_cim_valor,
         color="#D9B48F", title="Taxa de Natalidade (‰)", fmt="{:.1f}"),
    dict(labels=mort_dados["nome"].tolist(), valores=mort_dados["valor"].tolist(), cim_valor=mort_cim_valor,
         color="#8B5E3C", title="Taxa de Mortalidade (‰)", fmt="{:.1f}"),
]
fig, axes = barh_ref_grid_fig(paineis_nat_mort, ncols=2, figsize=(12, 5.6))
salvar(fig, "soc_08b_ranking_natalidade_mortalidade")

nat_max = nat_dados.loc[nat_dados["valor"].idxmax()]
mort_max = mort_dados.loc[mort_dados["valor"].idxmax()]
nat_idx = nat_dados.set_index("nome")["valor"]
mort_idx = mort_dados.set_index("nome")["valor"]
n_mort_supera_nat = int((mort_idx.reindex(nat_idx.index) > nat_idx).sum())
if n_mort_supera_nat == 11:
    frase_padrao = "Em todos os 11 municípios a mortalidade supera a natalidade"
elif n_mort_supera_nat == 0:
    frase_padrao = "Em nenhum município a mortalidade supera a natalidade"
else:
    frase_padrao = f"Em {n_mort_supera_nat} dos 11 municípios a mortalidade supera a natalidade"
narrativas["soc_08b"] = (
    f"Em {int(ultimo_ano_nat)}, {nat_max['nome']} tem a taxa de natalidade mais alta da CIM ({nat_max['valor']:.1f}‰), "
    f"e {mort_max['nome']} tem a taxa de mortalidade mais alta ({mort_max['valor']:.1f}‰). {frase_padrao}, "
    f"um padrão consistente com o envelhecimento demográfico da região."
)

print("✓ Sociedade (8 gráficos)")

# ═══════════════════════════════════════════════════════════════
# INTRODUÇÃO — texto rico sobre a CIM e os 11 municípios (para a capa)
# ═══════════════════════════════════════════════════════════════
municipios_pop = df_pop[(df_pop["ano"]==ultimo_ano_soc) & (~df_pop["nome"].isin(["Portugal", "Lezíria do Tejo"]))].sort_values("valor", ascending=False)
municipios_dens = df_dens[df_dens["ano"]==ultimo_ano_soc].set_index("nome")["valor"]

maior_mun = municipios_pop.iloc[0]
menor_mun = municipios_pop.iloc[-1]
mun_mais_denso = municipios_dens.idxmax()
mun_menos_denso = municipios_dens.idxmin()
pop_total_cim = df_pop[(df_pop["nome"]=="Lezíria do Tejo") & (df_pop["ano"]==ultimo_ano_soc)]["valor"].values[0]

def fmt_milhar(v):
    return f"{v:,.0f}".replace(",", " ")

narrativas["intro"] = (
    f"A Comunidade Intermunicipal (CIM) da Lezíria do Tejo integra 11 municípios do distrito de Santarém: "
    f"Almeirim, Alpiarça, Azambuja, Benavente, Cartaxo, Chamusca, Coruche, Golegã, Rio Maior, Salvaterra de Magos "
    f"e Santarém — sede da comunidade e o município mais populoso, com {fmt_milhar(maior_mun['valor'])} habitantes, "
    f"seguido por Benavente. Em {int(ultimo_ano_soc)}, a CIM contava com {fmt_milhar(pop_total_cim)} habitantes no total, "
    f"distribuídos de forma muito heterogénea: de {fmt_milhar(maior_mun['valor'])} habitantes em {maior_mun['nome']} a "
    f"apenas {fmt_milhar(menor_mun['valor'])} em {menor_mun['nome']}, uma diferença de escala de mais de 12 vezes entre "
    f"o maior e o menor município. Esta heterogeneidade repete-se na densidade populacional: {mun_mais_denso} é o "
    f"município mais denso ({municipios_dens[mun_mais_denso]:.0f} hab./km²), com um perfil claramente mais urbano, "
    f"enquanto {mun_menos_denso} é o menos denso ({municipios_dens[mun_menos_denso]:.0f} hab./km²), refletindo a sua "
    f"vocação rural e agrícola. Este relatório percorre os seis eixos de análise da CIM — Governança, Ambiente, "
    f"Mobilidade, Modos de Vida, Economia e Sociedade — sempre que possível comparando os 11 municípios entre si, "
    f"e não apenas cada um isoladamente face à média regional."
)

print("✓ Introdução gerada")

# Tabela dos 11 municípios (população + densidade) para a capa do relatório
tabela_municipios = []
for _, row in municipios_pop.iterrows():
    tabela_municipios.append({
        "municipio": row["nome"],
        "populacao": int(row["valor"]),
        "densidade": round(float(municipios_dens.get(row["nome"], 0)), 1)
    })
with open(f"{OUT}/tabela_municipios.json", "w", encoding="utf-8") as f:
    json.dump({"ano": int(ultimo_ano_soc), "municipios": tabela_municipios}, f, ensure_ascii=False, indent=2)
print("✓ Tabela de municípios gerada")

# ═══════════════════════════════════════════════════════════════
# GOVERNANÇA
# ═══════════════════════════════════════════════════════════════
gov = carregar("gov")

eleicoes_cfg = [
    ("aut", "Autárquicas", "gov_abstencao_aut_pct", "gov_participacao_aut_pct"),
    ("ar", "Legislativas (AR)", "gov_abstencao_ar_pct", "gov_participacao_ar_pct"),
    ("pres", "Presidenciais", "gov_abstencao_pres_pct", "gov_participacao_pres_pct"),
]

idx_gov = 1
for codigo, nome_eleicao, cod_abst, cod_part in eleicoes_cfg:
    # Nota: só se gera a Taxa de Abstenção (é a que consta no relatório). A Taxa de
    # Participação (cod_part) não é usada — para a reativar, basta voltar a iterar
    # sobre [("Abstenção", cod_abst, ...), ("Participação", cod_part, ...)] aqui.
    tipo, cod_metrica, label = "Abstenção", cod_abst, "Taxa de Abstenção"
    df_m = gov[gov["metrica_codigo"] == cod_metrica].copy()
    anos = sorted(df_m["ano"].unique())
    ultimo_ano = max(anos)

    fig, ax = choropleth_fig(df_m, cod_metrica, ultimo_ano, cmap="Purples",
                               title=f"{label} — {nome_eleicao} ({int(ultimo_ano)})")
    chave_mapa = f"gov_{idx_gov:02d}_mapa_{tipo.lower()}_{codigo}"
    salvar(fig, chave_mapa)

    cim_media = evolucao_cim(df_m)
    ultimo_dados = df_m[(df_m["ano"] == ultimo_ano) &
                         (~df_m["nome"].isin(["Portugal", "Lezíria do Tejo"]))].sort_values("nome")

    cim_valor_ultimo = cim_media["valor"].iloc[-1]
    fig, ax = barh_ref_fig(ultimo_dados["nome"].tolist(), ultimo_dados["valor"].tolist(),
                            cim_valor_ultimo, "#1F2A54",
                            title=f"{label} — {nome_eleicao} ({int(ultimo_ano)})", fmt="{:.1f}%")
    chave_grafico = f"gov_{idx_gov:02d}_evolucao_{tipo.lower()}_{codigo}"
    salvar(fig, chave_grafico)

    max_row = cim_media.loc[cim_media["valor"].idxmax()]
    narrativas[chave_mapa] = (
        f"Na última eleição {nome_eleicao.lower()} ({int(ultimo_ano)}), a {label.lower()} média na CIM foi de "
        f"{cim_media['valor'].iloc[-1]:.1f}%. O valor mais alto do período ocorreu em "
        f"{int(max_row['ano'])}, com {max_row['valor']:.1f}%."
    )
    mun_max = ultimo_dados.loc[ultimo_dados["valor"].idxmax()]
    mun_min = ultimo_dados.loc[ultimo_dados["valor"].idxmin()]
    narrativas[chave_grafico] = (
        f"Em {int(ultimo_ano)}, a {label.lower()} nas eleições {nome_eleicao.lower()} variou entre "
        f"{mun_min['valor']:.1f}% em {mun_min['nome']} e {mun_max['valor']:.1f}% em {mun_max['nome']}, "
        f"uma amplitude de {mun_max['valor'] - mun_min['valor']:.1f} pontos percentuais entre os 11 municípios. "
        f"A média da CIM foi de {cim_valor_ultimo:.1f}%."
    )
    idx_gov += 1

print("✓ Governança - Eleições (6 gráficos)")

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
], title="Acessos a Serviços de Telecomunicações /100hab — CIM", fmt="{:.0f}%")
salvar(fig, "gov_13_telecom_evolucao")

crescimento = ((cim_bl.values[-1] / cim_bl.values[0]) ** (1/(len(cim_bl)-1)) - 1) * 100
narrativas["gov_13"] = (
    f"O acesso a Banda Larga cresceu a uma taxa média anual de {crescimento:.2f}% entre {anos_gov[0]} e {anos_gov[-1]}. "
    f"Em {anos_gov[-1]}, a Telefonia continua a ser o serviço com maior penetração ({cim_tel.values[-1]:.0f} acessos/100hab), "
    f"seguida da TV ({cim_tv.values[-1]:.0f}) e Banda Larga ({cim_bl.values[-1]:.0f})."
)

fig, axes = kpis_row_fig([
    (f"{crescimento:.2f}%", "Taxa de Crescimento Médio\nBanda Larga", f"{anos_gov[0]}-{anos_gov[-1]}", "#1F4E79"),
    (f"{cim_bl.values[-1]:.1f}%", "Índice de Acessibilidade\nBanda Larga", str(anos_gov[-1]), "#1F4E79"),
])
salvar(fig, "gov_14_telecom_kpis")

narrativas["gov_14"] = (
    "Estes dois indicadores resumem, de forma direta, a transição digital da região: a taxa de crescimento médio "
    "mostra a velocidade de expansão da rede nos últimos anos, enquanto o índice de acessibilidade dá o retrato "
    "do momento atual. Lidos em conjunto, permitem perceber se a CIM está a acelerar, estabilizar ou perder ritmo "
    "na cobertura de banda larga face às metas de digitalização territorial."
)

print("✓ Governança - Telecomunicações (2 gráficos)")

# ═══════════════════════════════════════════════════════════════
# AMBIENTE — Energia (ATUALIZADO: per 1000hab) + Resíduos
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

# 1. Mapa Variação Anual do Consumo
ultimo_ano_var = df_var_cons["ano"].max()
fig, ax = choropleth_fig(df_var_cons, "amb_var_consumo_anual_pct", ultimo_ano_var, cmap="Greens", title=f"Variação Anual do Consumo ({int(ultimo_ano_var)})")
salvar(fig, "amb_01_mapa_variacao_consumo")

var_max = df_var_cons[df_var_cons["ano"]==ultimo_ano_var].sort_values("valor", ascending=False).iloc[0]
narrativas["amb_01"] = (
    f"Em {int(ultimo_ano_var)}, {var_max['nome']} registou a maior variação anual no consumo de eletricidade "
    f"({var_max['valor']:.1f}%), sinal de crescimento da atividade económica ou populacional no município."
)

# 2. KPI Total Contadores no Município
n_cpes_ultimo_ano = df_n_cpes["ano"].max()
n_cpes_mun = df_n_cpes[(df_n_cpes["nome"]==MUNICIPIO_REF) & (df_n_cpes["ano"]==n_cpes_ultimo_ano)]["valor"].values[0]
fig, axes = kpis_row_fig([
    (f"{n_cpes_mun:,.0f}".replace(",", " "), f"Total de Contadores\nem {MUNICIPIO_REF}", str(int(n_cpes_ultimo_ano)), "#548235"),
])
salvar(fig, "amb_02_kpi_contadores")

narrativas["amb_02"] = so_milhares(
    f"{MUNICIPIO_REF} tinha {n_cpes_mun:,.0f} pontos de consumo elétrico registados em {int(n_cpes_ultimo_ano)}, "
    f"refletindo a dimensão do parque residencial e empresarial ligado à rede."
)


# 3. Consumo Total de Eletricidade Anual (por 1000 hab) — evolução CIM
cons_cim_1k = df_cons_1k[~df_cons_1k["nome"].isin(["Portugal", "Lezíria do Tejo"])].groupby("ano")["valor"].mean()
fig, ax = linha_fig(cons_cim_1k.index, cons_cim_1k.values / 1000, "#548235", title="Consumo Total de Eletricidade Anual (por 1000hab, média CIM)", fmt="{:.0f}K")
salvar(fig, "amb_03_consumo_energia")

var_cons_1k = (cons_cim_1k.values[-1] - cons_cim_1k.values[0]) / cons_cim_1k.values[0] * 100
narrativas["amb_03"] = (
    f"O consumo de eletricidade por 1000 habitantes na CIM {'aumentou' if var_cons_1k > 0 else 'diminuiu'} "
    f"{abs(var_cons_1k):.1f}% entre {int(cons_cim_1k.index[0])} e {int(cons_cim_1k.index[-1])}, uma normalização "
    f"que permite comparar consumo entre municípios de dimensão populacional diferente, ao contrário do valor absoluto."
)

# 4. % Contadores Inteligentes
smart_cim = df_smart[df_smart["nome"] != "Portugal"].groupby("ano")["valor"].mean()
fig, ax = linha_fig(smart_cim.index, smart_cim.values, "#548235", title="% Contadores Inteligentes — CIM", fmt="{:.0f}%")
salvar(fig, "amb_04_contadores_inteligentes")

narrativas["amb_04"] = (
    f"A adoção de contadores inteligentes na CIM passou de {smart_cim.values[0]:.0f}% em {int(smart_cim.index[0])} para "
    f"{smart_cim.values[-1]:.0f}% em {int(smart_cim.index[-1])}, aproximando-se da cobertura total da rede em "
    f"apenas {len(smart_cim)} anos. Esta modernização acelerada é importante para a gestão eficiente da rede "
    f"elétrica, permitindo leituras remotas e deteção mais rápida de falhas ou consumos anómalos."
)

# 5. Consumo de Eletricidade em Baixa Tensão — ranking dos 11 municípios + linha CIM
ultimo_ano_bt = df_bt_1k["ano"].max()
bt_cim = valor_grupo(df_bt_1k, "Lezíria do Tejo", ultimo_ano_bt)
bt_mun = df_bt_1k[(df_bt_1k["nome"]==MUNICIPIO_REF) & (df_bt_1k["ano"]==ultimo_ano_bt)]["valor"].mean()
at_cim = valor_grupo(df_at_1k, "Lezíria do Tejo", ultimo_ano_bt)
at_mun = df_at_1k[(df_at_1k["nome"]==MUNICIPIO_REF) & (df_at_1k["ano"]==ultimo_ano_bt)]["valor"].mean()

bt_dados = df_bt_1k[(df_bt_1k["ano"]==ultimo_ano_bt) & (~df_bt_1k["nome"].isin(["Portugal", "Lezíria do Tejo"]))]
fig, ax = barh_ref_fig(bt_dados["nome"].tolist(), (bt_dados["valor"]/1000).tolist(), bt_cim/1000, "#A9D18E",
                        title=f"Consumo em Baixa Tensão por 1000hab, por Município ({int(ultimo_ano_bt)})", fmt="{:.0f}K")
salvar(fig, "amb_05_consumo_bt_at")

bt_max = bt_dados.sort_values("valor", ascending=False).iloc[0]
narrativas["amb_05"] = (
    f"{bt_max['nome']} tem o maior consumo em Baixa Tensão por 1000hab da CIM ({bt_max['valor']:.0f}), "
    f"tipicamente associado a maior densidade residencial e comercial. Em {MUNICIPIO_REF}, o consumo em "
    f"Alta/Média Tensão (indústria e grandes consumidores) "
    f"{'é superior' if at_mun > bt_mun else 'é inferior'} ao consumo em Baixa Tensão, um indicador do perfil "
    f"económico do município."
)


# 6. Membros Comunidades de Energia (mantém-se)
acc_ultimo_ano = df_acc["ano"].max()
acc_dados = df_acc[df_acc["ano"] == acc_ultimo_ano].sort_values("valor", ascending=True)
fig, ax = barh_fig(acc_dados["nome"].tolist(), acc_dados["valor"].tolist(), "#548235", title=f"N.º de Membros em Comunidades de Energia ({int(acc_ultimo_ano)})")
salvar(fig, "amb_06_comunidades_energia")

mun_lider_acc = acc_dados.iloc[-1]
narrativas["amb_06"] = (
    f"{mun_lider_acc['nome']} lidera a adesão a Comunidades de Energia com {mun_lider_acc['valor']:.0f} membros, "
    f"um modelo emergente de produção e partilha de energia renovável entre cidadãos e empresas locais."
)

print("✓ Ambiente - Energia (6 gráficos)")

# --- Resíduos (mantém-se igual) ---
ultimo_ano_res = df_aterro["ano"].max()
fig, ax = choropleth_fig(df_aterro, "amb_taxa_aterro_pct", ultimo_ano_res, cmap="Greens", title=f"Taxa de Deposição em Aterro ({int(ultimo_ano_res)})")
salvar(fig, "amb_07_mapa_aterro")

aterro_max = df_aterro[df_aterro["ano"]==ultimo_ano_res].sort_values("valor", ascending=False).iloc[0]
narrativas["amb_07"] = (
    f"{aterro_max['nome']} apresenta a maior taxa de deposição em aterro ({aterro_max['valor']:.0f}%), uma "
    f"oportunidade de melhoria através do reforço da recolha seletiva e valorização de resíduos."
)

municipios_res = sorted(df_aterro[df_aterro["nome"] != "Portugal"]["nome"].unique())
aterro_all = [valor_grupo(df_aterro, m, ultimo_ano_res) for m in municipios_res]
valor_all = [valor_grupo(df_valor, m, ultimo_ano_res) for m in municipios_res]
recic_all = [valor_grupo(df_recic, m, ultimo_ano_res) for m in municipios_res]

ordem_r = sorted(range(len(municipios_res)), key=lambda i: -valor_all[i])
labels_r = [municipios_res[i] for i in ordem_r]
aterro_ord = [aterro_all[i] for i in ordem_r]
valor_ord = [valor_all[i] for i in ordem_r]
recic_ord = [recic_all[i] for i in ordem_r]

fig, ax = barh_stacked100_fig(labels_r, {"Valorização": valor_ord, "Aterro": aterro_ord},
                                ["#375623", "#8C6244"], title=f"Destino dos Resíduos, por Município ({int(ultimo_ano_res)})")
salvar(fig, "amb_08_destino_residuos")

aterro_cim = valor_grupo(df_aterro, "Lezíria do Tejo", ultimo_ano_res)
aterro_mun = valor_grupo(df_aterro, MUNICIPIO_REF, ultimo_ano_res)
narrativas["amb_08"] = (
    f"{labels_r[0]} valoriza {valor_ord[0]:.0f}% dos seus resíduos (o melhor resultado da CIM), face a apenas "
    f"{valor_ord[-1]:.0f}% em {labels_r[-1]}, que deposita {aterro_ord[-1]:.0f}% em aterro. Em {MUNICIPIO_REF}, a "
    f"deposição em aterro é de {aterro_mun:.0f}%, {'acima' if aterro_mun > aterro_cim else 'abaixo'} da média da "
    f"CIM ({aterro_cim:.0f}%); da fração valorizada, {recic_ord[labels_r.index(MUNICIPIO_REF)]:.0f}% corresponde "
    f"especificamente a reciclagem multimaterial. Os valores agrupam-se por sistema intermunicipal de gestão de "
    f"resíduos: Chamusca, Golegã e Santarém (RESITEJO) têm taxas de aterro muito semelhantes entre si (~44%), "
    f"tal como os restantes municípios servidos por outro sistema comum, que rondam os 93%. Esta coincidência "
    f"decorre da infraestrutura de tratamento partilhada, não de uma diferença de comportamento de cada câmara "
    f"municipal isoladamente."
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
ax.set_title(f"Taxa de Valorização de Resíduos por Município ({int(ultimo_ano_res)})", fontsize=12, fontweight="bold", pad=12)
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax * 1.2)
salvar(fig, "amb_09_valorizacao_municipio")

narrativas["amb_09"] = (
    f"Há uma disparidade acentuada na taxa de valorização de resíduos entre municípios: "
    f"{municipios_ordem[0]} valoriza {valores_ordem[0]:.0f}% dos seus resíduos, face a apenas "
    f"{valores_ordem[-1]:.0f}% em {municipios_ordem[-1]}."
)

print("✓ Ambiente - Resíduos (3 gráficos)")

# ═══════════════════════════════════════════════════════════════
# MOBILIDADE — Parque Automóvel (ATUALIZADO: desagregado por tipo)
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

# 1. Mapa Registo de Veículos Novos
fig, ax = choropleth_fig(df_registo_total, "mob_registo_total_1000hab", ultimo_ano_mob, cmap="Purples", title=f"Registo de Veículos Novos /1000hab ({int(ultimo_ano_mob)})")
salvar(fig, "mob_01_mapa_veiculos")

reg_max = df_registo_total[(df_registo_total["ano"]==ultimo_ano_mob) & (df_registo_total["nome"] != "Lezíria do Tejo")].sort_values("valor", ascending=False).iloc[0]
reg_min_mob = df_registo_total[(df_registo_total["ano"]==ultimo_ano_mob) & (df_registo_total["nome"] != "Lezíria do Tejo")].sort_values("valor", ascending=True).iloc[0]
narrativas["mob_01"] = (
    f"{reg_max['nome']} regista a maior taxa de registo de veículos novos da CIM em {int(ultimo_ano_mob)} "
    f"({reg_max['valor']:.1f} /1000hab), mais do dobro do valor de {reg_min_mob['nome']} "
    f"({reg_min_mob['valor']:.1f} /1000hab). Esta disparidade pode refletir diferenças no rendimento disponível "
    f"das famílias, mas também na estrutura da frota agrícola e comercial de cada município."
)

# 2. Registo de Veículos Ligeiros — ranking dos 11 municípios + linha CIM
lig_cim = df_lig[(df_lig["nome"]=="Lezíria do Tejo") & (df_lig["ano"]==ultimo_ano_mob)]["valor"].values[0]
lig_mun = df_lig[(df_lig["nome"]==MUNICIPIO_REF) & (df_lig["ano"]==ultimo_ano_mob)]["valor"].values[0]
pes_cim = df_pes[(df_pes["nome"]=="Lezíria do Tejo") & (df_pes["ano"]==ultimo_ano_mob)]["valor"].values[0]
pes_mun = df_pes[(df_pes["nome"]==MUNICIPIO_REF) & (df_pes["ano"]==ultimo_ano_mob)]["valor"].values[0]
tra_cim = df_tra[(df_tra["nome"]=="Lezíria do Tejo") & (df_tra["ano"]==ultimo_ano_mob)]["valor"].values[0]
tra_mun = df_tra[(df_tra["nome"]==MUNICIPIO_REF) & (df_tra["ano"]==ultimo_ano_mob)]["valor"].values[0]

lig_dados = df_lig[(df_lig["ano"]==ultimo_ano_mob) & (~df_lig["nome"].isin(["Portugal", "Lezíria do Tejo"]))]
tra_dados = df_tra[(df_tra["ano"]==ultimo_ano_mob) & (~df_tra["nome"].isin(["Portugal", "Lezíria do Tejo"]))]
fig, ax = barh_ref_fig(lig_dados["nome"].tolist(), lig_dados["valor"].tolist(), lig_cim, "#4B2E83",
                        title=f"Registo de Veículos Ligeiros /1000hab, por Município ({int(ultimo_ano_mob)})", fmt="{:.2f}")
salvar(fig, "mob_02_registos_por_tipo")

tra_max = tra_dados.sort_values("valor", ascending=False).iloc[0]
narrativas["mob_02"] = (
    f"Os veículos ligeiros dominam claramente o registo de veículos novos em toda a CIM (média de {lig_cim:.2f}/1000hab). "
    f"No registo de tratores agrícolas — indicador da vocação rural —, {tra_max['nome']} destaca-se claramente "
    f"({tra_max['valor']:.2f}/1000hab), bastante acima da média da CIM ({tra_cim:.2f}). Em {MUNICIPIO_REF}, o "
    f"registo é de {lig_mun:.2f} ligeiros, {pes_mun:.2f} pesados e {tra_mun:.2f} tratores por 1000hab."
)


# 3. Evolução do peso do registo total (% CIM)
reg_pct_mun = df_registo_total_pct[df_registo_total_pct["nome"]==MUNICIPIO_REF].sort_values("ano")
fig, ax = linha_fig(reg_pct_mun["ano"], reg_pct_mun["valor"], "#8064A2", title=f"Evolução do Registo de Veículos — {MUNICIPIO_REF} (% da CIM)", fmt="{:.1f}%")
salvar(fig, "mob_03_evolucao_veiculos")

narrativas["mob_03"] = (
    f"O peso de {MUNICIPIO_REF} no registo total de veículos da CIM tem oscilado entre "
    f"{reg_pct_mun['valor'].min():.1f}% e {reg_pct_mun['valor'].max():.1f}% ao longo do período analisado, "
    f"terminando em {reg_pct_mun['valor'].iloc[-1]:.1f}% no último ano. Como {MUNICIPIO_REF} é o município mais "
    f"populoso da CIM, seria expectável um peso estável e proporcional à sua dimensão populacional; oscilações "
    f"acentuadas podem indicar campanhas de renovação de frota ou efeitos conjunturais específicos."
)

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

fig, ax = barh_stacked100_fig(labels_c, {"Rápidos/Ultrarrápidos": rap_ord, "Semirrápidos": semi_ord},
                                ["#5C4187", "#B8A2D9"], title=f"Pontos de Carregamento Elétrico, por Município ({int(ultimo_ano_carreg)})")
salvar(fig, "mob_04_pontos_carregamento")

narrativas["mob_04"] = (
    f"A rede de carregamento elétrico varia muito no tipo de pontos entre municípios: {labels_c[0]} tem "
    f"{rap_ord[0]:.0f}% de pontos rápidos/ultrarrápidos, enquanto {labels_c[-1]} não tem nenhum ponto rápido "
    f"(0%, só semirrápidos). Na média da CIM, {semi_cim:.0f}% dos pontos são semirrápidos e apenas {rap_cim:.0f}% "
    f"rápidos/ultrarrápidos. Nota: dado disponível apenas para {int(ultimo_ano_carreg)}, sem série histórica."
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

fig, ax = barh_stacked100_fig(labels_p, {"Privado": priv_ord, "Público": pub_ord},
                                ["#5C4187", "#B8A2D9"], title=f"Carregamento Privado vs Público, por Município ({int(ultimo_ano_carreg)})")
salvar(fig, "mob_05_privado_publico")

narrativas["mob_05"] = (
    f"O acesso à rede de carregamento é muito desigual entre municípios: em {labels_p[0]}, {priv_ord[0]:.0f}% dos "
    f"pontos são de acesso privado, enquanto em {labels_p[-1]} praticamente todos os pontos ({pub_ord[-1]:.0f}%) "
    f"são de acesso público — municípios onde o acesso é maioritariamente privado podem estar a limitar a "
    f"disponibilidade real de carregamento para quem não tem infraestrutura própria."
)

print("✓ Mobilidade - Parque Automóvel (5 gráficos)")

# ═══════════════════════════════════════════════════════════════
# MODOS DE VIDA — Saúde (ATUALIZADO: profissionais individuais)
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

# 1. Mapa Habitantes por Médico
fig, ax = choropleth_fig(df_hm, "mdv_hab_medico", ultimo_ano_saude, cmap="Oranges", title=f"Habitantes por Médico ({int(ultimo_ano_saude)})")
salvar(fig, "mdv_01_mapa_hab_medico")

hm_max = df_hm[df_hm["ano"]==ultimo_ano_saude].sort_values("valor", ascending=False).iloc[0]
hm_min = df_hm[df_hm["ano"]==ultimo_ano_saude].sort_values("valor", ascending=True).iloc[0]
narrativas["mdv_01"] = (
    f"{hm_max['nome']} tem o rácio mais desfavorável de habitantes por médico ({hm_max['valor']:.0f} "
    f"hab./médico), enquanto {hm_min['nome']} apresenta a melhor cobertura ({hm_min['valor']:.0f}) — uma "
    f"diferença de mais de {hm_max['valor']/hm_min['valor']:.1f} vezes entre os dois extremos. Esta assimetria "
    f"no acesso a cuidados médicos primários é um dos indicadores mais relevantes para políticas de coesão "
    f"territorial em saúde dentro da CIM."
)

# 2. Profissionais de saúde — números absolutos, grelha de rankings dos 11 municípios
# (a taxa por 1000 hab. já está coberta para médicos/farmacêuticos nos gráficos mdv_01/mdv_04;
# aqui mostra-se antes a dimensão real da equipa em cada município)
def dados_absolutos(df, ano):
    d = df[(df["ano"] == ano) & (~df["nome"].isin(["Portugal", "Lezíria do Tejo"]))].set_index("nome")["valor"]
    return d.dropna()

medicos_abs = dados_absolutos(df_medicos, ultimo_ano_saude)
farm_abs = dados_absolutos(df_farmaceuticos, ultimo_ano_saude)
dent_abs = dados_absolutos(df_dentistas, ultimo_ano_saude)
enf_abs = dados_absolutos(df_enfermeiros, ultimo_ano_saude)

paineis_saude = [
    dict(labels=medicos_abs.index.tolist(), valores=medicos_abs.values.tolist(),
         cim_valor=medicos_abs.median(), color="#C55A11", title="N.º de Médicos", fmt="{:.0f}"),
    dict(labels=farm_abs.index.tolist(), valores=farm_abs.values.tolist(),
         cim_valor=farm_abs.median(), color="#C55A11", title="N.º de Farmacêuticos", fmt="{:.0f}"),
    dict(labels=dent_abs.index.tolist(), valores=dent_abs.values.tolist(),
         cim_valor=dent_abs.median(), color="#C55A11", title="N.º de Dentistas", fmt="{:.0f}"),
    dict(labels=enf_abs.index.tolist(), valores=enf_abs.values.tolist(),
         cim_valor=enf_abs.median(), color="#C55A11", title="N.º de Enfermeiros", fmt="{:.0f}"),
]
fig, axes = barh_ref_grid_fig(paineis_saude, ncols=2, figsize=(12, 9), mostrar_linha=False)
salvar(fig, "mdv_02_kpis_profissionais")

medicos_mun = df_medicos[(df_medicos["nome"]==MUNICIPIO_REF) & (df_medicos["ano"]==ultimo_ano_saude)]["valor"].values[0]
farm_mun = df_farmaceuticos[(df_farmaceuticos["nome"]==MUNICIPIO_REF) & (df_farmaceuticos["ano"]==ultimo_ano_saude)]["valor"].values[0]
dent_mun = df_dentistas[(df_dentistas["nome"]==MUNICIPIO_REF) & (df_dentistas["ano"]==ultimo_ano_saude)]["valor"].values[0]
enf_mun = df_enfermeiros[(df_enfermeiros["nome"]==MUNICIPIO_REF) & (df_enfermeiros["ano"]==ultimo_ano_saude)]["valor"].values[0]

narrativas["mdv_02"] = (
    f"Em número absoluto de profissionais, {MUNICIPIO_REF} lidera nas quatro categorias — natural, já que é o "
    f"município mais populoso da CIM — com {medicos_mun:.0f} médicos, {enf_mun:.0f} enfermeiros, {farm_mun:.0f} "
    f"farmacêuticos e {dent_mun:.0f} dentistas em {int(ultimo_ano_saude)}. No extremo oposto, {medicos_abs.idxmin()} "
    f"tem apenas {medicos_abs.min():.0f} médicos e {enf_abs.idxmin()} tem {enf_abs.min():.0f} enfermeiros. Esta "
    f"leitura mostra a dimensão real de cada equipa de saúde local; para uma comparação da cobertura ajustada à "
    f"população, ver os rácios de habitantes por médico/farmacêutico mais à frente nesta secção."
)

# 3. Utentes inscritos CSP na CIM (KPI + evolução)
ultimo_ano_utentes = df_utentes["ano"].max()
utentes_cim = df_utentes[df_utentes["ano"]==ultimo_ano_utentes]["valor"].iloc[0]
fig, axes = kpis_row_fig([
    (f"{utentes_cim:,.0f}".replace(",", " "), "Utentes Inscritos no CSP\nna CIM", str(int(ultimo_ano_utentes)), "#C55A11"),
])
salvar(fig, "mdv_03_kpi_utentes")

narrativas["mdv_03"] = so_milhares(
    f"A CIM tinha {utentes_cim:,.0f} utentes inscritos nos Cuidados de Saúde Primários (CSP) em "
    f"{int(ultimo_ano_utentes)}, praticamente a totalidade da população residente."
)


# 4. Habitantes por Médico e por Farmacêutico — ranking dos 11 municípios + linha CIM
hm_cim = valor_grupo(df_hm, "Lezíria do Tejo", ultimo_ano_saude)
hf_cim = valor_grupo(df_hf, "Lezíria do Tejo", ultimo_ano_saude)
hm_dados = df_hm[(df_hm["ano"]==ultimo_ano_saude) & (~df_hm["nome"].isin(["Portugal", "Lezíria do Tejo"]))].sort_values("nome")
hf_dados = df_hf[(df_hf["ano"]==ultimo_ano_saude) & (~df_hf["nome"].isin(["Portugal", "Lezíria do Tejo"]))].sort_values("nome")
hm_mun = df_hm[(df_hm["nome"]==MUNICIPIO_REF) & (df_hm["ano"]==ultimo_ano_saude)]["valor"].values[0]
hf_mun = df_hf[(df_hf["nome"]==MUNICIPIO_REF) & (df_hf["ano"]==ultimo_ano_saude)]["valor"].values[0]

paineis_hab = [
    dict(labels=hm_dados["nome"].tolist(), valores=hm_dados["valor"].tolist(), cim_valor=hm_cim,
         color="#C55A11", title="Habitantes por Médico", fmt="{:.0f}", cim_label="CIM"),
    dict(labels=hf_dados["nome"].tolist(), valores=hf_dados["valor"].tolist(), cim_valor=hf_cim,
         color="#C55A11", title="Habitantes por Farmacêutico", fmt="{:.0f}", cim_label="CIM"),
]
fig, axes = barh_ref_grid_fig(paineis_hab, ncols=2, figsize=(12, 5.6))
salvar(fig, "mdv_04_hab_farmaceuticos_medicos")

narrativas["mdv_04"] = (
    f"Em {MUNICIPIO_REF}, há {hm_mun:.0f} habitantes por médico e {hf_mun:.0f} habitantes por farmacêutico "
    f"(média da CIM: {hm_cim:.0f} e {hf_cim:.0f}, respetivamente). A disparidade entre municípios é grande: "
    f"{hm_dados.sort_values('valor').iloc[0]['nome']} tem a melhor cobertura médica "
    f"({hm_dados.sort_values('valor').iloc[0]['valor']:.0f} hab./médico), face a "
    f"{hm_dados.sort_values('valor').iloc[-1]['nome']} ({hm_dados.sort_values('valor').iloc[-1]['valor']:.0f}). "
    f"Na cobertura farmacêutica, {hf_dados.sort_values('valor').iloc[0]['nome']} lidera "
    f"({hf_dados.sort_values('valor').iloc[0]['valor']:.0f} hab./farmacêutico), face a "
    f"{hf_dados.sort_values('valor').iloc[-1]['nome']} ({hf_dados.sort_values('valor').iloc[-1]['valor']:.0f})."
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
narrativas["mdv_05"] = (
    f"O número de consultas totais nos CSP {'aumentou' if var_consultas > 0 else 'diminuiu'} {abs(var_consultas):.1f}% "
    f"entre {int(consultas_total_cim.index[0])} e {int(consultas_total_cim.index[-1])}. A diferença entre consultas "
    f"totais e presenciais reflete o peso crescente da telemedicina e do atendimento não presencial."
)

print("✓ Modos de Vida - Saúde (5 gráficos)")

# ═══════════════════════════════════════════════════════════════
# MODOS DE VIDA — Segurança (ATUALIZADO: criminalidade desagregada)
# ═══════════════════════════════════════════════════════════════
df_acid = mdv[mdv["metrica_codigo"] == "mdv_acidentes_vitimas_1000hab"]
df_feridos = mdv[mdv["metrica_codigo"] == "mdv_feridos_acidentes"]
df_mortos = mdv[mdv["metrica_codigo"] == "mdv_mortos_acidentes"]
df_crim_total = mdv[mdv["metrica_codigo"] == "mdv_criminalidade_total"]
df_crim_patrim = mdv[mdv["metrica_codigo"] == "mdv_criminalidade_patrimonio"]
df_crim_integ = mdv[mdv["metrica_codigo"] == "mdv_criminalidade_integridade_fisica"]

ultimo_ano_seg = df_acid["ano"].max()

# 1. Acidentes de Viação (dual axis) — mantém-se
feridos_mun = df_feridos[(df_feridos["nome"] != "Portugal") & (df_feridos["ano"]==ultimo_ano_seg)].groupby("nome")["valor"].sum().sort_values(ascending=False)
fig, ax1 = plt.subplots(figsize=(8, 4.6))
ax1b = ax1.twinx()
ax1.bar(feridos_mun.index, feridos_mun.values, color="#F4B183", label="Feridos")
mortos_mun = df_mortos[(df_mortos["nome"] != "Portugal") & (df_mortos["ano"]==ultimo_ano_seg)].set_index("nome")["valor"].reindex(feridos_mun.index).fillna(0)
ax1b.plot(feridos_mun.index, mortos_mun.values, color="#C00000", marker="o", linewidth=2)
ax1.set_xticklabels(feridos_mun.index, rotation=40, ha="right", fontsize=9)
ax1.set_title(f"Acidentes de Viação com Vítimas ({int(ultimo_ano_seg)})", fontsize=12, fontweight="bold", pad=12)
ax1.set_ylabel("N.º de Feridos", fontsize=9.5)
ax1b.set_ylabel("N.º de Mortos", fontsize=9.5)
salvar(fig, "mdv_06_acidentes_viacao")

mun_mais_feridos = feridos_mun.index[0]
narrativas["mdv_06"] = (
    f"{mun_mais_feridos} regista o maior número absoluto de feridos em acidentes de viação em {int(ultimo_ano_seg)} "
    f"({feridos_mun.iloc[0]:.0f}), consistente com ser também o município mais populoso da CIM."
)

# 2. Taxa de Criminalidade Total — evolução no Município
crim_total_mun = df_crim_total[df_crim_total["nome"]==MUNICIPIO_REF].sort_values("ano")
fig, ax = linha_fig(crim_total_mun["ano"], crim_total_mun["valor"], "#C55A11", title=f"Taxa de Criminalidade Total — {MUNICIPIO_REF} (‰)", fmt="{:.0f}")
salvar(fig, "mdv_07_criminalidade_evolucao")

var_crim_mun = crim_total_mun["valor"].iloc[-1] - crim_total_mun["valor"].iloc[0]
crim_cim_evol = evolucao_cim(df_crim_total)
narrativas["mdv_07"] = (
    f"Em {MUNICIPIO_REF}, a taxa de criminalidade total {'subiu' if var_crim_mun > 0 else 'desceu'} "
    f"{abs(var_crim_mun):.1f} pontos entre {int(crim_total_mun['ano'].iloc[0])} e {int(crim_total_mun['ano'].iloc[-1])}, "
    f"terminando em {crim_total_mun['valor'].iloc[-1]:.1f}‰, um valor "
    f"{'acima' if crim_total_mun['valor'].iloc[-1] > crim_cim_evol['valor'].iloc[-1] else 'abaixo'} da média da CIM "
    f"nesse mesmo ano ({crim_cim_evol['valor'].iloc[-1]:.1f}‰)."
)

# 3. Criminalidade Total — ranking dos 11 municípios + linha CIM
patrim_cim = valor_grupo(df_crim_patrim, "Lezíria do Tejo", ultimo_ano_seg)
patrim_mun = valor_grupo(df_crim_patrim, MUNICIPIO_REF, ultimo_ano_seg)
total_cim = valor_grupo(df_crim_total, "Lezíria do Tejo", ultimo_ano_seg)
integ_cim = valor_grupo(df_crim_integ, "Lezíria do Tejo", ultimo_ano_seg)
integ_mun = valor_grupo(df_crim_integ, MUNICIPIO_REF, ultimo_ano_seg)

crim_total_dados = df_crim_total[(df_crim_total["ano"]==ultimo_ano_seg) & (~df_crim_total["nome"].isin(["Portugal", "Lezíria do Tejo"]))]
fig, ax = barh_ref_fig(crim_total_dados["nome"].tolist(), crim_total_dados["valor"].tolist(), total_cim, "#C55A11",
                        title=f"Taxa de Criminalidade Total por Município ({int(ultimo_ano_seg)}, ‰)", fmt="{:.1f}")
salvar(fig, "mdv_08_criminalidade_tipo")

crim_max = crim_total_dados.sort_values("valor", ascending=False).iloc[0]
crim_min = crim_total_dados.sort_values("valor", ascending=True).iloc[0]
narrativas["mdv_08"] = (
    f"{crim_max['nome']} tem a taxa de criminalidade mais alta da CIM ({crim_max['valor']:.1f}‰), face a "
    f"{crim_min['nome']} ({crim_min['valor']:.1f}‰) — quase o dobro de diferença. Em termos de composição, a "
    f"criminalidade patrimonial domina o total tanto na CIM ({patrim_cim:.1f}‰) como em {MUNICIPIO_REF} "
    f"({patrim_mun:.1f}‰), face à criminalidade contra a integridade física, bastante menor ({integ_cim:.1f}‰ "
    f"na CIM, {integ_mun:.1f}‰ em {MUNICIPIO_REF})."
)

print("✓ Modos de Vida - Segurança (3 gráficos)")

# ═══════════════════════════════════════════════════════════════
# MODOS DE VIDA — Educação (mantém-se, dados atualizados)
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

fig, ax = choropleth_fig(df_sem_esc, "mdv_sem_escolaridade_pct", ultimo_ano_edu, cmap="Oranges", title=f"Pop. Sem Nível de Escolaridade ({int(ultimo_ano_edu)})")
salvar(fig, "mdv_09_mapa_sem_escolaridade")

sem_esc_max = df_sem_esc[df_sem_esc["ano"]==ultimo_ano_edu].sort_values("valor", ascending=False).iloc[0]
sem_esc_min = df_sem_esc[df_sem_esc["ano"]==ultimo_ano_edu].sort_values("valor", ascending=True).iloc[0]
narrativas["mdv_09"] = (
    f"{sem_esc_max['nome']} tem a maior proporção de população sem nenhum nível de escolaridade "
    f"({sem_esc_max['valor']:.0f}%), segundo os Censos {int(ultimo_ano_edu)}, face a {sem_esc_min['valor']:.0f}% "
    f"em {sem_esc_min['nome']}. Este indicador tende a estar correlacionado com o envelhecimento populacional e "
    f"com a ruralidade, sendo relevante para orientar programas de educação de adultos e literacia digital."
)

def kpi_ensino(df, ano, municipio):
    return int(df[(df["nome"]==municipio) & (df["ano"]==ano)]["valor"].sum())

niveis = [
    ("Pré-Escolar", df_pre), ("1.º Ciclo", df_c1), ("2.º Ciclo", df_c2),
    ("3.º Ciclo", df_c3), ("Secundário", df_sec), ("Ens. Superior", df_sup),
]
valores_niveis = [kpi_ensino(df, ultimo_ano_edu_serie, MUNICIPIO_REF) for _, df in niveis]

# Distribuição por nível de ensino nos 11 municípios — barras empilhadas a 100%
municipios_edu = sorted(df_pre[(df_pre["ano"]==ultimo_ano_edu_serie) & (~df_pre["nome"].isin(["Portugal", "Lezíria do Tejo"]))]["nome"].unique())
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
                               title=f"Distribuição de Alunos Matriculados por Nível de Ensino ({int(ultimo_ano_edu_serie)})",
                               figsize=(9, 5.8))
salvar(fig, "mdv_10_kpis_niveis_ensino")

nivel_maior = niveis[valores_niveis.index(max(valores_niveis))][0]
pre_pct_range = tabela_niveis_mun["Pré-Escolar"]
mun_mais_pre = municipios_edu[pre_pct_range.index(max(pre_pct_range))]
mun_menos_pre = municipios_edu[pre_pct_range.index(min(pre_pct_range))]
narrativas["mdv_10"] = (
    f"Em {MUNICIPIO_REF}, o {nivel_maior} é o nível de ensino com mais alunos matriculados "
    f"({max(valores_niveis)}), à frente do 1.º Ciclo ({valores_niveis[niveis.index(next(n for n in niveis if n[0]=='1.º Ciclo'))]}). "
    f"A composição por nível varia entre municípios: o peso do Pré-Escolar no total de matriculados vai de "
    f"{min(pre_pct_range):.0f}% em {mun_menos_pre} a {max(pre_pct_range):.0f}% em {mun_mais_pre}, uma diferença que "
    f"pode refletir perfis demográficos distintos (municípios mais jovens vs. mais envelhecidos). Esta distribuição "
    f"por nível de ensino ajuda a antecipar necessidades futuras de vagas e recursos docentes à medida que as "
    f"coortes de alunos avançam entre ciclos."
)

trans_h_v = df_trans_h[(df_trans_h["nome"]==MUNICIPIO_REF) & (df_trans_h["ano"]==ultimo_ano_edu_serie)]["valor"].mean()
trans_m_v = df_trans_m[(df_trans_m["nome"]==MUNICIPIO_REF) & (df_trans_m["ano"]==ultimo_ano_edu_serie)]["valor"].mean()
fig, axes = donuts_row_fig([
    (trans_h_v, "#E8B33D", f"Transição/Retenção\nHomens ({int(ultimo_ano_edu_serie)})"),
    (trans_m_v, "#E8B33D", f"Transição/Retenção\nMulheres ({int(ultimo_ano_edu_serie)})"),
])
salvar(fig, "mdv_11_transicao_retencao")

narrativas["mdv_11"] = (
    f"A taxa de transição/conclusão no ensino básico em {MUNICIPIO_REF} é de {trans_h_v:.1f}% para os rapazes e "
    f"{trans_m_v:.1f}% para as raparigas — as raparigas têm uma taxa de sucesso "
    f"{trans_m_v - trans_h_v:.1f} pontos percentuais {'superior' if trans_m_v > trans_h_v else 'inferior'}, um "
    f"padrão consistente com estatísticas nacionais que apontam para melhor desempenho escolar médio das "
    f"raparigas no ensino básico."
)

print("✓ Modos de Vida - Educação (3 gráficos)")

# ═══════════════════════════════════════════════════════════════
# MODOS DE VIDA — Turismo (mantém-se, dados atualizados)
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

narrativas["mdv_12"] = (
    f"Metade dos alojamentos turísticos da CIM estavam vagos em {int(ultimo_ano_aloj)} ({vagos_cim:.0f}%), e "
    f"{sazonal_cim:.0f}% do uso registado é sazonal — dois indicadores que, em conjunto, sugerem uma oferta "
    f"turística ainda pouco otimizada, com potencial de crescimento fora da época alta se houver investimento "
    f"em animação e atratividade durante todo o ano."
)

fig, ax = choropleth_fig(df_dorm, "mdv_dormidas_100hab", ultimo_ano_tur, cmap="Oranges", title=f"Dormidas /100hab ({int(ultimo_ano_tur)})")
salvar(fig, "mdv_13_mapa_dormidas")

dorm_max = df_dorm[df_dorm["ano"]==ultimo_ano_tur].sort_values("valor", ascending=False).iloc[0]
dorm_cim_media = df_dorm[(df_dorm["ano"]==ultimo_ano_tur) & (df_dorm["nome"] != "Portugal")]["valor"].mean()
narrativas["mdv_13"] = (
    f"{dorm_max['nome']} concentra o maior número de dormidas por habitante da CIM em {int(ultimo_ano_tur)} "
    f"({dorm_max['valor']:.0f} /100hab), muito acima da média da CIM ({dorm_cim_media:.0f} /100hab). Esta "
    f"concentração espacial do turismo é comum em territórios com um polo histórico ou natural de destaque, e "
    f"sugere oportunidade para dispersar fluxos turísticos pelos restantes municípios."
)

vagos_mun = df_vagos[(df_vagos["ano"]==ultimo_ano_aloj) & (df_vagos["nome"] != "Portugal")].set_index("nome")["valor"]
fig, ax = barh_fig(vagos_mun.index.tolist(), vagos_mun.values.tolist(), "#F4B183", title=f"Alojamentos Vagos por Município — {int(ultimo_ano_aloj)}")
salvar(fig, "mdv_14_alojamentos_vagos")

narrativas["mdv_14"] = (
    f"A taxa de alojamentos vagos varia entre {vagos_mun.min():.0f}% e {vagos_mun.max():.0f}% consoante o município: "
    f"{vagos_mun.idxmin()} tem a menor proporção de alojamentos vagos ({vagos_mun.min():.0f}%), sinal de maior "
    f"ocupação turística, enquanto {vagos_mun.idxmax()} tem mais de metade dos alojamentos sem uso "
    f"({vagos_mun.max():.0f}%), o que pode indicar excesso de oferta face à procura local."
)

print("✓ Modos de Vida - Turismo (3 gráficos)")

# ═══════════════════════════════════════════════════════════════
# ECONOMIA — Emprego (mantém-se)
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
    (f"{tconta_cim:.2f}%", "Taxa de Trabalho\npor Conta Própria", str(int(ultimo_ano_emp)), "#C0504D"),
    (f"{tempreg_cim:.2f}%", "Taxa de\nEmpregadores", str(int(ultimo_ano_emp)), "#C0504D"),
])
salvar(fig, "eco_01_kpis_emprego")

narrativas["eco_01"] = (
    f"Na CIM, {tconta_cim:.1f}% da população ativa trabalha por conta própria e apenas {tempreg_cim:.1f}% é "
    f"empregadora — estrutura marcada pelo predomínio do trabalho por conta de outrem."
)

fig, ax = choropleth_fig(df_temprego, "eco_taxa_emprego_pct", ultimo_ano_emp, cmap="Reds", title=f"Taxa de Emprego ({int(ultimo_ano_emp)})")
salvar(fig, "eco_02_mapa_taxa_emprego")

temp_max = df_temprego[df_temprego["ano"]==ultimo_ano_emp].sort_values("valor", ascending=False).iloc[0]
narrativas["eco_02"] = (
    f"{temp_max['nome']} regista a maior taxa de emprego da CIM ({temp_max['valor']:.1f}%), segundo o Censos "
    f"{int(ultimo_ano_emp)}, {temp_max['valor'] - df_temprego[(df_temprego['ano']==ultimo_ano_emp) & (df_temprego['nome']!='Portugal')]['valor'].min():.1f} "
    f"pontos percentuais acima do município com a taxa mais baixa. Esta métrica reflete a proporção da "
    f"população em idade ativa efetivamente empregada, um indicador estrutural que muda pouco entre censos."
)

municipios_est = sorted(df_est_agri[df_est_agri["nome"] != "Portugal"]["nome"].unique())
agri_v_all = [valor_grupo(df_est_agri, m) for m in municipios_est]
ind_v_all = [valor_grupo(df_est_ind, m) for m in municipios_est]
serv_v_all = [valor_grupo(df_est_serv, m) for m in municipios_est]
agri_v = [valor_grupo(df_est_agri, "Lezíria do Tejo"), valor_grupo(df_est_agri, MUNICIPIO_REF)]
ind_v = [valor_grupo(df_est_ind, "Lezíria do Tejo"), valor_grupo(df_est_ind, MUNICIPIO_REF)]
serv_v = [valor_grupo(df_est_serv, "Lezíria do Tejo"), valor_grupo(df_est_serv, MUNICIPIO_REF)]

# ordenar municípios por peso decrescente de Serviços
ordem = sorted(range(len(municipios_est)), key=lambda i: -serv_v_all[i])
labels_ord = [municipios_est[i] for i in ordem]
agri_ord = [agri_v_all[i] for i in ordem]
ind_ord = [ind_v_all[i] for i in ordem]
serv_ord = [serv_v_all[i] for i in ordem]

fig, ax = barh_stacked100_fig(labels_ord, {"Agricultura": agri_ord, "Indústria": ind_ord, "Serviços": serv_ord},
                                ["#F2C4C1", "#C0504D", "#772C2A"], title="Estrutura Setorial do Emprego, por Município")
salvar(fig, "eco_03_estrutura_setorial_emprego")

narrativas["eco_03"] = (
    f"O setor dos Serviços domina a estrutura de emprego em quase todos os municípios da CIM (média de "
    f"{serv_v[0]:.0f}%), mas com pesos muito diferentes: {labels_ord[0]} tem {serv_ord[0]:.0f}% do emprego em "
    f"Serviços, face a apenas {serv_ord[-1]:.0f}% em {labels_ord[-1]}, onde a Agricultura ({agri_ord[-1]:.0f}%) "
    f"ou Indústria ({ind_ord[-1]:.0f}%) têm peso mais relevante."
)

nasc_mun = df_nasc[df_nasc["nome"]==MUNICIPIO_REF].sort_values("ano")
mort_mun = df_mort[df_mort["nome"]==MUNICIPIO_REF].sort_values("ano")

municipios_nasc = sorted(df_nasc[df_nasc["nome"] != "Portugal"]["nome"].unique())
municipios_nasc = [m for m in municipios_nasc if m != MUNICIPIO_REF] + [MUNICIPIO_REF]
dados_nasc = {}
for m in municipios_nasc:
    sub = df_nasc[df_nasc["nome"]==m].sort_values("ano")
    dados_nasc[m] = (sub["ano"].tolist(), sub["valor"].tolist())

fig, axes = small_multiples_fig(dados_nasc, "Empresas Nascidas por Município (evolução anual)",
                                  ylabel="N.º de empresas", color="#C0504D", destacar=MUNICIPIO_REF, figsize=(11, 8))
salvar(fig, "eco_04_dinamica_empresarial")

saldo_empresas = nasc_mun["valor"].iloc[-1] - mort_mun["valor"].iloc[-1]
crescimentos = {m: (dados_nasc[m][1][-1] - dados_nasc[m][1][0]) for m in dados_nasc}
mun_maior_cresc = max(crescimentos, key=crescimentos.get)
narrativas["eco_04"] = (
    f"O padrão de criação de empresas varia significativamente entre municípios. {mun_maior_cresc} foi o que mais "
    f"aumentou o número de empresas nascidas por ano ao longo da série. Em {MUNICIPIO_REF}, houve "
    f"{nasc_mun['valor'].iloc[-1]:.0f} empresas nascidas e {mort_mun['valor'].iloc[-1]:.0f} cessadas em "
    f"{int(nasc_mun['ano'].iloc[-1])} — um saldo empresarial "
    f"{'positivo' if saldo_empresas > 0 else 'negativo'} de {abs(saldo_empresas):.0f} empresas."
)

print("✓ Economia - Emprego (4 gráficos)")

# ═══════════════════════════════════════════════════════════════
# ECONOMIA — Rendimento (ATUALIZADO: Estrutura VN + Poder Compra em linha)
# ═══════════════════════════════════════════════════════════════
df_rend = eco[eco["metrica_codigo"] == "eco_rendimento_bruto_per_capita_e"]
df_irs = eco[eco["metrica_codigo"] == "eco_irs_per_capita_e"]
df_vn = eco[eco["metrica_codigo"] == "eco_vn_per_capita_e"]
df_ipc = eco[eco["metrica_codigo"] == "eco_ipc_base100"]
df_vn_agri = eco[eco["metrica_codigo"] == "eco_estrutura_vn_agricultura_pct"]
df_vn_ind = eco[eco["metrica_codigo"] == "eco_estrutura_vn_industria_pct"]
df_vn_serv = eco[eco["metrica_codigo"] == "eco_estrutura_vn_servicos_pct"]

ultimo_ano_rend = df_rend["ano"].max()
fig, ax = choropleth_fig(df_rend, "eco_rendimento_bruto_per_capita_e", ultimo_ano_rend, cmap="Reds", title=f"Rendimento Bruto per Capita — € ({int(ultimo_ano_rend)})")
salvar(fig, "eco_05_mapa_rendimento")

rend_max = df_rend[(df_rend["ano"]==ultimo_ano_rend) & (df_rend["nome"] != "Portugal")].sort_values("valor", ascending=False).iloc[0]
rend_min = df_rend[(df_rend["ano"]==ultimo_ano_rend) & (df_rend["nome"] != "Portugal")].sort_values("valor", ascending=True).iloc[0]
rend_cim_media = valor_grupo(df_rend, "Lezíria do Tejo", ultimo_ano_rend)
rend_ord_desc = df_rend[(df_rend["ano"]==ultimo_ano_rend) & (~df_rend["nome"].isin(["Portugal", "Lezíria do Tejo"]))].sort_values("valor", ascending=False)
n_acima = int((rend_ord_desc["valor"] > rend_cim_media).sum())
narrativas["eco_05"] = so_milhares(
    f"{rend_max['nome']} tem o maior rendimento bruto per capita da CIM ({rend_max['valor']:,.0f}€) em "
    f"{int(ultimo_ano_rend)}, face aos {rend_min['valor']:,.0f}€ de {rend_min['nome']}, o valor mais baixo — uma "
    f"diferença de {rend_max['valor'] - rend_min['valor']:,.0f}€. A média da CIM é de {rend_cim_media:,.0f}€, "
    f"com {n_acima} dos 11 municípios acima desse valor e os restantes {11 - n_acima} abaixo."
)


rend_ultimo = df_rend[(df_rend["ano"]==ultimo_ano_rend) & (df_rend["nome"] != "Portugal")].set_index("nome")["valor"]
irs_ultimo = df_irs[(df_irs["ano"]==ultimo_ano_rend) & (df_irs["nome"] != "Portugal")].set_index("nome")["valor"]
comuns = rend_ultimo.index.intersection(irs_ultimo.index)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(rend_ultimo[comuns], irs_ultimo[comuns], s=200, color="#D99795", edgecolor="#C0504D", alpha=0.85)
for m in comuns:
    ax.annotate(m, (rend_ultimo[m], irs_ultimo[m]), fontsize=8.5, ha="center", va="center")
ax.set_title(f"Rendimento per Capita vs IRS per Capita ({int(ultimo_ano_rend)})", fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Rendimento per capita (€)", fontsize=9.5)
ax.set_ylabel("IRS per capita (€)", fontsize=9.5)
salvar(fig, "eco_06_scatter_rendimento_irs")

narrativas["eco_06"] = (
    "Existe uma relação globalmente positiva entre rendimento per capita e IRS per capita entre os municípios "
    "da CIM — municípios com maior rendimento tendem a contribuir proporcionalmente mais em sede de IRS."
)

vn_ultimo = df_vn[df_vn["ano"]==df_vn["ano"].max()].sort_values("valor", ascending=False)
fig, ax = barh_fig(vn_ultimo["nome"].tolist(), vn_ultimo["valor"].tolist(), "#D99795", title=f"Volume de Negócios (€/hab) — {int(df_vn['ano'].max())}")
salvar(fig, "eco_07_volume_negocios")

vn_max = vn_ultimo.iloc[0]
narrativas["eco_07"] = so_milhares(
    f"{vn_max['nome']} destaca-se com um volume de negócios per capita muito superior ao dos restantes "
    f"municípios ({vn_max['valor']:,.0f}€/hab). Nota: métrica disponível apenas para 2022-2024 (limitação da fonte INE)."
)


# Estrutura do Volume de Negócios por Setor — 100% empilhada, 11 municípios
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

fig, ax = barh_stacked100_fig(labels_vn, {"Agricultura": agri_vn_ord, "Indústria": ind_vn_ord, "Serviços": serv_vn_ord},
                                ["#F2C4C1", "#C0504D", "#772C2A"], title=f"Estrutura do Volume de Negócios por Setor, por Município ({int(ultimo_ano_vn_est)})")
salvar(fig, "eco_08_estrutura_vn")

vn_ind_cim = valor_grupo(df_vn_ind, "Lezíria do Tejo", ultimo_ano_vn_est)
vn_ind_mun = valor_grupo(df_vn_ind, MUNICIPIO_REF, ultimo_ano_vn_est)
narrativas["eco_08"] = (
    f"Ao contrário da estrutura de emprego, a estrutura do Volume de Negócios revela um peso industrial muito "
    f"elevado em {labels_vn[0]} ({ind_vn_ord[0]:.0f}%) — o oposto do que se vê nos municípios liderados por "
    f"Serviços. Em {MUNICIPIO_REF}, a Indústria pesa {vn_ind_mun:.0f}% do Volume de Negócios, face a "
    f"{vn_ind_cim:.0f}% na média da CIM, sugerindo a presença de empresas de maior faturação no setor industrial local."
)

# Poder de Compra — small multiples por município
anos_ipc = sorted(df_ipc[df_ipc["nome"] != "Portugal"]["ano"].unique())
ipc_mun = df_ipc[df_ipc["nome"]==MUNICIPIO_REF].sort_values("ano")
ipc_cim = evolucao_cim(df_ipc)

municipios_ipc = sorted(df_ipc[~df_ipc["nome"].isin(["Portugal", "Lezíria do Tejo"])]["nome"].unique())
municipios_ipc = [m for m in municipios_ipc if m != MUNICIPIO_REF] + [MUNICIPIO_REF, "Lezíria do Tejo"]
dados_ipc = {}
for m in municipios_ipc:
    sub = df_ipc[df_ipc["nome"]==m].sort_values("ano")
    dados_ipc[m] = (sub["ano"].tolist(), sub["valor"].tolist())

fig, axes = small_multiples_fig(dados_ipc, "Poder de Compra por Município (Índice per capita, base 100 = PT)",
                                  ylabel="Índice per capita", color="#7B1E3A", destacar="Lezíria do Tejo", figsize=(11, 8))
salvar(fig, "eco_09_poder_compra")

ipc_max_mun = max(dados_ipc, key=lambda m: dados_ipc[m][1][-1])
ipc_min_mun = min((m for m in dados_ipc if m != "Lezíria do Tejo"), key=lambda m: dados_ipc[m][1][-1])
narrativas["eco_09"] = (
    f"Em {int(ipc_mun['ano'].max())}, o índice de poder de compra per capita de {MUNICIPIO_REF} "
    f"({ipc_mun['valor'].iloc[-1]:.0f}) está "
    f"{'acima' if ipc_mun['valor'].iloc[-1] > ipc_cim['valor'].iloc[-1] else 'abaixo'} da média da CIM "
    f"({ipc_cim['valor'].iloc[-1]:.0f}). {ipc_max_mun} lidera com {dados_ipc[ipc_max_mun][1][-1]:.0f}, face a "
    f"{ipc_min_mun} com {dados_ipc[ipc_min_mun][1][-1]:.0f} — o município mais bem posicionado tem quase o dobro "
    f"do poder de compra do menos bem posicionado. Nota: esta métrica é bienal, não anual."
)

print("✓ Economia - Rendimento (5 gráficos)")

with open(f"{OUT}/narrativas.json", "w", encoding="utf-8") as f:
    json.dump(narrativas, f, ensure_ascii=False, indent=2)

print(f"\n✓✓✓ TOTAL: {len(narrativas)} gráficos com narrativa gerados em {OUT}/")
