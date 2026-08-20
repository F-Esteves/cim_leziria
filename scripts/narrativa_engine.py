import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config" / "indicadores_metas.json"

# ──────────────────────────────────────────────────────────────
# Banco de frases-modelo (2-3 variantes por situação, para evitar
# repetição robótica no texto final). As chaves coincidem com
# os valores de `situacao` devolvidos por `classificar_variacao()`.
# ──────────────────────────────────────────────────────────────
FRASES = {
    "aumento": [
        "registou um aumento de {variacao:.1f}%",
        "cresceu {variacao:.1f}%",
        "teve uma subida de {variacao:.1f}%",
    ],
    "diminuicao": [
        "registou uma diminuição de {variacao:.1f}%",
        "decresceu {variacao:.1f}%",
        "teve uma queda de {variacao:.1f}%",
    ],
    "estabilidade": [
        "manteve-se estável (variação de apenas {variacao:.1f}%)",
        "não apresentou variação significativa ({variacao:.1f}%)",
        "manteve-se sem alterações relevantes ({variacao:.1f}%)",
    ],
    "cumprimento": [
        "cumpriu a meta definida ({valor:.1f} face à meta de {meta:.1f})",
        "atingiu o objetivo estabelecido ({valor:.1f} vs {meta:.1f} de meta)",
        "superou a expectativa ({valor:.1f} face aos {meta:.1f} previstos)",
    ],
    "incumprimento": [
        "não atingiu a meta definida ({valor:.1f} face à meta de {meta:.1f})",
        "ficou abaixo do objetivo ({valor:.1f} vs {meta:.1f} de meta)",
        "não conseguiu alcançar a meta ({valor:.1f} face aos {meta:.1f} esperados)",
    ],
    "sem_comparacao": [
        "regista o valor de {valor}{unidade}",
        "apresenta {valor}{unidade}",
        "situa-se em {valor}{unidade}",
        "atinge {valor}{unidade}",
        "totaliza {valor}{unidade}",
    ],
}

# ──────────────────────────────────────────────────────────────
# Contextos qualitativos por tipo de indicador (para enriquecer
# a narrativa além do simples "aumentou/diminuiu"). As chaves de
# cada indicador devem coincidir com o que é efetivamente avaliado:
# indicadores SEM meta usam chaves de variação (aumento/diminuicao/
# estabilidade); indicadores COM meta usam chaves de meta
# (cumprimento/incumprimento) — ver `_contexto_qualitativo()`.
# ──────────────────────────────────────────────────────────────
CONTEXTOS = {
    "soc_pop_total_cim": {
        "aumento": "contrariando a tendência de despovoamento observada noutras regiões do interior de Portugal",
        "diminuicao": "acompanhando a tendência de despovoamento observada noutras regiões do interior de Portugal",
        "estabilidade": "um sinal de estabilização demográfica na região",
    },
    "amb_pct_contadores_smart": {
        "cumprimento": "uma modernização acelerada que permite leituras remotas e deteção rápida de falhas ou consumos anómalos",
        "incumprimento": "um atraso que pode comprometer a gestão eficiente da rede elétrica e a faturação baseada em consumo real",
    },
    "amb_taxa_aterro_pct": {
        "cumprimento": "um resultado exemplar que coloca o município na frente da transição para a economia circular",
        "incumprimento": "uma oportunidade clara de melhoria através do reforço da recolha seletiva e valorização de resíduos",
    },
    "mdv_acidentes_vitimas_1000hab": {
        "cumprimento": "um progresso notável na segurança rodoviária, aproximando-se da Visão Zero 2030",
        "incumprimento": "um sinal de alerta que exige ações reforçadas de prevenção de acidentes",
    },
    "eco_taxa_emprego_pct": {
        "cumprimento": "um mercado de trabalho dinâmico que atrai e retém população ativa",
        "incumprimento": "um desafio estrutural que pode estar a limitar o desenvolvimento económico local",
    },
}


def carregar_config() -> Dict[str, Any]:
    """Carrega o JSON de configuração de metas."""
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["indicadores"]


def _indice_estavel(chave: str, n_opcoes: int) -> int:
    """Escolhe um índice de 0..n_opcoes-1 de forma determinística a partir de 'chave'.

    Usa hashlib em vez do hash() nativo do Python: hash() de strings é
    aleatorizado a cada execução do processo (hash randomization, por
    segurança), pelo que duas corridas do pipeline sobre os MESMOS dados
    podiam escolher frases diferentes sem nenhum dado ter mudado. Isto
    garante que o relatório é reprodutível: mesmos dados → mesmo texto.
    """
    digest = hashlib.md5(chave.encode("utf-8")).hexdigest()
    return int(digest, 16) % n_opcoes


def _formatar_valor(v: float) -> str:
    """Formata um número para inserir numa frase quando não há um formato
    específico (%, €, etc.) definido pelo contexto — números inteiros sem
    casas decimais, milhar separado por espaço (convenção PT)."""
    if v == int(v):
        return f"{int(v):,}".replace(",", " ")
    return f"{v:,.1f}".replace(",", " ")


def classificar_variacao(
    valor_atual: float,
    valor_anterior: Optional[float],
    limiar_estabilidade: float = 1.0,
) -> tuple[str, Optional[float]]:
    """
    Classifica a variação percentual entre dois valores.
    Retorna (situacao, variacao_pct).

    Devolve situacao="sem_comparacao" (variacao_pct=None) quando não há valor
    anterior para comparar — isto é DIFERENTE de "estabilidade": estabilidade
    significa que comparámos dois valores e a diferença é pequena; sem_comparacao
    significa que só temos um valor (ex.: um cartão de totais sem série temporal)
    e não faz sentido falar em "variação de 0.0%", porque nunca se calculou
    variação nenhuma.
    """
    if valor_anterior is None or valor_anterior == 0:
        return "sem_comparacao", None

    variacao_pct = (valor_atual - valor_anterior) / valor_anterior * 100

    if abs(variacao_pct) < limiar_estabilidade:
        situacao = "estabilidade"
    elif variacao_pct > 0:
        situacao = "aumento"
    else:
        situacao = "diminuicao"

    return situacao, variacao_pct


def classificar_meta(
    valor_atual: float,
    meta_valor: float,
    direcao_boa: Optional[str] = None,
) -> str:
    """Classifica cumprimento/incumprimento de meta.

    Para indicadores em que MENOS é melhor (direcao_boa="diminuicao", ex.: taxa
    de aterro, acidentes por 1000 hab.), a meta cumpre-se por o valor atual
    ser MENOR OU IGUAL à meta — o inverso do caso por omissão ("mais é
    melhor"). Sem este ajuste, indicadores de "menos é melhor" ficam sempre
    classificados ao contrário (um valor muito acima da meta seria lido
    como sucesso).
    """
    if direcao_boa == "diminuicao":
        return "cumprimento" if valor_atual <= meta_valor else "incumprimento"
    return "cumprimento" if valor_atual >= meta_valor else "incumprimento"


def avaliar_indicador(
    chave: str,
    valor_atual: float,
    valor_anterior: Optional[float],
    config: Optional[Dict[str, Any]] = None,
    meta_valor_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Avalia um indicador completo:
    - classifica variação (aumento/diminuição/estabilidade)
    - classifica meta (cumprimento/incumprimento), se aplicável
    - decide se a situação atual é 'boa', 'má' ou 'neutra' notícia

    meta_valor_override: usa este valor em vez do 'meta_valor' do JSON de
    configuração. Necessário para metas RELATIVAS (ex.: Visão Zero 2030 —
    "reduzir 50% face a 2019 de cada município"), em que o JSON tem
    tem_meta=True mas meta_valor=None porque o alvo não é um número fixo
    igual para todos — tem de ser calculado por quem chama esta função
    (ex.: 0.5 * valor_2019_do_municipio) e passado aqui.
    """
    if config is None:
        config = carregar_config()

    cfg = config.get(chave, {})
    tem_meta = cfg.get("tem_meta", False)
    direcao_boa = cfg.get("direcao_boa")  # "aumento" ou "diminuicao" ou None
    meta_valor = meta_valor_override if meta_valor_override is not None else cfg.get("meta_valor")
    meta_ano = cfg.get("meta_ano")

    situacao, variacao_pct = classificar_variacao(valor_atual, valor_anterior)

    resultado = {
        "chave": chave,
        "valor_atual": valor_atual,
        "valor_anterior": valor_anterior,
        "variacao_pct": variacao_pct,
        "situacao": situacao,
        "tem_meta": tem_meta,
        "meta_valor": meta_valor,
        "meta_ano": meta_ano,
        "meta_cumprida": None,
        "situacao_meta": None,
        "noticia": None,
        "config": cfg,
    }

    if tem_meta and meta_valor is not None:
        situacao_meta = classificar_meta(valor_atual, meta_valor, direcao_boa)
        resultado["situacao_meta"] = situacao_meta
        resultado["meta_cumprida"] = situacao_meta == "cumprimento"

    # Decide se é "boa", "má" ou "neutra" notícia. Combina DUAS fontes de sinal:
    # a tendência (aumento/diminuição vs. direção desejada) e o cumprimento da
    # meta (quando existe). Isto importa sobretudo quando não há tendência
    # disponível (situacao="sem_comparacao", ex.: um só valor, sem série
    # temporal) — sem esta combinação, um indicador a falhar badly uma meta
    # apareceria como "neutro" só por faltar o histórico, escondendo o problema.
    noticia_tendencia = None
    if situacao in ("aumento", "diminuicao") and direcao_boa is not None:
        noticia_tendencia = "boa" if situacao == direcao_boa else "má"

    noticia_meta = None
    if tem_meta and resultado["situacao_meta"] is not None:
        noticia_meta = "boa" if resultado["situacao_meta"] == "cumprimento" else "má"

    if noticia_tendencia and noticia_meta:
        # Se os dois sinais concordam, usa-se esse veredito; se discordam
        # (ex.: a melhorar mas ainda longe da meta), fica "neutra" — sinal
        # misto, não é claramente boa nem má notícia.
        resultado["noticia"] = noticia_tendencia if noticia_tendencia == noticia_meta else "neutra"
    elif noticia_tendencia:
        resultado["noticia"] = noticia_tendencia
    elif noticia_meta:
        resultado["noticia"] = noticia_meta
    else:
        resultado["noticia"] = "neutra"

    return resultado


def _contexto_qualitativo(chave: str, tem_meta: bool, situacao: str, situacao_meta: Optional[str]) -> str:
    """Vai buscar a frase de contexto qualitativo certa para o indicador.

    Indicadores com meta definida (tem_meta=True) são avaliados pelo
    cumprimento/incumprimento da meta; os restantes são avaliados pela
    tendência (aumento/diminuição/estabilidade). Usar a chave errada faz
    com que o contexto nunca seja encontrado (dicionário sem essa chave).
    """
    if chave not in CONTEXTOS:
        return ""
    bloco = CONTEXTOS[chave]
    if tem_meta and situacao_meta in bloco:
        return bloco[situacao_meta]
    if situacao in bloco:
        return bloco[situacao]
    return ""


def gerar_narrativa(
    chave: str,
    valor_atual: float,
    valor_anterior: Optional[float],
    contexto: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    seed_extra: str = "",
    meta_valor_override: Optional[float] = None,
    unidade: str = "",
) -> str:
    """
    Gera uma narrativa automática para um único indicador.

    Parâmetros:
    - chave: ID do indicador (ex.: "soc_pop_total_cim")
    - valor_atual: valor mais recente
    - valor_anterior: valor do período anterior (ou None)
    - contexto: dict com dados adicionais (ex.: {"sujeito": "a população da CIM", "ano_inicial": 2021, "ano_final": 2025})
    - config: configuração de metas (opcional, carrega automaticamente se None)
    - seed_extra: texto extra a somar à chave ao escolher a variante de frase,
      para diferenciar duas chamadas com a MESMA chave (ex.: gráfico de mapa vs.
      gráfico de evolução do mesmo indicador) — evita que produzam texto idêntico.
    - meta_valor_override: valor de meta calculado em tempo de execução, para
      indicadores com meta RELATIVA (tem_meta=True e meta_valor=None no JSON,
      ex.: Visão Zero 2030 — ver avaliar_indicador()).
    - unidade: sufixo a somar ao valor quando não há valor_anterior nem meta
      (ex.: "%", " €") — só é usado na frase de recurso "{sujeito} regista
      o valor de {valor}{unidade}." Sem isto, um 'sujeito' que seja só um
      nome de município (ex.: "Almeirim") produz frases sem unidade
      ("Almeirim regista o valor de 7.1.", sem indicar que é uma percentagem).

    Retorna: string com a narrativa completa.
    """
    if config is None:
        config = carregar_config()

    cfg_indicador = config.get(chave, {})
    tem_meta = cfg_indicador.get("tem_meta", False)
    meta_valor = meta_valor_override if meta_valor_override is not None else cfg_indicador.get("meta_valor")
    meta_ano = cfg_indicador.get("meta_ano")

    avaliacao = avaliar_indicador(chave, valor_atual, valor_anterior, config, meta_valor_override)
    situacao = avaliacao["situacao"]
    variacao_pct = avaliacao["variacao_pct"]
    situacao_meta = avaliacao["situacao_meta"]

    chave_variante = f"{chave}{seed_extra}"

    # ──────────────────────────────────────────────────────────────
    # 1. Frase base de variação (aumento/diminuição/estabilidade).
    #    Quando não há valor_anterior ("sem_comparacao"), não existe
    #    variação nenhuma para descrever — fica None e trata-se à parte.
    # ──────────────────────────────────────────────────────────────
    if situacao == "sem_comparacao":
        frase_base = None
    else:
        idx = _indice_estavel(chave_variante, len(FRASES[situacao]))
        frase_base = FRASES[situacao][idx].format(variacao=abs(variacao_pct))

    # ──────────────────────────────────────────────────────────────
    # 2. Se houver meta, adiciona frase de cumprimento/incumprimento
    # ──────────────────────────────────────────────────────────────
    frase_meta = ""
    if tem_meta and situacao_meta:
        idx_meta = _indice_estavel(f"{chave_variante}_meta", len(FRASES[situacao_meta]))
        frase_meta = FRASES[situacao_meta][idx_meta].format(valor=valor_atual, meta=meta_valor)
        if meta_ano:
            frase_meta += f" (meta prevista para {int(meta_ano)})"

    # ──────────────────────────────────────────────────────────────
    # 3. Contexto qualitativo específico do indicador (se existir)
    # ──────────────────────────────────────────────────────────────
    contexto_qual = _contexto_qualitativo(chave, tem_meta, situacao, situacao_meta)

    # ──────────────────────────────────────────────────────────────
    # 4. Monta o texto final — o contexto qualitativo (CONTEXTOS) é escrito
    #    como continuação em minúscula (ex.: "contrariando a tendência...")
    #    por isso junta-se com vírgula à frase base, não como frase nova.
    # ──────────────────────────────────────────────────────────────
    contexto = contexto or {}
    sujeito = contexto.get("sujeito", "O indicador")
    ano_inicial = contexto.get("ano_inicial")
    ano_final = contexto.get("ano_final")

    partes = []

    if frase_base is not None:
        if ano_inicial and ano_final and ano_inicial != ano_final:
            frase_1 = f"Entre {int(ano_inicial)} e {int(ano_final)}, {sujeito} {frase_base}"
        else:
            frase_1 = f"{sujeito[0].upper()}{sujeito[1:]} {frase_base}"
        if contexto_qual and not frase_meta:
            frase_1 += f", {contexto_qual}"
        frase_1 += "."
        partes.append(frase_1)
    elif not frase_meta:
        # Sem valor anterior E sem meta: não há variação nem cumprimento a
        # relatar — só resta afirmar o valor atual, em vez de inventar uma
        # "variação de 0.0%" que nunca foi calculada.
        valor_fmt = _formatar_valor(valor_atual)
        idx_sc = _indice_estavel(chave_variante, len(FRASES["sem_comparacao"]))
        verbo = FRASES["sem_comparacao"][idx_sc].format(valor=valor_fmt, unidade=unidade)
        frase_1 = f"{sujeito[0].upper()}{sujeito[1:]} {verbo}"
        if contexto_qual:
            frase_1 += f", {contexto_qual}"
        if not frase_1.endswith("."):
            frase_1 += "."
        partes.append(frase_1)
    # Se frase_base é None mas HÁ frase_meta, não se acrescenta frase_1
    # nenhuma aqui — a frase de meta a seguir já é autossuficiente
    # (diz o valor atual e a meta), não faz falta repetir o valor.

    if frase_meta:
        frase_2 = frase_meta[0].upper() + frase_meta[1:]
        if contexto_qual and (frase_base is not None or not partes):
            frase_2 += f", {contexto_qual}"
        frase_2 += "."
        partes.append(frase_2)

    return " ".join(partes)
