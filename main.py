"""Pipeline de triagem inteligente de mensagens — BKM Advogados.

Fluxo: lê mensagens.json → deduplica → classifica via LLM → enriquece com a base
de clientes → grava em SQLite → gera resumo_diario.md.

Uso:
    python main.py
    python main.py --entrada mensagens.json --db triagem.db --saida resumo_diario.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

import database
from classifier import CATEGORIAS_VALIDAS, MODELO_PADRAO, Classificador, ErroTriagem

logger = logging.getLogger("triagem")

# Preços públicos da API Anthropic para claude-sonnet-4-6 (USD por 1 milhão de tokens).
PRECO_ENTRADA_POR_MTOK = 3.00
PRECO_SAIDA_POR_MTOK = 15.00

ROTULOS_CATEGORIA = {
    "urgente_prazo": "Urgente / prazo",
    "duvida_processo": "Dúvida sobre processo",
    "agendamento": "Agendamento",
    "financeiro": "Financeiro",
    "documento_recebido": "Documento recebido",
    "spam_irrelevante": "Spam / irrelevante",
}


# ------------------------------------------------------------------ base de clientes


class BaseClientes:
    """Índice simples da carteira de clientes, para detectar remetente conhecido.

    Casamento por contato (telefone/e-mail) é exato; por nome é normalizado
    (sem acento, minúsculo). Serve de sinal auxiliar — nunca sobrescreve a
    classificação do LLM.
    """

    def __init__(self, clientes: list[dict[str, Any]]) -> None:
        self.por_contato: dict[str, dict[str, Any]] = {}
        self.por_nome: dict[str, dict[str, Any]] = {}
        for cliente in clientes:
            for contato in cliente.get("contatos", []):
                self.por_contato[_normalizar(contato)] = cliente
            self.por_nome[_normalizar(cliente["nome"])] = cliente

    @classmethod
    def carregar(cls, caminho: Path) -> "BaseClientes":
        if not caminho.exists():
            logger.warning("base de clientes não encontrada em %s — seguindo sem ela", caminho)
            return cls([])
        return cls(json.loads(caminho.read_text(encoding="utf-8")))

    def identificar(self, remetente: str, nome_cliente: str | None) -> dict[str, Any] | None:
        cliente = self.por_contato.get(_normalizar(remetente))
        if cliente:
            return cliente
        if not nome_cliente:
            return None
        alvo = _normalizar(nome_cliente)
        for nome, cliente in self.por_nome.items():
            if alvo == nome or alvo in nome or nome in alvo:
                return cliente
        return None


def _normalizar(texto: str) -> str:
    semi = unicodedata.normalize("NFKD", texto or "")
    semi = "".join(c for c in semi if not unicodedata.combining(c))
    return " ".join(semi.lower().split())


# ------------------------------------------------------------------ processamento


def processar(
    mensagens: list[dict[str, Any]],
    classificador: Classificador,
    conn: sqlite3.Connection,
    clientes: BaseClientes,
    hoje: date,
) -> dict[str, int]:
    """Triagem de todas as mensagens. Retorna contadores de desfecho."""
    contadores = {"ok": 0, "duplicada": 0, "falha": 0}

    for mensagem in mensagens:
        msg_id = mensagem.get("id")
        canal = mensagem.get("canal", "desconhecido")
        remetente = mensagem.get("de", "desconhecido")
        texto = mensagem.get("texto", "")
        hash_conteudo = database.calcular_hash(canal, remetente, texto)

        base = {
            "id": msg_id,
            "canal": canal,
            "remetente": remetente,
            "texto": texto,
            "hash_conteudo": hash_conteudo,
        }

        # Bônus: deduplicação. Reenvios idênticos não consomem nova chamada de API.
        original = database.buscar_original(conn, hash_conteudo)
        if original is not None and original["id"] != msg_id:
            logger.info("msg #%s é duplicata da #%s — pulando LLM", msg_id, original["id"])
            database.salvar(
                conn,
                {
                    **base,
                    "status": "duplicada",
                    "categoria": original["categoria"],
                    "resumo_uma_frase": original["resumo_uma_frase"],
                    "remetente_externo": original["remetente_externo"],
                    "duplicada_de": original["id"],
                },
            )
            contadores["duplicada"] += 1
            continue

        try:
            resultado = classificador.classificar(mensagem, hoje)
        except ErroTriagem as exc:
            logger.error("msg #%s: FALHA DEFINITIVA — %s", msg_id, exc)
            database.salvar(conn, {**base, "status": "falha", "erro": str(exc)})
            contadores["falha"] += 1
            continue

        cliente = clientes.identificar(remetente, resultado.nome_cliente)
        avisos = list(resultado.avisos)
        if cliente and resultado.remetente_externo:
            # Sinal conflitante: vale revisão humana, mas não sobrescrevemos o LLM.
            avisos.append(
                f"LLM marcou remetente_externo, mas o contato consta na carteira ({cliente['id']})"
            )
            logger.warning("msg #%s: %s", msg_id, avisos[-1])

        database.salvar(
            conn,
            {
                **base,
                "status": "ok",
                "categoria": resultado.categoria,
                "nome_cliente": resultado.nome_cliente or (cliente["nome"] if cliente else None),
                "numero_processo": resultado.numero_processo,
                "numero_processo_valido": int(resultado.numero_processo_valido),
                "data_prazo": resultado.data_prazo,
                "resumo_uma_frase": resultado.resumo_uma_frase,
                "remetente_externo": int(resultado.remetente_externo),
                "cliente_existente": int(cliente is not None),
                "cliente_id": cliente["id"] if cliente else None,
                "confianca": resultado.confianca,
                "justificativa": resultado.justificativa_categoria,
                "avisos": "; ".join(avisos) or None,
                "tentativas": resultado.tentativas,
                "modelo": resultado.modelo,
                "tokens_entrada": resultado.tokens_entrada,
                "tokens_saida": resultado.tokens_saida,
            },
        )
        contadores["ok"] += 1
        logger.info(
            "msg #%s → %-18s | prazo=%s | externo=%s | tentativas=%d",
            msg_id,
            resultado.categoria,
            resultado.data_prazo or "-",
            resultado.remetente_externo,
            resultado.tentativas,
        )

    return contadores


# ------------------------------------------------------------------ resumo diário


def _situacao_prazo(data_prazo: str | None, hoje: date) -> str:
    if not data_prazo:
        return "sem data explícita"
    prazo = datetime.strptime(data_prazo, "%Y-%m-%d").date()
    dias = (prazo - hoje).days
    if dias < 0:
        return f"**VENCIDO há {abs(dias)} dia(s)**"
    if dias == 0:
        return "**VENCE HOJE**"
    if dias == 1:
        return "vence amanhã"
    return f"faltam {dias} dias"


def _ordem_prazo(linha: sqlite3.Row) -> tuple[int, str]:
    # Sem data vai para o fim da lista de urgentes.
    return (1, "") if not linha["data_prazo"] else (0, linha["data_prazo"])


def gerar_resumo_diario(
    conn: sqlite3.Connection,
    caminho_saida: Path,
    hoje: date,
    contadores: dict[str, int],
) -> None:
    """Escreve o resumo_diario.md: urgentes no topo, totais e destaques externos."""
    linhas = database.listar(conn)
    triadas = [linha for linha in linhas if linha["status"] == "ok"]
    urgentes = sorted(
        [linha for linha in triadas if linha["categoria"] == "urgente_prazo"],
        key=_ordem_prazo,
    )
    externos = [linha for linha in triadas if linha["remetente_externo"]]
    totais = {linha["categoria"]: linha["total"] for linha in database.contar_por_categoria(conn)}

    out: list[str] = []
    out.append(f"# Resumo diário de triagem — {hoje.strftime('%d/%m/%Y')}\n")
    out.append(
        f"**{len(linhas)}** mensagens recebidas: "
        f"{contadores['ok']} triadas, "
        f"{contadores['duplicada']} duplicadas, "
        f"{contadores['falha']} com falha.\n"
    )

    # 1. Urgentes no topo, com prazo e cliente.
    out.append("## 🔴 Urgentes — prazo\n")
    if not urgentes:
        out.append("_Nenhuma mensagem com prazo hoje._\n")
    else:
        out.append("| # | Prazo | Situação | Cliente | Processo | Origem | Assunto |")
        out.append("|---|-------|----------|---------|----------|--------|---------|")
        for linha in urgentes:
            origem = "⚠️ **PARTE EXTERNA**" if linha["remetente_externo"] else "Cliente"
            processo = linha["numero_processo"] or "—"
            if linha["numero_processo"] and not linha["numero_processo_valido"]:
                processo += " ⚠️(fora do padrão CNJ)"
            out.append(
                f"| {linha['id']} "
                f"| {_formatar_data(linha['data_prazo'])} "
                f"| {_situacao_prazo(linha['data_prazo'], hoje)} "
                f"| {linha['nome_cliente'] or '—'} "
                f"| {processo} "
                f"| {origem} "
                f"| {linha['resumo_uma_frase']} |"
            )
        out.append("")

    # 2. Destaque de remetentes externos (não só os urgentes).
    out.append("## ⚠️ Mensagens de parte externa / contrária\n")
    if not externos:
        out.append("_Nenhuma._\n")
    else:
        out.append(
            "Estas mensagens **não vieram de cliente** e não devem ser respondidas "
            "pelo atendimento sem passar pelo advogado responsável.\n"
        )
        for linha in externos:
            prazo = (
                f" — prazo **{_formatar_data(linha['data_prazo'])}** "
                f"({_situacao_prazo(linha['data_prazo'], hoje)})"
                if linha["data_prazo"]
                else ""
            )
            out.append(
                f"- **#{linha['id']}** ({linha['canal']}, `{linha['remetente']}`) — "
                f"{ROTULOS_CATEGORIA.get(linha['categoria'], linha['categoria'])}"
                f"{prazo}: {linha['resumo_uma_frase']}"
            )
        out.append("")

    # 3. Totais por categoria.
    out.append("## Totais por categoria\n")
    out.append("| Categoria | Quantidade |")
    out.append("|-----------|------------|")
    for categoria in CATEGORIAS_VALIDAS:
        out.append(f"| {ROTULOS_CATEGORIA[categoria]} | {totais.get(categoria, 0)} |")
    out.append(f"| **Total triado** | **{len(triadas)}** |")
    out.append("")

    # 4. Detalhamento, para quem vai executar.
    out.append("## Detalhamento por categoria\n")
    for categoria in CATEGORIAS_VALIDAS:
        do_grupo = [linha for linha in triadas if linha["categoria"] == categoria]
        if not do_grupo:
            continue
        out.append(f"### {ROTULOS_CATEGORIA[categoria]} ({len(do_grupo)})\n")
        for linha in do_grupo:
            marcas = []
            if linha["remetente_externo"]:
                marcas.append("⚠️ parte externa")
            if linha["cliente_existente"]:
                marcas.append(f"cliente {linha['cliente_id']}")
            else:
                marcas.append("contato novo")
            if linha["numero_processo"]:
                marcas.append(f"proc. {linha['numero_processo']}")
            if linha["data_prazo"]:
                marcas.append(f"prazo {_formatar_data(linha['data_prazo'])}")
            if linha["confianca"] is not None and linha["confianca"] < 0.7:
                marcas.append(f"⚠️ confiança baixa ({linha['confianca']:.2f})")
            out.append(
                f"- **#{linha['id']}** ({linha['canal']}) {linha['resumo_uma_frase']}  \n"
                f"  <sub>{' · '.join(marcas)}</sub>"
            )
        out.append("")

    # 5. Pendências operacionais.
    problemas = [linha for linha in linhas if linha["status"] == "falha" or linha["avisos"]]
    if problemas:
        out.append("## 🛠️ Revisão humana recomendada\n")
        for linha in problemas:
            motivo = linha["erro"] or linha["avisos"]
            out.append(f"- **#{linha['id']}** ({linha['status']}): {motivo}")
        out.append("")

    duplicadas = [linha for linha in linhas if linha["status"] == "duplicada"]
    if duplicadas:
        out.append("## Duplicadas ignoradas\n")
        for linha in duplicadas:
            out.append(f"- **#{linha['id']}** é reenvio da **#{linha['duplicada_de']}**")
        out.append("")

    uso = database.totais_de_uso(conn)
    custo = _custo_usd(uso["entrada"], uso["saida"])
    entrada_fmt = f"{uso['entrada']:,}".replace(",", ".")
    saida_fmt = f"{uso['saida']:,}".replace(",", ".")
    out.append("---\n")
    out.append(
        f"<sub>Gerado automaticamente em {datetime.now().strftime('%d/%m/%Y %H:%M')} · "
        f"{entrada_fmt} tokens de entrada / {saida_fmt} de saída · "
        f"custo desta execução: US$ {custo:.4f}</sub>"
    )

    caminho_saida.write_text("\n".join(out) + "\n", encoding="utf-8")


def _formatar_data(iso: str | None) -> str:
    if not iso:
        return "—"
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%d/%m/%Y")


def _custo_usd(tokens_entrada: int, tokens_saida: int) -> float:
    return (
        tokens_entrada / 1_000_000 * PRECO_ENTRADA_POR_MTOK
        + tokens_saida / 1_000_000 * PRECO_SAIDA_POR_MTOK
    )


# ------------------------------------------------------------------ CLI


def configurar_log(caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    formato = "%(asctime)s %(levelname)-8s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=formato,
        handlers=[
            logging.FileHandler(caminho, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # O SDK da Anthropic é verboso em nível INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triagem inteligente de mensagens (BKM)")
    parser.add_argument("--entrada", default="mensagens.json", help="JSON com as mensagens")
    parser.add_argument("--clientes", default="clientes.json", help="carteira de clientes")
    parser.add_argument("--db", default="triagem.db", help="arquivo SQLite de saída")
    parser.add_argument("--saida", default="resumo_diario.md", help="resumo diário em Markdown")
    parser.add_argument("--log", default="logs/triagem.log", help="arquivo de log")
    parser.add_argument("--modelo", default=MODELO_PADRAO, help="modelo da API Anthropic")
    parser.add_argument(
        "--hoje",
        default=None,
        help="data de referência YYYY-MM-DD (padrão: hoje) — útil para reproduzir execuções",
    )
    parser.add_argument("--limite", type=int, default=None, help="processar apenas as N primeiras")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()
    configurar_log(Path(args.log))

    hoje = date.fromisoformat(args.hoje) if args.hoje else date.today()

    entrada = Path(args.entrada)
    if not entrada.exists():
        logger.error("arquivo de entrada não encontrado: %s", entrada)
        return 1

    mensagens = json.loads(entrada.read_text(encoding="utf-8"))
    if args.limite:
        mensagens = mensagens[: args.limite]

    # A chave vem de ANTHROPIC_API_KEY no ambiente/.env — nunca do código.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY não definida.")
        logger.error("Copie .env.example para .env e preencha com a sua chave.")
        return 1
    client = anthropic.Anthropic(max_retries=3)

    classificador = Classificador(client=client, modelo=args.modelo)
    clientes = BaseClientes.carregar(Path(args.clientes))
    conn = database.conectar(args.db)

    logger.info(
        "iniciando triagem de %d mensagem(ns) | modelo=%s | referência=%s",
        len(mensagens),
        args.modelo,
        hoje.isoformat(),
    )
    try:
        contadores = processar(mensagens, classificador, conn, clientes, hoje)
        gerar_resumo_diario(conn, Path(args.saida), hoje, contadores)
    finally:
        conn.close()

    logger.info(
        "concluído: %d triadas, %d duplicadas, %d falhas",
        contadores["ok"],
        contadores["duplicada"],
        contadores["falha"],
    )
    # Consumo bruto inclui as tentativas descartadas — é o número que aparece na fatura.
    logger.info(
        "uso: %d tokens de entrada, %d de saída → US$ %.4f nesta execução",
        classificador.tokens_entrada_total,
        classificador.tokens_saida_total,
        _custo_usd(classificador.tokens_entrada_total, classificador.tokens_saida_total),
    )
    logger.info("resumo diário gravado em %s", args.saida)

    return 1 if contadores["falha"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
