"""Classificação e extração estruturada de mensagens via LLM (Anthropic).

Responsabilidades deste módulo:
  1. Montar o prompt (system + user) enviado ao modelo.
  2. Chamar a API da Anthropic.
  3. Exigir e validar saída JSON estrita, com retry em caso de resposta fora do padrão.
  4. Aplicar validações determinísticas pós-LLM (regex CNJ, enum de categoria, datas).

O LLM é o motor de classificação e extração. O regex existe apenas como *validação*
da saída do modelo — nunca como classificador.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

MODELO_PADRAO = "claude-sonnet-4-6"

CATEGORIAS_VALIDAS = (
    "urgente_prazo",
    "duvida_processo",
    "agendamento",
    "financeiro",
    "documento_recebido",
    "spam_irrelevante",
)

# Formato CNJ: NNNNNNN-DD.AAAA.J.TR.OOOO (Res. 65/2008 do CNJ).
# Usado como validação pós-LLM do campo numero_processo.
CNJ_REGEX = re.compile(r"^\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}$")

CAMPOS_OBRIGATORIOS = (
    "categoria",
    "nome_cliente",
    "numero_processo",
    "data_prazo",
    "resumo_uma_frase",
    "remetente_externo",
    "confianca",
    "justificativa_categoria",
)

SYSTEM_PROMPT = """\
Você é o sistema de triagem de mensagens de um escritório de advocacia brasileiro \
com atuação predominante em Direito do Trabalho e Previdenciário.

Sua função é ler UMA mensagem recebida (WhatsApp ou e-mail) e devolver uma \
classificação e os dados estruturados dela, para que a equipe saiba o que priorizar.

# 1. Categorias (escolha exatamente uma)

- `urgente_prazo`: há prazo processual, judicial ou de resposta explicitamente \
mencionado, ou intimação/notificação que exige manifestação. É a categoria de maior \
precedência: se a mensagem tem prazo E é dúvida, classifique como `urgente_prazo`.
- `duvida_processo`: pergunta sobre andamento, audiência, status, ou consulta \
jurídica sem prazo mencionado. Inclui potencial cliente perguntando se tem direito.
- `agendamento`: pedido de reunião, consulta, horário ou visita ao escritório.
- `financeiro`: honorários, boletos, comprovantes de pagamento, parcelas, contratos \
de honorários, cobranças.
- `documento_recebido`: a mensagem entrega ou anexa um documento solicitado \
(CTPS, laudo, RG, holerite, foto de documento).
- `spam_irrelevante`: propaganda, golpe, corrente, mensagem sem relação com o \
escritório.

Regras de desempate:
- Prazo explícito sempre vence as demais categorias.
- Documento anexado + pergunta sem prazo → `documento_recebido`.
- Comprovante de pagamento anexado → `financeiro` (o assunto prevalece sobre o anexo).
- Consulta de área não atendida ainda é `duvida_processo`, não `spam_irrelevante`. \
Spam é apenas conteúdo comercial/fraudulento sem relação com serviços jurídicos.

# 2. Campos estruturados

Extraia SOMENTE o que estiver na mensagem ou for inequívoco a partir dela. \
Nunca invente. Quando o dado não existir, use `null`.

- `nome_cliente` (string|null): nome da pessoa que assina ou se identifica na \
mensagem. Não deduza nome a partir de e-mail ou telefone; se a pessoa não se \
identificou no texto, use `null`. Terceiros citados (quem indicou, o advogado da \
outra parte) não são o cliente.
- `numero_processo` (string|null): número no formato CNJ \
`NNNNNNN-DD.AAAA.J.TR.OOOO`. Copie exatamente como aparece, preservando a \
pontuação. Se a mensagem citar outro identificador (número de contrato, protocolo), \
use `null` — não converta.
- `data_prazo` (string|null): data no formato `YYYY-MM-DD`. Preencha quando:
    (a) a mensagem trouxer data explícita ligada a prazo, audiência ou compromisso; ou
    (b) a mensagem usar expressão relativa inequívoca ("hoje", "amanhã", "depois de \
amanhã"), que você deve resolver a partir da DATA DE REFERÊNCIA informada.
  Deixe `null` para expressões vagas ("semana que vem", "nos próximos dias", "logo") \
e mencione a expressão original no resumo.
- `resumo_uma_frase` (string): uma única frase objetiva, em português, no máximo 200 \
caracteres, que diga o que a pessoa quer. Sem saudação e sem repetir o texto original.
- `remetente_externo` (boolean): `true` quando o remetente NÃO é cliente ou potencial \
cliente do escritório, mas sim parte externa do processo — advogado da parte \
contrária, preposto/departamento jurídico da empresa ré, perito, cartório, \
oficial de justiça, órgão público. Também `true` para spam/desconhecido comercial. \
`false` para cliente, potencial cliente, familiar do cliente. Na dúvida entre cliente \
e parte contrária, use o conteúdo (quem propõe acordo "conforme conversado, Dr." é \
tipicamente o advogado adverso) e o endereço do remetente (domínio de outro \
escritório é forte indício).
- `confianca` (number): 0.0 a 1.0, sua confiança na categoria escolhida.
- `justificativa_categoria` (string): no máximo 160 caracteres explicando a escolha, \
para auditoria humana.

# 3. Formato de saída — OBRIGATÓRIO

Responda com UM único objeto JSON válido e nada mais. Sem texto antes ou depois, \
sem crases, sem markdown, sem comentários. Todas as chaves abaixo são obrigatórias \
e devem aparecer mesmo quando o valor for `null`.

{"categoria":"urgente_prazo|duvida_processo|agendamento|financeiro|documento_recebido|spam_irrelevante","nome_cliente":null,"numero_processo":null,"data_prazo":null,"resumo_uma_frase":"","remetente_externo":false,"confianca":0.0,"justificativa_categoria":""}
"""


class ErroTriagem(Exception):
    """Falha definitiva ao triar uma mensagem (todas as tentativas esgotadas)."""


@dataclass
class ResultadoTriagem:
    """Saída validada do LLM para uma mensagem, mais metadados de execução."""

    categoria: str
    nome_cliente: str | None
    numero_processo: str | None
    numero_processo_valido: bool
    data_prazo: str | None
    resumo_uma_frase: str
    remetente_externo: bool
    confianca: float
    justificativa_categoria: str
    tentativas: int = 1
    modelo: str = MODELO_PADRAO
    tokens_entrada: int = 0
    tokens_saida: int = 0
    avisos: list[str] = field(default_factory=list)


class Classificador:
    """Encapsula prompt, chamada à API e validação da resposta."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        modelo: str = MODELO_PADRAO,
        max_tentativas: int = 3,
        pausa_entre_tentativas: float = 1.0,
    ) -> None:
        # max_tentativas = 1 chamada inicial + 2 retries, conforme requisito.
        self.client = client
        self.modelo = modelo
        self.max_tentativas = max_tentativas
        self.pausa_entre_tentativas = pausa_entre_tentativas
        # Consumo bruto: inclui tentativas descartadas, que também são cobradas.
        self.tokens_entrada_total = 0
        self.tokens_saida_total = 0

    # ------------------------------------------------------------------ público

    def classificar(self, mensagem: dict[str, Any], hoje: date) -> ResultadoTriagem:
        """Classifica uma mensagem, com retry em caso de resposta fora do padrão.

        Levanta ErroTriagem quando todas as tentativas falham.
        """
        historico: list[dict[str, Any]] = [
            {"role": "user", "content": self._montar_mensagem_usuario(mensagem, hoje)}
        ]
        ultimo_erro = ""

        for tentativa in range(1, self.max_tentativas + 1):
            try:
                resposta = self.client.messages.create(
                    model=self.modelo,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    thinking={"type": "disabled"},
                    output_config={"effort": "low"},
                    messages=historico,
                )
            except anthropic.APIError as exc:
                ultimo_erro = f"erro de API: {exc}"
                logger.warning(
                    "msg #%s tentativa %d/%d falhou na chamada da API: %s",
                    mensagem.get("id"),
                    tentativa,
                    self.max_tentativas,
                    exc,
                )
                self._aguardar(tentativa)
                continue

            self.tokens_entrada_total += resposta.usage.input_tokens
            self.tokens_saida_total += resposta.usage.output_tokens
            texto_bruto = self._extrair_texto(resposta)

            try:
                dados = self._parse_e_valida(texto_bruto)
            except ValueError as exc:
                ultimo_erro = str(exc)
                logger.warning(
                    "msg #%s tentativa %d/%d devolveu resposta inválida (%s). Bruto: %r",
                    mensagem.get("id"),
                    tentativa,
                    self.max_tentativas,
                    exc,
                    texto_bruto[:400],
                )
                # Devolve o erro ao modelo para que ele corrija na próxima tentativa.
                historico.append({"role": "assistant", "content": texto_bruto or "(vazio)"})
                historico.append(
                    {
                        "role": "user",
                        "content": (
                            f"Sua resposta anterior foi rejeitada pelo validador: {exc}. "
                            "Responda novamente APENAS com o objeto JSON válido, "
                            "com todas as chaves obrigatórias, sem crases e sem "
                            "nenhum texto fora do JSON."
                        ),
                    }
                )
                self._aguardar(tentativa)
                continue

            return self._montar_resultado(dados, tentativa, resposta)

        raise ErroTriagem(
            f"não foi possível triar a mensagem após {self.max_tentativas} tentativas "
            f"(último erro: {ultimo_erro})"
        )

    # ------------------------------------------------------------------ internos

    @staticmethod
    def _montar_mensagem_usuario(mensagem: dict[str, Any], hoje: date) -> str:
        return (
            f"DATA DE REFERÊNCIA: {hoje.isoformat()}\n"
            f"CANAL: {mensagem.get('canal', 'desconhecido')}\n"
            f"REMETENTE: {mensagem.get('de', 'desconhecido')}\n"
            "MENSAGEM:\n"
            "<<<\n"
            f"{mensagem.get('texto', '')}\n"
            ">>>\n\n"
            "Classifique e extraia os dados desta mensagem. Responda apenas com o JSON."
        )

    @staticmethod
    def _extrair_texto(resposta: anthropic.types.Message) -> str:
        partes = [bloco.text for bloco in resposta.content if bloco.type == "text"]
        return "".join(partes).strip()

    @staticmethod
    def _isolar_json(texto: str) -> str:
        """Remove cercas de markdown e texto solto ao redor do objeto JSON.

        Defensivo: o prompt já pede JSON puro, mas modelos ocasionalmente
        envolvem a saída em ```json. Isso não é parsing de conteúdo — apenas
        tolerância de formato antes da validação real.
        """
        limpo = texto.strip()
        if limpo.startswith("```"):
            limpo = re.sub(r"^```[a-zA-Z]*\s*", "", limpo)
            limpo = re.sub(r"\s*```$", "", limpo)
            limpo = limpo.strip()
        inicio = limpo.find("{")
        fim = limpo.rfind("}")
        if inicio == -1 or fim == -1 or fim < inicio:
            return limpo
        return limpo[inicio : fim + 1]

    def _parse_e_valida(self, texto_bruto: str) -> dict[str, Any]:
        """Faz o parse do JSON e valida schema, enum e tipos. Levanta ValueError."""
        if not texto_bruto:
            raise ValueError("resposta vazia do modelo")

        try:
            dados = json.loads(self._isolar_json(texto_bruto))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON inválido ({exc.msg})") from exc

        if not isinstance(dados, dict):
            raise ValueError("o JSON retornado não é um objeto")

        faltando = [campo for campo in CAMPOS_OBRIGATORIOS if campo not in dados]
        if faltando:
            raise ValueError(f"campos obrigatórios ausentes: {', '.join(faltando)}")

        categoria = dados["categoria"]
        if categoria not in CATEGORIAS_VALIDAS:
            raise ValueError(
                f"categoria {categoria!r} fora do enum permitido "
                f"({', '.join(CATEGORIAS_VALIDAS)})"
            )

        if not isinstance(dados["remetente_externo"], bool):
            raise ValueError("remetente_externo deve ser booleano")

        resumo = dados["resumo_uma_frase"]
        if not isinstance(resumo, str) or not resumo.strip():
            raise ValueError("resumo_uma_frase deve ser uma string não vazia")

        for campo in ("nome_cliente", "numero_processo", "data_prazo"):
            valor = dados[campo]
            if valor is not None and not isinstance(valor, str):
                raise ValueError(f"{campo} deve ser string ou null")

        try:
            dados["confianca"] = float(dados["confianca"])
        except (TypeError, ValueError) as exc:
            raise ValueError("confianca deve ser numérica") from exc
        if not 0.0 <= dados["confianca"] <= 1.0:
            raise ValueError("confianca deve estar entre 0.0 e 1.0")

        return dados

    def _montar_resultado(
        self,
        dados: dict[str, Any],
        tentativa: int,
        resposta: anthropic.types.Message,
    ) -> ResultadoTriagem:
        avisos: list[str] = []

        numero = normalizar_vazio(dados["numero_processo"])
        numero_valido = bool(numero) and validar_cnj(numero)
        if numero and not numero_valido:
            # Alucinação de formato: mantemos o valor para auditoria, mas marcado.
            avisos.append(f"numero_processo {numero!r} não segue o formato CNJ")

        data_prazo = normalizar_vazio(dados["data_prazo"])
        if data_prazo and not data_valida(data_prazo):
            avisos.append(f"data_prazo {data_prazo!r} inválida — descartada")
            data_prazo = None

        resumo = dados["resumo_uma_frase"].strip()
        if len(resumo) > 220:
            resumo = resumo[:217].rstrip() + "..."
            avisos.append("resumo truncado em 220 caracteres")

        return ResultadoTriagem(
            categoria=dados["categoria"],
            nome_cliente=normalizar_vazio(dados["nome_cliente"]),
            numero_processo=numero,
            numero_processo_valido=numero_valido,
            data_prazo=data_prazo,
            resumo_uma_frase=resumo,
            remetente_externo=dados["remetente_externo"],
            confianca=dados["confianca"],
            justificativa_categoria=str(dados["justificativa_categoria"]).strip()[:200],
            tentativas=tentativa,
            modelo=self.modelo,
            tokens_entrada=resposta.usage.input_tokens,
            tokens_saida=resposta.usage.output_tokens,
            avisos=avisos,
        )

    def _aguardar(self, tentativa: int) -> None:
        if self.pausa_entre_tentativas > 0:
            time.sleep(self.pausa_entre_tentativas * tentativa)


# ---------------------------------------------------------------- utilitários


def validar_cnj(numero: str) -> bool:
    """Valida o FORMATO do número CNJ (NNNNNNN-DD.AAAA.J.TR.OOOO)."""
    return bool(CNJ_REGEX.match(numero.strip()))


def data_valida(texto: str) -> bool:
    """Confere se a string é uma data real no formato YYYY-MM-DD."""
    try:
        datetime.strptime(texto.strip(), "%Y-%m-%d")
    except ValueError:
        return False
    return True


def normalizar_vazio(valor: Any) -> str | None:
    """Converte strings vazias, 'null' e 'none' textuais em None."""
    if valor is None:
        return None
    texto = str(valor).strip()
    if not texto or texto.lower() in {"null", "none", "n/a", "-"}:
        return None
    return texto
