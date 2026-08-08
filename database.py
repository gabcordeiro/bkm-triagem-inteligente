"""Persistência em SQLite dos resultados de triagem.

Tabela única: `mensagens_triadas`. Cada linha representa uma mensagem recebida,
independentemente do desfecho (`ok`, `duplicada` ou `falha`) — nada é descartado
silenciosamente, o que permite auditoria e reprocessamento.
"""

from __future__ import annotations

import hashlib
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS mensagens_triadas (
    id                     INTEGER PRIMARY KEY,
    canal                  TEXT    NOT NULL,
    remetente              TEXT    NOT NULL,
    texto                  TEXT    NOT NULL,
    hash_conteudo          TEXT    NOT NULL,
    status                 TEXT    NOT NULL CHECK (status IN ('ok','duplicada','falha')),
    categoria              TEXT,
    nome_cliente           TEXT,
    numero_processo        TEXT,
    numero_processo_valido INTEGER,
    data_prazo             TEXT,
    resumo_uma_frase       TEXT,
    remetente_externo      INTEGER NOT NULL DEFAULT 0,
    cliente_existente      INTEGER NOT NULL DEFAULT 0,
    cliente_id             TEXT,
    confianca              REAL,
    justificativa          TEXT,
    avisos                 TEXT,
    duplicada_de           INTEGER,
    erro                   TEXT,
    tentativas             INTEGER NOT NULL DEFAULT 0,
    modelo                 TEXT,
    tokens_entrada         INTEGER NOT NULL DEFAULT 0,
    tokens_saida           INTEGER NOT NULL DEFAULT 0,
    processado_em          TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mensagens_categoria ON mensagens_triadas (categoria);
CREATE INDEX IF NOT EXISTS idx_mensagens_hash      ON mensagens_triadas (hash_conteudo);
CREATE INDEX IF NOT EXISTS idx_mensagens_prazo     ON mensagens_triadas (data_prazo);
"""

COLUNAS = (
    "id",
    "canal",
    "remetente",
    "texto",
    "hash_conteudo",
    "status",
    "categoria",
    "nome_cliente",
    "numero_processo",
    "numero_processo_valido",
    "data_prazo",
    "resumo_uma_frase",
    "remetente_externo",
    "cliente_existente",
    "cliente_id",
    "confianca",
    "justificativa",
    "avisos",
    "duplicada_de",
    "erro",
    "tentativas",
    "modelo",
    "tokens_entrada",
    "tokens_saida",
    "processado_em",
)


def conectar(caminho: str | Path) -> sqlite3.Connection:
    """Abre a conexão e garante que o schema existe."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(caminho)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def calcular_hash(canal: str, remetente: str, texto: str) -> str:
    """Impressão digital do conteúdo, usada para deduplicação.

    Normaliza acentuação, caixa e espaços em branco para que reenvios com
    pequenas variações de digitação sejam reconhecidos como a mesma mensagem.
    """
    normalizado = unicodedata.normalize("NFKD", texto)
    normalizado = "".join(c for c in normalizado if not unicodedata.combining(c))
    normalizado = " ".join(normalizado.lower().split())
    base = f"{canal.strip().lower()}|{remetente.strip().lower()}|{normalizado}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def buscar_original(conn: sqlite3.Connection, hash_conteudo: str) -> sqlite3.Row | None:
    """Retorna a primeira mensagem triada com sucesso que tenha o mesmo conteúdo."""
    cur = conn.execute(
        "SELECT * FROM mensagens_triadas "
        "WHERE hash_conteudo = ? AND status = 'ok' "
        "ORDER BY id LIMIT 1",
        (hash_conteudo,),
    )
    return cur.fetchone()


def salvar(conn: sqlite3.Connection, registro: dict[str, Any]) -> None:
    """Insere ou substitui o registro de uma mensagem (idempotente por `id`)."""
    registro.setdefault("processado_em", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    valores = [registro.get(coluna) for coluna in COLUNAS]
    placeholders = ", ".join("?" for _ in COLUNAS)
    conn.execute(
        f"INSERT OR REPLACE INTO mensagens_triadas ({', '.join(COLUNAS)}) "
        f"VALUES ({placeholders})",
        valores,
    )
    conn.commit()


def listar(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Todas as mensagens, em ordem de id."""
    return conn.execute("SELECT * FROM mensagens_triadas ORDER BY id").fetchall()


def contar_por_categoria(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Totais por categoria, considerando apenas mensagens triadas com sucesso."""
    return conn.execute(
        "SELECT categoria, COUNT(*) AS total FROM mensagens_triadas "
        "WHERE status = 'ok' GROUP BY categoria ORDER BY total DESC, categoria"
    ).fetchall()


def totais_de_uso(conn: sqlite3.Connection) -> sqlite3.Row:
    """Soma de tokens consumidos — base para acompanhar custo real."""
    return conn.execute(
        "SELECT COALESCE(SUM(tokens_entrada), 0) AS entrada, "
        "       COALESCE(SUM(tokens_saida), 0)   AS saida "
        "FROM mensagens_triadas"
    ).fetchone()
