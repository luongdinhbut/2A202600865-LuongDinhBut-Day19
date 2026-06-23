from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import requests
from networkx.readwrite import json_graph


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "dataset"
DEFAULT_OUTPUTS_DIR = ROOT / "outputs"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
    "what",
    "which",
    "why",
    "how",
    "nhung",
    "nao",
    "the",
    "va",
    "voi",
    "cua",
    "tai",
    "sao",
    "nhu",
    "the",
    "nao",
    "trong",
    "duoc",
    "noi",
    "ve",
}


@dataclass
class Document:
    doc_id: str
    path: Path
    query: str
    title: str
    link: str
    content: str


class FireworksChatClient:
    def __init__(self) -> None:
        load_env_file(ROOT / ".env")
        self.api_key = os.getenv("FIREWORKS_API_KEY", "").strip()
        self.base_url = os.getenv(
            "FIREWORKS_BASE_URL",
            "https://api.fireworks.ai/inference/v1/chat/completions",
        ).strip()
        self.model = os.getenv(
            "FIREWORKS_MODEL",
            "accounts/fireworks/models/deepseek-v4-pro",
        ).strip()
        self.max_tokens = env_int("FIREWORKS_MAX_TOKENS", 8192)
        self.top_k = env_int("FIREWORKS_TOP_K", 40)
        self.temperature = env_float("FIREWORKS_TEMPERATURE", 0.1)
        self.timeout = env_int("FIREWORKS_TIMEOUT_SECONDS", 180)
        self.retries = env_int("FIREWORKS_RETRIES", 4)
        self.retry_sleep = env_float("FIREWORKS_RETRY_SLEEP_SECONDS", 2.0)
        self.use_json_mode = env_bool("FIREWORKS_USE_JSON_MODE", False)
        self.calls: list[dict[str, Any]] = []

    def require_api_key(self) -> None:
        if not self.api_key or self.api_key.lower() in {"replace_me", "your_key_here"}:
            raise RuntimeError(
                "Missing FIREWORKS_API_KEY. Dien API key vao file .env roi chay lai."
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        expect_json: bool = False,
        label: str = "chat",
    ) -> str:
        self.require_api_key()
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "top_k": self.top_k,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": self.temperature,
            "messages": messages,
        }
        if expect_json and self.use_json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        data = self._post(payload)
        elapsed = time.perf_counter() - started
        content = data["choices"][0]["message"]["content"]
        self.calls.append(
            {
                "label": label,
                "elapsed_seconds": round(elapsed, 3),
                "usage": data.get("usage", {}),
            }
        )
        return content

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_response: requests.Response | None = None
        last_error: requests.RequestException | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.retries:
                    sleep_for = self.retry_sleep * (2**attempt)
                    print(
                        f"[fireworks] request failed ({exc}); "
                        f"sleeping {sleep_for:.1f}s before retry {attempt + 1}/{self.retries}"
                    )
                    time.sleep(sleep_for)
                    continue
                raise RuntimeError(f"Fireworks request failed after retries: {exc}") from exc
            if (
                response.status_code >= 400
                and "response_format" in payload
                and self.use_json_mode
            ):
                retry_payload = dict(payload)
                retry_payload.pop("response_format", None)
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    data=json.dumps(retry_payload),
                    timeout=self.timeout,
                )
            last_response = response
            if not is_retryable_fireworks_response(response):
                break
            if attempt < self.retries:
                sleep_for = self.retry_sleep * (2**attempt)
                print(
                    f"[fireworks] retryable API response ({response.status_code}); "
                    f"sleeping {sleep_for:.1f}s before retry {attempt + 1}/{self.retries}"
                )
                time.sleep(sleep_for)
        response = last_response
        if response is None:
            raise RuntimeError(f"Fireworks API did not return a response: {last_error}")
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"Fireworks API error: {response.text}") from exc
        return response.json()


def is_retryable_fireworks_response(response: requests.Response) -> bool:
    text = response.text.lower()
    return (
        response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
        or "overloaded" in text
        or "temporarily unavailable" in text
        or "rate limit" in text
    )


class FlatRetriever:
    def __init__(self, docs: list[Document], chunk_chars: int = 1200) -> None:
        self.chunks = build_chunks(docs, chunk_chars=chunk_chars)
        self.doc_freq: Counter[str] = Counter()
        self.term_freqs: list[Counter[str]] = []
        for chunk in self.chunks:
            tf = Counter(tokenize(chunk["text"]))
            self.term_freqs.append(tf)
            self.doc_freq.update(tf.keys())

    def search(self, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        query_tf = Counter(tokenize(query))
        scores: list[tuple[float, int]] = []
        for idx, tf in enumerate(self.term_freqs):
            score = cosine_tfidf(query_tf, tf, self.doc_freq, len(self.chunks))
            if score > 0:
                scores.append((score, idx))
        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            item = dict(self.chunks[idx])
            item["score"] = round(score, 4)
            results.append(item)
        return results


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def read_documents(dataset_dir: Path, limit: int | None = None) -> list[Document]:
    paths = sorted(dataset_dir.glob("doc_*.txt"), key=doc_sort_key)
    if limit:
        paths = paths[:limit]
    return [parse_document(path) for path in paths]


def doc_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"doc_(\d+)\.txt$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    doc_id = path.stem
    return Document(
        doc_id=doc_id,
        path=path,
        query=find_prefixed_line(raw, "Query:"),
        title=find_prefixed_line(raw, "Title:"),
        link=find_prefixed_line(raw, "Link:"),
        content=extract_full_content(raw),
    )


def find_prefixed_line(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def extract_full_content(text: str) -> str:
    if "Full Content:" in text:
        return text.split("Full Content:", 1)[1].strip()
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "\n\n[TRUNCATED]"


def index_corpus(
    *,
    dataset_dir: Path,
    outputs_dir: Path,
    limit_docs: int | None,
    force: bool,
    client: FireworksChatClient,
) -> nx.MultiDiGraph:
    docs = read_documents(dataset_dir, limit_docs)
    ensure_output_dirs(outputs_dir)
    extraction_dir = outputs_dir / "cache" / "extractions"
    extractions = []
    for idx, doc in enumerate(docs, start=1):
        print(f"[index] {idx}/{len(docs)} extracting {doc.doc_id}: {doc.title[:80]}")
        extractions.append(extract_document(doc, client, extraction_dir, force=force))

    graph = build_graph(extractions)
    save_graph_artifacts(graph, outputs_dir, extractions, client)
    return graph


def extract_document(
    doc: Document,
    client: FireworksChatClient,
    extraction_dir: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    extraction_dir.mkdir(parents=True, exist_ok=True)
    cache_path = extraction_dir / f"{doc.doc_id}.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    max_doc_chars = env_int("MAX_DOC_CHARS", 7000)
    max_triples = env_int("MAX_TRIPLES_PER_DOC", 20)
    parse_retries = env_int("EXTRACTION_PARSE_RETRIES", 2)
    raw_dir = extraction_dir.parent / "raw_responses"
    last_error: Exception | None = None
    raw = ""

    for attempt in range(1, parse_retries + 2):
        system, user = build_extraction_messages(
            doc,
            max_doc_chars=max_doc_chars,
            max_triples=max_triples,
            attempt=attempt,
        )
        raw = client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            expect_json=True,
            max_tokens=env_int("EXTRACTION_MAX_TOKENS", 6000),
            label=f"extract:{doc.doc_id}:attempt_{attempt}",
        )
        save_raw_response(raw_dir, doc.doc_id, attempt, raw)
        try:
            parsed = parse_json_object(raw)
            normalized = normalize_extraction(parsed, doc)
            cache_path.write_text(
                json.dumps(normalized, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return normalized
        except Exception as exc:
            last_error = exc
            print(
                f"[index] invalid JSON for {doc.doc_id} on attempt {attempt}: {exc}"
            )

    if env_bool("ALLOW_HEURISTIC_EXTRACTION_FALLBACK", True):
        print(f"[index] using heuristic fallback for {doc.doc_id}")
        fallback = heuristic_extraction(doc, last_error)
        cache_path.write_text(
            json.dumps(fallback, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return fallback

    raw_path = raw_dir / f"{doc.doc_id}_attempt_{parse_retries + 1}.txt"
    raise RuntimeError(
        f"Could not parse JSON extraction for {doc.doc_id}. "
        f"Last error: {last_error}. Raw response saved at {raw_path}"
    )


def build_extraction_messages(
    doc: Document,
    *,
    max_doc_chars: int,
    max_triples: int,
    attempt: int,
) -> tuple[str, str]:
    attempt_triples = max_triples
    doc_chars = max_doc_chars
    if attempt == 2:
        attempt_triples = min(max_triples, 12)
        doc_chars = min(max_doc_chars, 4500)
    elif attempt >= 3:
        attempt_triples = min(max_triples, 8)
        doc_chars = min(max_doc_chars, 3200)

    system = (
        "You extract a compact knowledge graph from market research text. "
        "Return only one valid JSON object. Do not wrap it in markdown. "
        "Do not include analysis, reasoning, prose, or code fences. "
        "Be brief: triples are more important than entity descriptions."
    )
    retry_note = ""
    if attempt > 1:
        retry_note = (
            "\nImportant retry instruction: the previous response was not valid JSON. "
            "Start your answer with `{` and end with `}`. Return a smaller minified JSON object. "
            "Use fewer triples and omit long descriptions."
        )
    entity_rule = (
        'Use short entities: {"name":"Tesla","type":"Company","description":""}. '
        "The description field must be empty or under 12 words."
    )
    if attempt >= 3:
        entity_rule = (
            "Keep entities minimal. Include only entities that appear in triples. "
            'Use {"name":"Tesla","type":"Company","description":""}.'
        )
    user = f"""
Task: Extract factual entities and relation triples for a GraphRAG system about the electric vehicle sector.

Rules:
- Return JSON only.
- Prefer concrete entities: companies, agencies, reports, countries, cities, policies, market metrics, dates, technologies, consumer concerns.
- Use concise UPPER_SNAKE_CASE predicates.
- Keep evidence under 18 words.
- Do not invent facts.
- Extract at most {attempt_triples} high-value triples and at most {attempt_triples + 6} entities.
- {entity_rule}
- Use this schema:
{{
  "entities": [
    {{"name": "Tesla", "type": "Company", "description": ""}}
  ],
  "triples": [
    {{"subject": "Tesla", "predicate": "MARKET_SHARE", "object": "51.3% in Q1 2024", "evidence": "Tesla's share was 51.3%", "confidence": 0.94}}
  ]
}}

Document metadata:
- doc_id: {doc.doc_id}
- query: {doc.query}
- title: {doc.title}
- link: {doc.link}

Document text:
{truncate_text(doc.content, doc_chars)}
{retry_note}
""".strip()
    return system, user


def save_raw_response(raw_dir: Path, doc_id: str, attempt: int, raw: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{doc_id}_attempt_{attempt}.txt"
    path.write_text(raw or "", encoding="utf-8")


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = (raw or "").strip().lstrip("\ufeff")
    if not cleaned:
        raise ValueError("LLM returned an empty response instead of JSON.")
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S | re.I)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = parse_embedded_json_object(cleaned)
    if isinstance(parsed, list):
        parsed = {"triples": parsed}
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from LLM.")
    return parsed


def parse_embedded_json_object(cleaned: str) -> Any:
    decoder = json.JSONDecoder()
    start = cleaned.find("{")
    if start == -1:
        preview = cleaned[:160].replace("\n", " ")
        raise ValueError(f"No JSON object found in LLM response. Preview: {preview!r}")
    try:
        parsed, _end = decoder.raw_decode(cleaned[start:])
        return parsed
    except json.JSONDecodeError:
        end = cleaned.rfind("}")
        if end == -1 or end <= start:
            preview = cleaned[:160].replace("\n", " ")
            raise ValueError(f"Malformed JSON from LLM. Preview: {preview!r}")
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            preview = cleaned[:160].replace("\n", " ")
            raise ValueError(f"Malformed JSON from LLM: {exc}. Preview: {preview!r}") from exc


def heuristic_extraction(doc: Document, error: Exception | None) -> dict[str, Any]:
    sentences = split_sentences(doc.content)
    key_sentences = rank_key_sentences(sentences, limit=8)
    entities = [
        {"name": doc.title, "type": "Report", "description": "Document title"},
        {"name": source_name_from_link(doc.link), "type": "Source", "description": doc.link},
    ]
    for entity in guess_entities(doc.content)[:18]:
        entities.append({"name": entity, "type": "Unknown", "description": ""})

    triples = [
        {
            "subject": doc.title,
            "predicate": "SOURCE_URL",
            "object": doc.link,
            "evidence": doc.link,
            "confidence": 0.6,
            "doc_id": doc.doc_id,
            "title": doc.title,
            "link": doc.link,
        }
    ]
    for sentence in key_sentences:
        triples.append(
            {
                "subject": doc.title,
                "predicate": "KEY_FINDING",
                "object": sentence[:220],
                "evidence": sentence[:300],
                "confidence": 0.45,
                "doc_id": doc.doc_id,
                "title": doc.title,
                "link": doc.link,
            }
        )
    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "link": doc.link,
        "query": doc.query,
        "entities": entities,
        "triples": triples,
        "warnings": [f"Used heuristic fallback because LLM JSON parsing failed: {error}"],
    }


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if len(sentence.strip()) > 40
    ]


def rank_key_sentences(sentences: list[str], limit: int) -> list[str]:
    keywords = {
        "sales",
        "market",
        "growth",
        "tesla",
        "charging",
        "incentive",
        "price",
        "ev",
        "electric",
        "vehicle",
        "battery",
        "consumer",
        "forecast",
    }
    scored = []
    for sentence in sentences:
        terms = set(tokenize(sentence))
        score = len(terms & keywords) + min(3, len(re.findall(r"\d", sentence)))
        scored.append((score, sentence))
    scored.sort(reverse=True)
    return [sentence for score, sentence in scored[:limit] if score > 0]


def guess_entities(text: str) -> list[str]:
    candidates = re.findall(
        r"\b(?:[A-Z][A-Za-z&.\-]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z&.\-]+|[A-Z]{2,}))*",
        text,
    )
    cleaned = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip(" ,.;:()[]{}")
        if len(candidate) < 3 or candidate.lower() in {"full content", "query", "title"}:
            continue
        key = normalize_name(candidate)
        if key and key not in seen:
            seen.add(key)
            cleaned.append(candidate)
    return cleaned


def source_name_from_link(link: str) -> str:
    match = re.search(r"https?://(?:www\.)?([^/]+)", link)
    return match.group(1) if match else "Unknown source"


def normalize_extraction(parsed: dict[str, Any], doc: Document) -> dict[str, Any]:
    entities = []
    seen_entities: set[str] = set()
    for item in coerce_list(parsed.get("entities")):
        if isinstance(item, str):
            entity = {"name": item, "type": "Unknown", "description": ""}
        elif isinstance(item, dict):
            entity = {
                "name": str(item.get("name", "")).strip(),
                "type": str(item.get("type", "Unknown")).strip() or "Unknown",
                "description": str(item.get("description", "")).strip(),
            }
        else:
            continue
        if not entity["name"]:
            continue
        key = normalize_name(entity["name"])
        if key in seen_entities:
            continue
        seen_entities.add(key)
        entities.append(entity)

    raw_triples = (
        parsed.get("triples")
        or parsed.get("relations")
        or parsed.get("edges")
        or parsed.get("facts")
        or []
    )
    triples = []
    for item in coerce_list(raw_triples):
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", item.get("source", ""))).strip()
        predicate = str(item.get("predicate", item.get("relation", ""))).strip()
        obj = str(item.get("object", item.get("target", ""))).strip()
        if not subject or not predicate or not obj:
            continue
        confidence = item.get("confidence", 0.8)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.8
        triples.append(
            {
                "subject": subject,
                "predicate": normalize_predicate(predicate),
                "object": obj,
                "evidence": str(item.get("evidence", "")).strip()[:500],
                "confidence": max(0.0, min(1.0, confidence)),
                "doc_id": doc.doc_id,
                "title": doc.title,
                "link": doc.link,
            }
        )
        for name in (subject, obj):
            key = normalize_name(name)
            if key not in seen_entities:
                seen_entities.add(key)
                entities.append({"name": name, "type": "Unknown", "description": ""})

    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "link": doc.link,
        "query": doc.query,
        "entities": entities,
        "triples": triples,
    }


def coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def build_graph(extractions: list[dict[str, Any]]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for extraction in extractions:
        for entity in extraction.get("entities", []):
            name = str(entity.get("name", "")).strip()
            if name:
                add_node(graph, name, entity.get("type", "Unknown"), entity.get("description", ""))
        for triple in extraction.get("triples", []):
            subject = triple["subject"]
            obj = triple["object"]
            subject_id = add_node(graph, subject, "Unknown", "")
            object_id = add_node(graph, obj, "Unknown", "")
            graph.add_edge(
                subject_id,
                object_id,
                predicate=triple["predicate"],
                evidence=triple.get("evidence", ""),
                confidence=triple.get("confidence", 0.8),
                doc_id=triple.get("doc_id", ""),
                title=triple.get("title", ""),
                link=triple.get("link", ""),
            )
    return graph


def add_node(graph: nx.MultiDiGraph, name: str, entity_type: str, description: str) -> str:
    node_id = stable_node_id(name)
    if node_id not in graph:
        graph.add_node(
            node_id,
            label=clean_label(name),
            type=str(entity_type or "Unknown"),
            description=str(description or ""),
        )
    else:
        node = graph.nodes[node_id]
        if node.get("type") == "Unknown" and entity_type:
            node["type"] = str(entity_type)
        if not node.get("description") and description:
            node["description"] = str(description)
    return node_id


def stable_node_id(name: str) -> str:
    base = normalize_name(name)
    if base:
        return base[:80]
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:12]
    return f"node_{digest}"


def clean_label(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip()


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_predicate(predicate: str) -> str:
    text = unicodedata.normalize("NFKD", str(predicate))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text.upper() or "RELATED_TO"


def ensure_output_dirs(outputs_dir: Path) -> None:
    (outputs_dir / "cache" / "extractions").mkdir(parents=True, exist_ok=True)


def save_graph_artifacts(
    graph: nx.MultiDiGraph,
    outputs_dir: Path,
    extractions: list[dict[str, Any]],
    client: FireworksChatClient,
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    triples_path = outputs_dir / "triples.jsonl"
    with triples_path.open("w", encoding="utf-8") as f:
        for extraction in extractions:
            for triple in extraction.get("triples", []):
                f.write(json.dumps(triple, ensure_ascii=False) + "\n")

    graph_json = json_graph.node_link_data(graph)
    (outputs_dir / "graph.json").write_text(
        json.dumps(graph_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    nx.write_graphml(graph, outputs_dir / "graph.graphml")
    graph_image = save_graph_image(graph, outputs_dir / "knowledge_graph.png")
    summary = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "documents_indexed": len(extractions),
        "triples": sum(len(item.get("triples", [])) for item in extractions),
        "graph_image": str(graph_image),
        "llm_calls_this_run": client.calls,
    }
    (outputs_dir / "graph_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[index] saved graph with {summary['nodes']} nodes and {summary['edges']} edges")


def save_graph_image(graph: nx.MultiDiGraph, path: Path, max_nodes: int = 60) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        svg_path = path.with_suffix(".svg")
        save_graph_svg(graph, svg_path, max_nodes=max_nodes)
        print(f"[visualize] matplotlib is not installed; saved SVG fallback: {svg_path}")
        return svg_path
    if graph.number_of_nodes() == 0:
        return path
    degree_rank = sorted(graph.degree, key=lambda item: item[1], reverse=True)
    keep = {node for node, _degree in degree_rank[:max_nodes]}
    sub = graph.subgraph(keep).copy()
    simple = nx.Graph(sub)
    plt.figure(figsize=(18, 12))
    pos = circular_positions(list(simple.nodes), width=16.0, height=10.0)
    node_sizes = [350 + 80 * simple.degree(node) for node in simple.nodes]
    nx.draw_networkx_nodes(
        simple,
        pos,
        node_size=node_sizes,
        node_color="#6FB1FC",
        edgecolors="#1D3557",
        linewidths=0.8,
        alpha=0.92,
    )
    nx.draw_networkx_edges(simple, pos, width=0.8, alpha=0.35, edge_color="#444444")
    labels = {
        node: shorten(str(sub.nodes[node].get("label", node)), 28)
        for node in simple.nodes
    }
    nx.draw_networkx_labels(simple, pos, labels=labels, font_size=8)
    plt.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_graph_svg(graph: nx.MultiDiGraph, path: Path, max_nodes: int = 60) -> None:
    if graph.number_of_nodes() == 0:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>", encoding="utf-8")
        return
    degree_rank = sorted(graph.degree, key=lambda item: item[1], reverse=True)
    keep = {node for node, _degree in degree_rank[:max_nodes]}
    sub = nx.Graph(graph.subgraph(keep).copy())
    width = 1400
    height = 900
    pos = circular_positions(list(sub.nodes), width=float(width - 160), height=float(height - 160))
    shifted = {node: (x + 80, y + 80) for node, (x, y) in pos.items()}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#1d2633}.edge{stroke:#6b7280;stroke-width:1;opacity:.45}.node{fill:#6fb1fc;stroke:#1d3557;stroke-width:1.2}</style>',
    ]
    for source, target in sub.edges():
        x1, y1 = shifted[source]
        x2, y2 = shifted[target]
        parts.append(f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
    for node in sub.nodes():
        x, y = shifted[node]
        degree = sub.degree(node)
        radius = min(28, 9 + degree * 1.8)
        label = xml_escape(shorten(str(graph.nodes[node].get("label", node)), 30))
        parts.append(f'<circle class="node" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}"/>')
        parts.append(f'<text x="{x + radius + 4:.1f}" y="{y + 4:.1f}">{label}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def circular_positions(nodes: list[str], *, width: float, height: float) -> dict[str, tuple[float, float]]:
    if not nodes:
        return {}
    radius_x = width / 2
    radius_y = height / 2
    center_x = radius_x
    center_y = radius_y
    return {
        node: (
            center_x + radius_x * 0.92 * math.cos(2 * math.pi * idx / len(nodes)),
            center_y + radius_y * 0.92 * math.sin(2 * math.pi * idx / len(nodes)),
        )
        for idx, node in enumerate(nodes)
    }


def xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def load_graph(outputs_dir: Path) -> nx.MultiDiGraph:
    path = outputs_dir / "graph.json"
    if not path.exists():
        raise RuntimeError("Graph not found. Hay chay: python graphrag_lab.py index")
    data = json.loads(path.read_text(encoding="utf-8"))
    graph = json_graph.node_link_graph(data)
    if not isinstance(graph, nx.MultiDiGraph):
        graph = nx.MultiDiGraph(graph)
    return graph


def build_chunks(docs: list[Document], chunk_chars: int = 1200) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for doc in docs:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.content) if p.strip()]
        current = ""
        chunk_index = 0
        for paragraph in paragraphs:
            if len(current) + len(paragraph) + 2 <= chunk_chars:
                current = f"{current}\n\n{paragraph}".strip()
                continue
            if current:
                chunk_index += 1
                chunks.append(make_chunk(doc, chunk_index, current))
            current = paragraph[:chunk_chars]
        if current:
            chunk_index += 1
            chunks.append(make_chunk(doc, chunk_index, current))
    return chunks


def make_chunk(doc: Document, chunk_index: int, text: str) -> dict[str, str]:
    return {
        "doc_id": doc.doc_id,
        "chunk_id": f"{doc.doc_id}_chunk_{chunk_index}",
        "title": doc.title,
        "link": doc.link,
        "text": f"Title: {doc.title}\nSource: {doc.link}\n{text}",
    }


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9\-]{1,}", normalized)
    return [token for token in tokens if token not in STOPWORDS]


def cosine_tfidf(
    query_tf: Counter[str],
    doc_tf: Counter[str],
    doc_freq: Counter[str],
    total_docs: int,
) -> float:
    if not query_tf or not doc_tf:
        return 0.0
    dot = 0.0
    q_norm = 0.0
    d_norm = 0.0
    terms = set(query_tf) | set(doc_tf)
    for term in terms:
        idf = math.log((1 + total_docs) / (1 + doc_freq.get(term, 0))) + 1
        q_weight = query_tf.get(term, 0) * idf
        d_weight = doc_tf.get(term, 0) * idf
        dot += q_weight * d_weight
        q_norm += q_weight * q_weight
        d_norm += d_weight * d_weight
    if q_norm == 0 or d_norm == 0:
        return 0.0
    return dot / math.sqrt(q_norm * d_norm)


def query_flat(
    question: str,
    docs: list[Document],
    client: FireworksChatClient,
    *,
    top_k: int,
) -> dict[str, Any]:
    retriever = FlatRetriever(docs)
    chunks = retriever.search(question, top_k=top_k)
    context = "\n\n".join(
        f"[{item['doc_id']}] {item['text']}" for item in chunks
    )
    answer = answer_from_context(
        question,
        context,
        client,
        mode="Flat RAG",
        label="answer:flat",
    )
    return {"mode": "flat", "context_items": chunks, "answer": answer}


def query_graph(
    question: str,
    graph: nx.MultiDiGraph,
    client: FireworksChatClient,
    *,
    hops: int,
    top_k: int,
) -> dict[str, Any]:
    focus_entities = extract_focus_entities(question, client)
    seeds = match_graph_nodes(graph, question, focus_entities, top_k=top_k)
    facts = collect_graph_facts(graph, seeds, hops=hops, max_facts=45)
    context = textualize_facts(facts)
    if not context:
        context = "No graph facts matched the question."
    answer = answer_from_context(
        question,
        context,
        client,
        mode="GraphRAG",
        label="answer:graph",
    )
    return {
        "mode": "graph",
        "focus_entities": focus_entities,
        "seed_nodes": seeds,
        "facts": facts,
        "answer": answer,
    }


def extract_focus_entities(question: str, client: FireworksChatClient) -> list[str]:
    system = "Extract search entities from a user question. Return JSON only."
    user = f"""
Return a JSON object with this schema:
{{"entities": ["Tesla", "Q1 2024", "EV sales"]}}

Question:
{question}
""".strip()
    try:
        raw = client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            expect_json=True,
            max_tokens=env_int("QUERY_ENTITY_MAX_TOKENS", 256),
            label="extract_query_entities",
        )
        parsed = parse_json_object(raw)
        entities = [str(item).strip() for item in coerce_list(parsed.get("entities"))]
        return [item for item in entities if item]
    except Exception:
        return fallback_query_entities(question)


def fallback_query_entities(question: str) -> list[str]:
    capitalized = re.findall(r"\b[A-Z][A-Za-z0-9&.\-]+(?:\s+[A-Z][A-Za-z0-9&.\-]+)*", question)
    tokens = [token for token in tokenize(question) if len(token) > 3]
    return capitalized + tokens[:6]


def match_graph_nodes(
    graph: nx.MultiDiGraph,
    question: str,
    focus_entities: list[str],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    query_terms = set(tokenize(question + " " + " ".join(focus_entities)))
    candidates: list[tuple[float, str]] = []
    for node_id, attrs in graph.nodes(data=True):
        label = str(attrs.get("label", node_id))
        label_terms = set(tokenize(label))
        score = 0.0
        norm_label = normalize_name(label)
        for entity in focus_entities:
            norm_entity = normalize_name(entity)
            if norm_entity and norm_entity == norm_label:
                score = max(score, 5.0)
            elif norm_entity and (norm_entity in norm_label or norm_label in norm_entity):
                score = max(score, 3.5)
        if query_terms and label_terms:
            score += len(query_terms & label_terms) / max(1, len(label_terms))
        if score > 0:
            candidates.append((score, node_id))
    candidates.sort(reverse=True)
    return [
        {
            "node_id": node_id,
            "label": graph.nodes[node_id].get("label", node_id),
            "score": round(score, 3),
        }
        for score, node_id in candidates[:top_k]
    ]


def collect_graph_facts(
    graph: nx.MultiDiGraph,
    seeds: list[dict[str, Any]],
    *,
    hops: int,
    max_facts: int,
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, int]] = set()
    queue: deque[tuple[str, int]] = deque((seed["node_id"], 0) for seed in seeds)
    visited = {seed["node_id"]: 0 for seed in seeds}
    while queue and len(facts) < max_facts:
        node, distance = queue.popleft()
        if distance >= hops:
            continue
        for source, target, key, attrs in graph.out_edges(node, keys=True, data=True):
            if (source, target, key) not in seen_edges:
                seen_edges.add((source, target, key))
                facts.append(edge_fact(graph, source, target, attrs, direction="out"))
            if target not in visited or visited[target] > distance + 1:
                visited[target] = distance + 1
                queue.append((target, distance + 1))
            if len(facts) >= max_facts:
                break
        if len(facts) >= max_facts:
            break
        for source, target, key, attrs in graph.in_edges(node, keys=True, data=True):
            if (source, target, key) not in seen_edges:
                seen_edges.add((source, target, key))
                facts.append(edge_fact(graph, source, target, attrs, direction="in"))
            if source not in visited or visited[source] > distance + 1:
                visited[source] = distance + 1
                queue.append((source, distance + 1))
            if len(facts) >= max_facts:
                break
    return facts


def edge_fact(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    attrs: dict[str, Any],
    *,
    direction: str,
) -> dict[str, Any]:
    return {
        "source": graph.nodes[source].get("label", source),
        "predicate": attrs.get("predicate", "RELATED_TO"),
        "target": graph.nodes[target].get("label", target),
        "evidence": attrs.get("evidence", ""),
        "doc_id": attrs.get("doc_id", ""),
        "title": attrs.get("title", ""),
        "link": attrs.get("link", ""),
        "confidence": attrs.get("confidence", ""),
        "direction": direction,
    }


def textualize_facts(facts: list[dict[str, Any]]) -> str:
    lines = []
    for fact in facts:
        evidence = f" Evidence: {fact['evidence']}" if fact.get("evidence") else ""
        lines.append(
            f"[{fact.get('doc_id')}] {fact['source']} --{fact['predicate']}--> "
            f"{fact['target']}.{evidence}"
        )
    return "\n".join(lines)


def answer_from_context(
    question: str,
    context: str,
    client: FireworksChatClient,
    *,
    mode: str,
    label: str,
) -> str:
    context = context[: env_int("MAX_ANSWER_CONTEXT_CHARS", 18000)]
    system = (
        "You are a careful RAG assistant. Answer in Vietnamese. "
        "Use only the provided context. Cite sources using [doc_id]. "
        "If the context is insufficient, say so clearly."
    )
    user = f"""
Mode: {mode}

Question:
{question}

Context:
{context}

Answer in Vietnamese with concise evidence-backed reasoning.
""".strip()
    min_chars = env_int("MIN_ANSWER_CHARS", 80)
    retries = env_int("ANSWER_RETRIES", 1)
    answer = ""
    for attempt in range(1, retries + 2):
        attempt_user = user
        if attempt > 1:
            attempt_user += (
                "\n\nPrevious answer was too short or incomplete. "
                "Answer fully with citations from the context."
            )
        answer = client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": attempt_user},
            ],
            max_tokens=env_int("ANSWER_MAX_TOKENS", 1200),
            label=f"{label}:attempt_{attempt}",
        )
        if len(answer.strip()) >= min_chars:
            return answer
        print(
            f"[answer] short answer for {label} on attempt {attempt}; "
            f"length={len(answer.strip())}"
        )
    return answer


def load_questions(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        questions = questions[:limit]
    return questions


def evaluate(
    *,
    dataset_dir: Path,
    outputs_dir: Path,
    limit_docs: int | None,
    limit_questions: int | None,
    judge: bool,
    client: FireworksChatClient,
) -> list[dict[str, Any]]:
    docs = read_documents(dataset_dir, limit_docs)
    graph = load_graph(outputs_dir)
    questions = load_questions(ROOT / "benchmark_questions.json", limit_questions)
    existing_by_id = load_existing_evaluation(outputs_dir) if env_bool("EVALUATION_RESUME", True) else {}
    results = []
    for idx, item in enumerate(questions, start=1):
        question = item["question"]
        existing = existing_by_id.get(item["id"])
        if is_complete_evaluation(existing, judge):
            if existing:
                print(f"[eval] {idx}/{len(questions)} {item['id']}: cached")
                results.append(existing)
                write_evaluation_artifacts(results, outputs_dir)
                continue

        print(f"[eval] {idx}/{len(questions)} {item['id']}: {question}")
        flat = query_flat(question, docs, client, top_k=6)
        graph_result = query_graph(
            question,
            graph,
            client,
            hops=env_int("GRAPH_HOPS", 2),
            top_k=6,
        )
        judged = (
            judge_pair(
                question,
                flat["answer"],
                graph_result["answer"],
                client,
                outputs_dir=outputs_dir,
                question_id=item["id"],
            )
            if judge
            else {}
        )
        results.append(
            {
                "id": item["id"],
                "question": question,
                "flat_answer": flat["answer"],
                "graph_answer": graph_result["answer"],
                "graph_seed_nodes": graph_result.get("seed_nodes", []),
                "judge": judged,
            }
        )
        write_evaluation_artifacts(results, outputs_dir)
    return results


def is_complete_evaluation(existing: dict[str, Any] | None, judge: bool) -> bool:
    if not existing:
        return False
    min_chars = env_int("MIN_ANSWER_CHARS", 80)
    flat_ok = len(str(existing.get("flat_answer", "")).strip()) >= min_chars
    graph_ok = len(str(existing.get("graph_answer", "")).strip()) >= min_chars
    judge_ok = bool(existing.get("judge")) if judge else True
    return flat_ok and graph_ok and judge_ok


def load_existing_evaluation(outputs_dir: Path) -> dict[str, dict[str, Any]]:
    path = outputs_dir / "evaluation_results.json"
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(rows, list):
        return {}
    return {
        str(item.get("id")): item
        for item in rows
        if isinstance(item, dict) and item.get("id")
    }


def judge_pair(
    question: str,
    flat_answer: str,
    graph_answer: str,
    client: FireworksChatClient,
    *,
    outputs_dir: Path | None = None,
    question_id: str = "unknown",
) -> dict[str, Any]:
    system = (
        "Return only a minified JSON object. Start with { and end with }. "
        "No prose, no markdown, no reasoning."
    )
    base_user = f"""
Question:
{question}

Flat RAG answer:
{flat_answer[:3500]}

GraphRAG answer:
{graph_answer[:3500]}

Judge grounding, specificity, and citation quality. Return exactly this JSON schema:
{{
  "winner": "graph|flat|tie",
  "flat_grounded": true,
  "graph_grounded": true,
  "flat_risk": "short note",
  "graph_advantage": "short note"
}}
""".strip()
    retries = env_int("JUDGE_PARSE_RETRIES", 1)
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        user = base_user
        if attempt > 1:
            user += "\n\nPrevious output was invalid. Return ONLY JSON, no explanation."
        raw = client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            expect_json=True,
            max_tokens=env_int("JUDGE_MAX_TOKENS", 700),
            label=f"judge:{question_id}:attempt_{attempt}",
        )
        save_judge_raw_response(outputs_dir, question_id, attempt, raw)
        try:
            return normalize_judge_result(parse_json_object(raw))
        except Exception as exc:
            last_error = exc
            print(f"[eval] invalid judge JSON for {question_id} on attempt {attempt}: {exc}")

    fallback = heuristic_judge(question, flat_answer, graph_answer)
    fallback["judge_warning"] = f"LLM judge JSON parse failed: {last_error}"
    return fallback


def save_judge_raw_response(
    outputs_dir: Path | None,
    question_id: str,
    attempt: int,
    raw: str,
) -> None:
    if outputs_dir is None:
        return
    raw_dir = outputs_dir / "cache" / "raw_judge_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{question_id}_attempt_{attempt}.txt").write_text(
        raw or "",
        encoding="utf-8",
    )


def normalize_judge_result(parsed: dict[str, Any]) -> dict[str, Any]:
    winner = str(parsed.get("winner", "tie")).strip().lower()
    if winner not in {"graph", "flat", "tie"}:
        winner = "tie"
    return {
        "winner": winner,
        "flat_grounded": bool(parsed.get("flat_grounded", False)),
        "graph_grounded": bool(parsed.get("graph_grounded", False)),
        "flat_risk": str(parsed.get("flat_risk", ""))[:240],
        "graph_advantage": str(parsed.get("graph_advantage", ""))[:240],
    }


def heuristic_judge(question: str, flat_answer: str, graph_answer: str) -> dict[str, Any]:
    flat_citations = len(re.findall(r"\[doc_\d+\]", flat_answer))
    graph_citations = len(re.findall(r"\[doc_\d+\]", graph_answer))
    flat_terms = len(set(tokenize(question)) & set(tokenize(flat_answer)))
    graph_terms = len(set(tokenize(question)) & set(tokenize(graph_answer)))
    flat_score = flat_citations * 2 + flat_terms + min(len(flat_answer), 1200) / 400
    graph_score = graph_citations * 2 + graph_terms + min(len(graph_answer), 1200) / 400
    if graph_score > flat_score + 1:
        winner = "graph"
    elif flat_score > graph_score + 1:
        winner = "flat"
    else:
        winner = "tie"
    return {
        "winner": winner,
        "flat_grounded": flat_citations > 0,
        "graph_grounded": graph_citations > 0,
        "flat_risk": "Heuristic judge used because LLM judge did not return JSON.",
        "graph_advantage": "Compared citation count, answer coverage, and overlap with question terms.",
    }


def write_evaluation_artifacts(results: list[dict[str, Any]], outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "evaluation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = [
        "| ID | Question | Winner | Flat Risk | Graph Advantage |",
        "|---|---|---|---|---|",
    ]
    for item in results:
        judge = item.get("judge") or {}
        rows.append(
            "| {id} | {question} | {winner} | {flat_risk} | {graph_advantage} |".format(
                id=item["id"],
                question=escape_md(item["question"]),
                winner=escape_md(str(judge.get("winner", "not judged"))),
                flat_risk=escape_md(str(judge.get("flat_risk", ""))),
                graph_advantage=escape_md(str(judge.get("graph_advantage", ""))),
            )
        )
    (outputs_dir / "benchmark_table.md").write_text("\n".join(rows), encoding="utf-8")


def escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def write_report(
    *,
    outputs_dir: Path,
    dataset_dir: Path,
    limit_docs: int | None,
    results: list[dict[str, Any]] | None,
    client: FireworksChatClient,
    elapsed_seconds: float,
) -> None:
    graph_summary_path = outputs_dir / "graph_summary.json"
    graph_summary = {}
    if graph_summary_path.exists():
        graph_summary = json.loads(graph_summary_path.read_text(encoding="utf-8"))

    winner_counts = Counter()
    hallucination_cases = []
    for item in results or []:
        judge = item.get("judge") or {}
        winner = str(judge.get("winner", "not judged"))
        winner_counts[winner] += 1
        if winner == "graph":
            hallucination_cases.append(item)

    calls = client.calls
    usage = aggregate_usage(calls)
    docs_count = len(read_documents(dataset_dir, limit_docs))
    lines = [
        "# Báo cáo GraphRAG Lab",
        "",
        "## Cấu hình",
        "",
        f"- LLM: `{client.model}` qua Fireworks API",
        "- Graph framework: NetworkX",
        f"- Dataset documents: {docs_count}",
        f"- Runtime this run: {elapsed_seconds:.1f} seconds",
        "",
        "## Indexing và Graph Construction",
        "",
        f"- Nodes: {graph_summary.get('nodes', 'N/A')}",
        f"- Edges/triples: {graph_summary.get('edges', 'N/A')}",
        f"- Cached extraction files: `outputs/cache/extractions/`",
        f"- Graph file: `outputs/graph.graphml`",
        f"- Graph image: `{graph_summary.get('graph_image', 'outputs/knowledge_graph.svg')}`",
        "",
        "## Querying",
        "",
        "- Flat RAG: truy xuất chunk văn bản bằng TF-IDF local, sau đó gửi context cho LLM.",
        "- GraphRAG: LLM trích xuất focus entities từ câu hỏi, NetworkX tìm node gần nhất, duyệt 2-hop, textualize triples, sau đó gửi context graph cho LLM.",
        "",
        "## Evaluation",
        "",
        f"- Questions evaluated: {len(results or [])}",
        f"- Winner counts: `{dict(winner_counts)}`",
        "- Bảng benchmark: `outputs/benchmark_table.md`",
        "",
        "## Các trường hợp GraphRAG tốt hơn Flat RAG",
        "",
    ]
    if hallucination_cases:
        for item in hallucination_cases[:8]:
            judge = item.get("judge") or {}
            lines.append(
                f"- **{item['id']}**: {item['question']} "
                f"Graph advantage: {judge.get('graph_advantage', '')}"
            )
    else:
        lines.append("- Chưa có judge hoặc chưa ghi nhận trường hợp GraphRAG thắng rõ ràng.")

    lines.extend(
        [
            "",
            "## Chi phí và token usage",
            "",
            f"- LLM calls this run: {len(calls)}",
            f"- Prompt tokens: {usage.get('prompt_tokens', 0)}",
            f"- Completion tokens: {usage.get('completion_tokens', 0)}",
            f"- Total tokens: {usage.get('total_tokens', 0)}",
            "",
            "Lưu ý: nếu một số extraction lấy từ cache, token của các lần chạy trước sẽ không được cộng vào số liệu trên.",
        ]
    )
    (outputs_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate_usage(calls: list[dict[str, Any]]) -> dict[str, int]:
    total = Counter()
    for call in calls:
        usage = call.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key, 0)
            if isinstance(value, int):
                total[key] += value
    return dict(total)


def run_smoke_test(dataset_dir: Path) -> None:
    docs = read_documents(dataset_dir, limit=3)
    retriever = FlatRetriever(docs)
    hits = retriever.search("Tesla EV sales Q1 2024", top_k=3)
    fake_extractions = [
        {
            "entities": [
                {"name": "Tesla", "type": "Company", "description": "EV market leader"},
                {"name": "US EV market", "type": "Market", "description": "Electric vehicle market in the United States"},
            ],
            "triples": [
                {
                    "subject": "Tesla",
                    "predicate": "INFLUENCED",
                    "object": "US EV market",
                    "evidence": "Smoke-test synthetic edge.",
                    "confidence": 1.0,
                    "doc_id": "smoke",
                }
            ],
        }
    ]
    graph = build_graph(fake_extractions)
    seeds = match_graph_nodes(graph, "Tesla ảnh hưởng thị trường EV Mỹ thế nào?", ["Tesla"], top_k=3)
    facts = collect_graph_facts(graph, seeds, hops=2, max_facts=5)
    print("[smoke] documents:", len(docs))
    print("[smoke] flat hits:", [item["chunk_id"] for item in hits])
    print("[smoke] graph nodes:", graph.number_of_nodes())
    print("[smoke] graph facts:", textualize_facts(facts))
    print("[smoke] OK")


def main() -> int:
    parser = argparse.ArgumentParser(description="NetworkX GraphRAG lab with Fireworks DeepSeek Pro")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
        subparser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)

    index_parser = subparsers.add_parser("index", help="Extract triples and build NetworkX graph")
    add_common(index_parser)
    index_parser.add_argument("--limit-docs", type=int, default=None)
    index_parser.add_argument("--force", action="store_true", help="Ignore extraction cache")

    query_parser = subparsers.add_parser("query", help="Ask one question")
    add_common(query_parser)
    query_parser.add_argument("question")
    query_parser.add_argument("--mode", choices=["flat", "graph"], default="graph")
    query_parser.add_argument("--limit-docs", type=int, default=None)
    query_parser.add_argument("--top-k", type=int, default=6)
    query_parser.add_argument("--hops", type=int, default=env_int("GRAPH_HOPS", 2))

    eval_parser = subparsers.add_parser("evaluate", help="Run benchmark questions")
    add_common(eval_parser)
    eval_parser.add_argument("--limit-docs", type=int, default=None)
    eval_parser.add_argument("--limit-questions", type=int, default=None)
    eval_parser.add_argument("--skip-judge", action="store_true")

    all_parser = subparsers.add_parser("all", help="Run index, benchmark, and report")
    add_common(all_parser)
    all_parser.add_argument("--limit-docs", type=int, default=None)
    all_parser.add_argument("--limit-questions", type=int, default=None)
    all_parser.add_argument("--force", action="store_true")
    all_parser.add_argument("--skip-judge", action="store_true")

    smoke_parser = subparsers.add_parser("smoke-test", help="Run local checks without API calls")
    smoke_parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)

    args = parser.parse_args()
    try:
        if args.command == "smoke-test":
            run_smoke_test(args.dataset_dir)
            return 0

        client = FireworksChatClient()
        started = time.perf_counter()
        if args.command == "index":
            index_corpus(
                dataset_dir=args.dataset_dir,
                outputs_dir=args.outputs_dir,
                limit_docs=args.limit_docs,
                force=args.force,
                client=client,
            )
        elif args.command == "query":
            docs = read_documents(args.dataset_dir, args.limit_docs)
            if args.mode == "flat":
                result = query_flat(args.question, docs, client, top_k=args.top_k)
            else:
                graph = load_graph(args.outputs_dir)
                result = query_graph(
                    args.question,
                    graph,
                    client,
                    hops=args.hops,
                    top_k=args.top_k,
                )
            print(result["answer"])
        elif args.command == "evaluate":
            results = evaluate(
                dataset_dir=args.dataset_dir,
                outputs_dir=args.outputs_dir,
                limit_docs=args.limit_docs,
                limit_questions=args.limit_questions,
                judge=not args.skip_judge,
                client=client,
            )
            write_report(
                outputs_dir=args.outputs_dir,
                dataset_dir=args.dataset_dir,
                limit_docs=args.limit_docs,
                results=results,
                client=client,
                elapsed_seconds=time.perf_counter() - started,
            )
        elif args.command == "all":
            index_corpus(
                dataset_dir=args.dataset_dir,
                outputs_dir=args.outputs_dir,
                limit_docs=args.limit_docs,
                force=args.force,
                client=client,
            )
            results = evaluate(
                dataset_dir=args.dataset_dir,
                outputs_dir=args.outputs_dir,
                limit_docs=args.limit_docs,
                limit_questions=args.limit_questions,
                judge=not args.skip_judge,
                client=client,
            )
            write_report(
                outputs_dir=args.outputs_dir,
                dataset_dir=args.dataset_dir,
                limit_docs=args.limit_docs,
                results=results,
                client=client,
                elapsed_seconds=time.perf_counter() - started,
            )
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
