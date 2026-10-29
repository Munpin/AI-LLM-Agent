"""
도시 정책 RAG 서버 (Google Cloud Run / FastAPI)

환경 변수
  GOOGLE_API_KEY       (필수) 구글 AI API 키 - Gemini 답변/질의 확장
  OPENAI_API_KEY       (필수) OpenAI 임베딩
  SUPABASE_URL         (필수) https://xxxx.supabase.co
  SUPABASE_SERVICE_KEY (필수) service_role 키 (서버에서만 사용)
  LLM_MODEL            (선택) 기본 gemini-3.5-flash-lite
  EMBEDDING_MODEL      (선택) 기본 text-embedding-3-small (1536차원)
  ALLOWED_ORIGINS      (선택) CORS 허용 출처, 쉼표 구분. 기본 *
  MAX_PDF_MB / CHUNK_SIZE / CHUNK_OVERLAP / MIN_SIMILARITY (선택)

엔드포인트
  GET  /health[?deep=true]        연결 상태 점검
  POST /api/pdf/extract           PDF -> 텍스트(JSON)
  POST /api/index[?stream=false]  PDF -> 청크 -> 임베딩 -> Supabase (진행률 NDJSON 스트림)
  POST /api/chat[?stream=false]   질의 확장 -> 검색 -> 답변 (토큰 NDJSON 스트림)
"""
import asyncio
import io
import json
import logging
import os
import time
from functools import lru_cache
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.callbacks import AsyncCallbackHandler, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader
from supabase import Client, create_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag")

# ─────────────────────────────────────────────────────────────
# 1. 설정
# ─────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
MAX_PDF_MB = int(os.getenv("MAX_PDF_MB", "25"))          # Cloud Run 요청 한도(32MB) 이하로 유지
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.0"))
EMBED_BATCH = 50                                          # 임베딩/삽입 배치 크기
MAX_HISTORY_MESSAGES = 12                                 # 질의 확장에 쓸 최근 대화 수

# 기능별 필요한 환경변수 (누락 시 어떤 이름이 빠졌는지 정확히 알려주기 위함)
REQUIRED_ENV = {
    "google": ["GOOGLE_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "supabase": ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"],
}

# ─────────────────────────────────────────────────────────────
# 2. 오류 모델: 사용자가 원인과 해결책을 바로 알 수 있도록 code/message/hint/detail 제공
# ─────────────────────────────────────────────────────────────
class AppError(Exception):
    def __init__(self, code: str, message: str, hint: str = "", status: int = 400, detail: str = ""):
        super().__init__(message)
        self.code, self.message, self.hint, self.status, self.detail = code, message, hint, status, detail

    def to_event(self) -> dict:
        return {"type": "error", "code": self.code, "message": self.message,
                "hint": self.hint, "detail": self.detail, "status": self.status}


def translate_exception(exc: Exception) -> AppError:
    """라이브러리 예외(OpenAI/Google/Supabase/네트워크)를 이해하기 쉬운 AppError로 변환."""
    if isinstance(exc, AppError):
        return exc
    name, module, text = type(exc).__name__, type(exc).__module__ or "", str(exc)
    low = text.lower()
    detail = f"{name}: {text[:600]}"

    # --- OpenAI (임베딩) ---
    if module.startswith("openai"):
        if name == "AuthenticationError":
            return AppError("OPENAI_AUTH", "OpenAI API 키가 올바르지 않습니다.",
                            "OPENAI_API_KEY 값을 확인하세요(공백/줄바꿈 포함 여부, 폐기된 키 여부).", 502, detail)
        if name == "RateLimitError":
            if "insufficient_quota" in low:
                return AppError("OPENAI_QUOTA", "OpenAI 크레딧(쿼터)이 소진되었습니다.",
                                "OpenAI 결제/크레딧 상태를 확인하세요.", 502, detail)
            return AppError("OPENAI_RATE_LIMIT", "OpenAI 요청 한도를 초과했습니다.",
                            "잠시 후 다시 시도하세요.", 429, detail)
        if name in ("APIConnectionError", "APITimeoutError"):
            return AppError("OPENAI_NETWORK", "OpenAI 서버에 연결하지 못했습니다.",
                            "Cloud Run의 외부 네트워크(VPC egress) 설정을 확인하세요.", 502, detail)
        return AppError("OPENAI_ERROR", "OpenAI 임베딩 호출 중 오류가 발생했습니다.", "", 502, detail)

    # --- Supabase / PostgREST ---
    code = str(getattr(exc, "code", "") or "")
    if "postgrest" in module or "supabase" in module or name == "APIError":
        if code == "PGRST202" or "could not find the function" in low:
            return AppError("DB_FUNCTION_MISSING", "Supabase에 match_documents 함수가 없습니다.",
                            "supabase_schema.sql을 SQL Editor에서 실행하세요.", 500, detail)
        if code == "42P01" or ("relation" in low and "does not exist" in low) or "could not find the table" in low:
            return AppError("DB_TABLE_MISSING", "Supabase에 documents 테이블이 없습니다.",
                            "supabase_schema.sql을 SQL Editor에서 실행하세요.", 500, detail)
        if "dimension" in low:
            return AppError("DB_DIMENSION_MISMATCH", "임베딩 차원이 DB 컬럼(vector(1536))과 다릅니다.",
                            f"EMBEDDING_MODEL({EMBEDDING_MODEL})의 차원과 SQL의 vector(N)을 일치시키세요.", 500, detail)
        if "jwt" in low or "invalid api key" in low or code in ("401", "PGRST301"):
            return AppError("DB_AUTH", "Supabase 인증에 실패했습니다.",
                            "SUPABASE_SERVICE_KEY는 anon 키가 아니라 service_role 키여야 합니다.", 502, detail)
        return AppError("DB_ERROR", "Supabase 쿼리 중 오류가 발생했습니다.", "", 502, detail)
    if module.startswith(("httpx", "httpcore")):
        return AppError("DB_NETWORK", "Supabase 서버에 연결하지 못했습니다.",
                        "SUPABASE_URL 주소와 프로젝트 일시중지(paused) 여부를 확인하세요.", 502, detail)

    # --- Google AI (Gemini) ---
    if "google" in module or "genai" in module or "gemini" in low or "generativelanguage" in low:
        if "api key not valid" in low or "api_key_invalid" in low or "unauthenticated" in low:
            return AppError("GOOGLE_AUTH", "구글 AI API 키가 올바르지 않습니다.",
                            "GOOGLE_API_KEY를 확인하세요(AI Studio에서 발급한 키).", 502, detail)
        if "not found" in low or "404" in low:
            return AppError("MODEL_NOT_FOUND", f"모델 '{LLM_MODEL}'을(를) 찾을 수 없습니다.",
                            "LLM_MODEL 환경변수의 모델명 철자와 사용 가능 여부를 확인하세요.", 502, detail)
        if "resource_exhausted" in low or "429" in low or "quota" in low:
            return AppError("GOOGLE_QUOTA", "구글 AI API 호출 한도를 초과했습니다.",
                            "잠시 후 재시도하거나 요금제/쿼터를 확인하세요.", 429, detail)
        if "permission_denied" in low or "403" in low:
            return AppError("GOOGLE_PERMISSION", "구글 AI API 권한이 없습니다.",
                            "키의 API 제한 설정과 프로젝트의 Generative Language API 활성화를 확인하세요.", 502, detail)
        if "safety" in low or "blocked" in low:
            return AppError("GOOGLE_BLOCKED", "안전 정책으로 응답이 차단되었습니다.", "질문 표현을 바꿔 보세요.", 422, detail)
        if "503" in low or "unavailable" in low or "overloaded" in low:
            return AppError("GOOGLE_UNAVAILABLE", "구글 AI 서버가 일시적으로 과부하입니다.", "잠시 후 재시도하세요.", 503, detail)
        return AppError("GOOGLE_ERROR", "구글 AI 호출 중 오류가 발생했습니다.", "", 502, detail)

    return AppError("UNEXPECTED_ERROR", "예상하지 못한 서버 오류가 발생했습니다.",
                    "Cloud Run 로그에서 상세 스택트레이스를 확인하세요.", 500, detail)


def missing_env(groups: list[str]) -> list[str]:
    return [n for g in groups for n in REQUIRED_ENV[g] if not os.getenv(n)]


def ensure_configured(groups: list[str]) -> None:
    miss = missing_env(groups)
    if miss:
        raise AppError("ENV_MISSING", f"필수 환경변수가 설정되지 않았습니다: {', '.join(miss)}",
                       "Cloud Run 서비스의 환경변수/시크릿에 추가한 뒤 새 리비전을 배포하세요.", 503)

# ─────────────────────────────────────────────────────────────
# 3. 클라이언트 객체 (지연 생성 + 캐시)
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_supabase() -> Client:
    ensure_configured(["supabase"])
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    ensure_configured(["openai"])
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=os.environ["OPENAI_API_KEY"])


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    ensure_configured(["google"])
    return ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=os.environ["GOOGLE_API_KEY"])

# ─────────────────────────────────────────────────────────────
# 4. PDF -> 텍스트 -> 청크
# ─────────────────────────────────────────────────────────────
def extract_pdf_pages(data: bytes) -> list[dict]:
    """PDF 바이트에서 페이지별 텍스트를 추출한다. (CPU 작업이라 스레드에서 실행)"""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise AppError("PDF_ENCRYPTED", "암호가 걸린 PDF입니다.", "암호를 해제한 파일을 업로드하세요.", 422)
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").replace("\x00", "").strip()
            except Exception as e:  # 한 페이지 실패가 전체 실패가 되지 않도록
                log.warning("페이지 %d 추출 실패: %s", i, e)
                text = ""
            pages.append({"page": i, "text": text})
    except AppError:
        raise
    except Exception as e:
        raise AppError("PDF_UNREADABLE", "PDF를 읽을 수 없습니다. 손상된 파일일 수 있습니다.",
                       "다른 PDF로 다시 시도하거나 파일을 다시 내보내기 하세요.", 422, f"{type(e).__name__}: {e}")
    if not any(p["text"] for p in pages):
        raise AppError("PDF_NO_TEXT", "PDF에서 텍스트를 찾지 못했습니다.",
                       "스캔 이미지 PDF는 OCR을 먼저 적용해야 합니다.", 422)
    return pages


def split_pages(pages: list[dict]) -> list[dict]:
    """페이지별로 청크를 나눠 {page, chunk_index, content} 목록을 만든다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""])
    chunks, idx = [], 0
    for p in pages:
        for piece in splitter.split_text(p["text"]) if p["text"] else []:
            if piece.strip():
                chunks.append({"page": p["page"], "chunk_index": idx, "content": piece})
                idx += 1
    return chunks


async def read_pdf_upload(file: UploadFile) -> tuple[str, bytes]:
    name = file.filename or "unnamed.pdf"
    if not name.lower().endswith(".pdf"):
        raise AppError("FILE_NOT_PDF", f"PDF 파일만 업로드할 수 있습니다. (받은 파일: {name})", "확장자가 .pdf인 파일을 선택하세요.", 415)
    data = await file.read()
    if not data:
        raise AppError("FILE_EMPTY", "업로드된 파일이 비어 있습니다.", "", 400)
    if len(data) > MAX_PDF_MB * 1024 * 1024:
        raise AppError("FILE_TOO_LARGE", f"파일이 너무 큽니다. ({len(data)/1048576:.1f}MB / 최대 {MAX_PDF_MB}MB)",
                       "PDF를 나누어 업로드하세요.", 413)
    if not data.startswith(b"%PDF"):
        raise AppError("FILE_NOT_PDF", "확장자는 .pdf지만 실제 PDF 형식이 아닙니다.", "파일을 다시 확인하세요.", 415)
    return name, data

# ─────────────────────────────────────────────────────────────
# 5. 벡터 DB 리트리버 (Supabase match_documents RPC)
# ─────────────────────────────────────────────────────────────
class SupabaseRetriever(BaseRetriever):
    k: int = 5
    filter: dict = Field(default_factory=dict)
    min_similarity: float = 0.0

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        vec = get_embeddings().embed_query(query)                     # 1) 질의 임베딩
        res = get_supabase().rpc("match_documents", {                # 2) 유사도 검색
            "query_embedding": vec, "match_count": self.k, "filter": self.filter}).execute()
        docs = []
        for row in res.data or []:
            if row["similarity"] < self.min_similarity:
                continue
            meta = dict(row.get("metadata") or {})
            meta.update(id=row["id"], doc_name=row["doc_name"], page=row["page"],
                        chunk_index=row["chunk_index"], similarity=row["similarity"])
            docs.append(Document(page_content=row["content"], metadata=meta))
        return docs

# ─────────────────────────────────────────────────────────────
# 6. 프롬프트 / 체인
# ─────────────────────────────────────────────────────────────
AUGMENT_SYSTEM = (
    "너는 대화 맥락을 이해하는 질의 변환기야. "
    "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한 문장의 명료한 질문으로 변환하라. "
    "대명사나 '이', '저', '그'와 같은 지시 표현은 대화 내용에서 찾은 명확한 명사로 표현하라. "
    "질문에 답하지 말고, 변환된 질문 한 문장만 출력하라.")

ANSWER_SYSTEM = (
    "너는 문서에 기반해 답변하는 도시 정책 전문가야.\n"
    "규칙: 1) 반드시 [참고 문서]에 있는 내용만 근거로 답한다. "
    "2) 근거가 된 문장 끝에 [출처 번호]를 붙인다. 예: [출처 1] "
    "3) 문서에 근거가 없으면 '제공된 문서에서 해당 내용을 찾을 수 없습니다.'라고 답한다. "
    "4) 마크다운 없이 간결한 일반 텍스트로 답한다.")

# (1) 시스템 메시지 + (2) 대화 기록 자리(MessagesPlaceholder) + 현재 질문
augment_prompt = ChatPromptTemplate.from_messages([
    ("system", AUGMENT_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "[참고 문서]\n{context}\n\n[질문]\n{question}"),
])


def build_augmentation_chain():
    """(3) 프롬프트 | LLM | StrOutputParser 를 파이프라인으로 연결."""
    return augment_prompt | get_llm() | StrOutputParser()


def build_answer_chain():
    return answer_prompt | get_llm() | StrOutputParser()


def to_messages(history: list["ChatMessage"]) -> list:
    out = []
    for m in history[-MAX_HISTORY_MESSAGES:]:
        out.append(HumanMessage(content=m.content) if m.role.lower() in ("user", "human")
                   else AIMessage(content=m.content))
    return out


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[출처 {i}] ({d.metadata.get('doc_name')}, {d.metadata.get('page')}쪽)\n{d.page_content}"
        for i, d in enumerate(docs, start=1))

# ─────────────────────────────────────────────────────────────
# 7. StreamHandler: LLM이 토큰을 만들 때마다 큐에 넣어 클라이언트로 흘려보냄
# ─────────────────────────────────────────────────────────────
_DONE = object()


class StreamHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.parts: list[str] = []

    async def on_chat_model_start(self, serialized, messages, **kwargs):
        await self.queue.put({"type": "status", "message": "답변 생성 중…"})

    async def on_llm_new_token(self, token, **kwargs):
        if isinstance(token, list):  # 일부 모델은 블록 리스트로 반환
            token = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in token)
        if token:
            self.parts.append(token)
            await self.queue.put({"type": "token", "text": token})

    async def on_llm_error(self, error, **kwargs):
        log.error("LLM 스트리밍 오류: %s", error)

    @property
    def text(self) -> str:
        return "".join(self.parts)

# ─────────────────────────────────────────────────────────────
# 8. 파이프라인 (이벤트를 yield 하는 비동기 제너레이터)
# ─────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(5, ge=1, le=20)
    doc_name: Optional[str] = None


async def index_pipeline(filename: str, data: bytes, replace: bool) -> AsyncIterator[dict]:
    t0, inserted = time.perf_counter(), False
    try:
        yield {"type": "progress", "stage": "extract", "percent": 2, "message": "PDF에서 텍스트 추출 중…"}
        pages = await asyncio.to_thread(extract_pdf_pages, data)
        yield {"type": "progress", "stage": "extract", "percent": 10, "message": f"{len(pages)}쪽 텍스트 추출 완료"}

        chunks = await asyncio.to_thread(split_pages, pages)
        if not chunks:
            raise AppError("NO_CHUNKS", "청크를 만들 수 있는 텍스트가 없습니다.", "", 422)
        yield {"type": "progress", "stage": "chunk", "percent": 15,
               "message": f"{len(chunks)}개 청크 생성 (크기 {CHUNK_SIZE}자, 겹침 {CHUNK_OVERLAP}자)"}

        sb = get_supabase()
        if replace:
            await asyncio.to_thread(lambda: sb.table("documents").delete().eq("doc_name", filename).execute())
            yield {"type": "progress", "stage": "cleanup", "percent": 17, "message": "같은 이름의 기존 데이터 삭제"}

        emb, total = get_embeddings(), len(chunks)
        for start in range(0, total, EMBED_BATCH):
            batch = chunks[start:start + EMBED_BATCH]
            vecs = await emb.aembed_documents([c["content"] for c in batch])
            rows = [{"doc_name": filename, "page": c["page"], "chunk_index": c["chunk_index"],
                     "content": c["content"], "embedding": v,
                     "metadata": {"doc_name": filename, "page": c["page"], "chunk_index": c["chunk_index"]}}
                    for c, v in zip(batch, vecs)]
            await asyncio.to_thread(lambda r=rows: sb.table("documents").insert(r).execute())
            inserted =
