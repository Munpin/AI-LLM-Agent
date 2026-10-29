# -*- coding: utf-8 -*-
"""
=====================================================================
 main.py  -  도시 정책 문서 RAG AI 서버 (Google Cloud Run / FastAPI)
=====================================================================

[전체 흐름]
  1) POST /extract          PDF(base64) → 페이지별 텍스트 추출 → 클라이언트 미리보기용 반환
  2) POST /ingest/start     PDF → 텍스트 → 청크 분할 → OpenAI 임베딩 → Supabase(pgvector) 저장
                            (백그라운드 스레드에서 실행, 진행 현황은 GET /ingest/status/{job_id})
  3) POST /query/start      질의 확장(Query Augmentation) → Supabase 리트리버 검색
                            → Gemini 답변 생성 (StreamHandler 가 토큰을 실시간으로 버퍼에 기록)
     GET  /query/poll/{id}  버퍼에 쌓인 토큰을 offset 기준으로 가져감
                            (Apps Script UrlFetchApp 은 스트리밍 수신이 불가능하므로 폴링으로 중계)
     POST /query/stream     NDJSON 실시간 스트리밍 (fetch 등으로 직접 연결하는 클라이언트용)
  4) GET  /health           서버/환경변수/Supabase 연결 상태 점검

[환경 변수]  ※ API 키는 코드에 절대 하드코딩하지 않음
  GOOGLE_API_KEY            Google AI(Gemini) API 키                  (필수)
  OPENAI_API_KEY            OpenAI 임베딩 API 키                      (필수)
  SUPABASE_URL              Supabase 프로젝트 URL                     (필수)
  SUPABASE_SERVICE_ROLE_KEY Supabase service_role 키 (서버 전용)      (필수)
  LLM_MODEL                 기본값 gemini-3.5-flash-lite
  EMBEDDING_MODEL           기본값 text-embedding-3-small (1536차원 → schema.sql 과 일치해야 함)
  SERVER_API_KEY            (선택) 설정 시 요청 헤더 X-API-Key 가 일치해야 호출 허용

[주의]
  작업(job) 상태를 인스턴스 메모리에 보관하므로 Cloud Run 배포 시
  --max-instances=1 (또는 --session-affinity) 와 --no-cpu-throttling 을 권장함.
  (응답을 반환한 뒤에도 백그라운드 스레드가 CPU 를 사용해야 하기 때문)
"""

import base64
import binascii
import hmac
import io
import json
import logging
import os
import queue
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# PDF 처리
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# LangChain 구성 요소
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import SupabaseVectorStore

# Supabase 클라이언트
from supabase import Client, create_client


# =====================================================================
# 0. 로깅 / 환경 변수
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rag-server")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SERVER_API_KEY = os.getenv("SERVER_API_KEY", "")

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "documents")          # schema.sql 의 테이블명
SUPABASE_QUERY_NAME = os.getenv("SUPABASE_QUERY_NAME", "match_documents")  # schema.sql 의 함수명

MAX_PDF_MB = int(os.getenv("MAX_PDF_MB", "20"))   # 업로드 허용 최대 크기(MB)
EMBED_BATCH_SIZE = 50                               # 임베딩/저장 배치 크기
MAX_HISTORY_TURNS = 10                              # 질의 확장에 사용할 최근 대화 메시지 수
JOB_TTL_SECONDS = 60 * 60                           # 작업 상태 보관 시간(1시간)

# 각 환경 변수 이름 ↔ 실제 값 (상태 점검/오류 메시지용)
REQUIRED_ENV = {
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY,
}

# 외부 서비스별로 확인해야 할 환경 변수 (오류 힌트에 사용)
SERVICE_ENV_HINT = {
    "OPENAI": "OPENAI_API_KEY",
    "GEMINI": "GOOGLE_API_KEY",
    "SUPABASE": "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY",
}


# =====================================================================
# 1. 오류 정의 - 모든 오류를 {code, message, hint} 형태로 통일
# =====================================================================
class RagError(Exception):
    """
    서버 전역에서 사용하는 구조화된 오류.
      code    : 기계가 읽을 수 있는 오류 코드 (예: PDF_NO_TEXT)
      message : 사용자에게 보여줄 한국어 설명 (무엇이 잘못되었는가)
      hint    : 해결 방법 (어떻게 고치면 되는가)
      status  : HTTP 상태 코드
    """

    def __init__(self, code: str, message: str, hint: str = "", status: int = 500):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


def require_env(names: List[str]) -> None:
    """필요한 환경 변수가 비어 있으면 어떤 변수가 빠졌는지 명시하여 오류를 발생시킨다."""
    missing = [n for n in names if not REQUIRED_ENV.get(n)]
    if missing:
        raise RagError(
            "CONFIG_MISSING",
            f"서버 환경 변수가 설정되지 않았습니다: {', '.join(missing)}",
            "Cloud Run 콘솔 > 서비스 > 새 버전 수정 > '변수 및 보안 비밀'에서 값을 등록한 뒤 재배포하세요.",
            500,
        )


def guess_service(exc: Exception, default: str) -> str:
    """예외 메시지/모듈명으로 어떤 외부 서비스에서 발생한 오류인지 추정한다."""
    text = (type(exc).__module__ + " " + str(exc)).lower()
    if "openai" in text:
        return "OPENAI"
    if "google" in text or "gemini" in text or "genai" in text:
        return "GEMINI"
    if "postgrest" in text or "supabase" in text or "pgrst" in text:
        return "SUPABASE"
    return default


def classify_external_error(exc: Exception, service: str) -> RagError:
    """
    OpenAI / Gemini / Supabase 에서 발생한 예외를 원인별로 분류해
    사용자가 바로 조치할 수 있는 세밀한 오류 메시지로 변환한다.
    """
    if isinstance(exc, RagError):
        return exc

    raw = str(exc)
    low = raw.lower()
    env_hint = SERVICE_ENV_HINT.get(service, "관련 환경 변수")
    short = raw[:300]  # 원문 오류는 너무 길 수 있으므로 앞부분만 첨부

    # --- 인증 실패 ---
    if any(s in low for s in ["401", "invalid api key", "incorrect api key",
                               "api key not valid", "api_key_invalid", "unauthorized",
                               "invalid jwt", "authentication"]):
        return RagError(f"{service}_AUTH_FAILED",
                        f"{service} 인증에 실패했습니다. API 키가 잘못되었거나 만료되었습니다.",
                        f"Cloud Run 환경 변수 {env_hint} 값을 확인하세요. (원문: {short})", 502)

    # --- 권한 부족 ---
    if "403" in low or "permission" in low or "forbidden" in low:
        return RagError(f"{service}_PERMISSION_DENIED",
                        f"{service} 요청이 권한 부족으로 거부되었습니다.",
                        f"API 키의 사용 권한/프로젝트 설정을 확인하세요. (원문: {short})", 502)

    # --- 사용량 한도 / 요청 과다 ---
    if any(s in low for s in ["429", "rate limit", "quota", "resource_exhausted",
                               "insufficient_quota", "too many requests"]):
        return RagError(f"{service}_RATE_LIMITED",
                        f"{service} 사용량 한도를 초과했거나 요청이 너무 많습니다.",
                        "잠시 후 다시 시도하거나 결제/할당량(Quota) 설정을 확인하세요.", 429)

    # --- 모델 이름 오류 ---
    if ("404" in low or "not found" in low) and "model" in low:
        model = LLM_MODEL if service == "GEMINI" else EMBEDDING_MODEL
        return RagError(f"{service}_MODEL_NOT_FOUND",
                        f"{service} 모델 '{model}'을(를) 찾을 수 없습니다.",
                        "모델 이름이 정확한지, 해당 API 키로 사용 가능한 모델인지 확인하세요. "
                        "(환경 변수 LLM_MODEL / EMBEDDING_MODEL 로 변경 가능)", 502)

    # --- Supabase: 함수/테이블/차원 문제 ---
    if service == "SUPABASE":
        if SUPABASE_QUERY_NAME in low or "pgrst202" in low or "could not find the function" in low:
            return RagError("SUPABASE_FUNCTION_MISSING",
                            f"Supabase 검색 함수 '{SUPABASE_QUERY_NAME}'이(가) 없거나 인자 형식이 다릅니다.",
                            "Supabase SQL Editor 에서 schema.sql 을 실행했는지 확인하세요.", 500)
        if "42p01" in low or "pgrst205" in low or ("relation" in low and "does not exist" in low) \
                or "could not find the table" in low:
            return RagError("SUPABASE_TABLE_MISSING",
                            f"Supabase 테이블 '{SUPABASE_TABLE}'이(가) 존재하지 않습니다.",
                            "Supabase SQL Editor 에서 schema.sql 을 실행해 테이블을 생성하세요.", 500)
        if "dimension" in low:
            return RagError("SUPABASE_DIMENSION_MISMATCH",
                            "임베딩 벡터 차원과 Supabase 테이블의 vector 차원이 일치하지 않습니다.",
                            f"EMBEDDING_MODEL({EMBEDDING_MODEL})의 차원과 schema.sql 의 vector(1536)을 맞추세요.", 500)

    # --- Gemini: 안전 필터 차단 ---
    if service == "GEMINI" and ("safety" in low or "blocked" in low):
        return RagError("GEMINI_CONTENT_BLOCKED",
                        "Gemini 안전 필터에 의해 응답 생성이 차단되었습니다.",
                        "질문 표현을 바꿔 다시 시도하세요.", 422)

    # --- 시간 초과 / 네트워크 ---
    if "timeout" in low or "timed out" in low or "deadline" in low:
        return RagError(f"{service}_TIMEOUT",
                        f"{service} 응답 시간이 초과되었습니다.",
                        "잠시 후 다시 시도하세요. 문서가 크다면 청크 크기를 조정해 보세요.", 504)
    if any(s in low for s in ["connection", "connect", "name or service not known",
                               "getaddrinfo", "ssl", "network"]):
        return RagError(f"{service}_CONNECTION_FAILED",
                        f"{service} 서버에 연결할 수 없습니다.",
                        f"URL/네트워크 설정을 확인하세요({env_hint}). (원문: {short})", 502)

    # --- 그 외 ---
    return RagError(f"{service}_ERROR",
                    f"{service} 처리 중 예상치 못한 오류가 발생했습니다: {type(exc).__name__}",
                    f"원문 오류: {short}", 502)


# =====================================================================
# 2. 외부 클라이언트 (지연 초기화 + 싱글톤)
#    - 서버 기동 시 환경 변수가 없어도 /health 는 응답할 수 있도록 지연 생성
# =====================================================================
_client_lock = threading.RLock()
_supabase: Optional[Client] = None
_embeddings: Optional[OpenAIEmbeddings] = None
_vector_store: Optional[SupabaseVectorStore] = None


def get_supabase() -> Client:
    """Supabase 클라이언트 (service_role 키 사용 → RLS 우회, 서버에서만 사용)."""
    global _supabase
    require_env(["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"])
    with _client_lock:
        if _supabase is None:
            try:
                _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            except Exception as e:
                raise RagError("SUPABASE_CLIENT_INIT_FAILED",
                               "Supabase 클라이언트 생성에 실패했습니다.",
                               f"SUPABASE_URL 형식(https://xxxx.supabase.co)과 키를 확인하세요. (원문: {e})",
                               500)
        return _supabase


def get_embeddings() -> OpenAIEmbeddings:
    """OpenAI 임베딩 모델 (text-embedding-3-small, 1536차원)."""
    global _embeddings
    require_env(["OPENAI_API_KEY"])
    with _client_lock:
        if _embeddings is None:
            _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        return _embeddings


def get_vector_store() -> SupabaseVectorStore:
    """Supabase 벡터 스토어 (검색 시 match_documents RPC 함수 사용)."""
    global _vector_store
    with _client_lock:
        if _vector_store is None:
            _vector_store = SupabaseVectorStore(
                client=get_supabase(),
                embedding=get_embeddings(),
                table_name=SUPABASE_TABLE,
                query_name=SUPABASE_QUERY_NAME,
            )
        return _vector_store


def get_retriever(k: int, source_filter: Optional[str]):
    """
    Supabase 리트리버 생성.
      k             : 가져올 청크 수
      source_filter : 특정 파일명으로 검색 범위를 제한 (metadata @> {"source": ...})
    """
    search_kwargs: Dict[str, Any] = {"k": k}
    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}
    return get_vector_store().as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def build_llm(temperature: float) -> ChatGoogleGenerativeAI:
    """Gemini 채팅 모델 객체 생성 (API 키는 환경 변수에서만 읽음)."""
    require_env(["GOOGLE_API_KEY"])
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        max_retries=2,
    )


# =====================================================================
# 3. 실시간 스트리밍 핸들러 (StreamHandler)
# =====================================================================
class StreamHandler(BaseCallbackHandler):
    """
    LLM 이 토큰을 하나 생성할 때마다 호출되는 LangChain 콜백 핸들러.

    on_token 콜백을 주입받아 토큰을 원하는 곳으로 즉시 전달한다.
      - /query/start  : 작업(job) 버퍼에 누적 → 클라이언트가 /query/poll 로 가져감
      - /query/stream : 큐(Queue)에 넣음 → NDJSON 으로 즉시 전송
    """

    def __init__(self, on_token: Callable[[str], None]):
        self.on_token = on_token
        self.token_count = 0     # 실제로 스트리밍된 토큰 수 (폴백 판단용)
        self.started_at = 0.0

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self.started_at = time.time()

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.started_at = time.time()

    def on_llm_new_token(self, token: Any, **kwargs) -> None:
        # 일부 모델은 토큰을 리스트/객체 형태로 줄 수 있으므로 문자열만 처리
        if not isinstance(token, str) or token == "":
            return
        self.token_count += 1
        self.on_token(token)

    def on_llm_end(self, response, **kwargs) -> None:
        elapsed = time.time() - self.started_at if self.started_at else 0
        logger.info("LLM 스트리밍 종료: 토큰 %d개, %.2f초", self.token_count, elapsed)

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        logger.error("LLM 스트리밍 중 오류: %s", error)


# =====================================================================
# 4. 프롬프트 / 체인 정의
# =====================================================================

# --- 4-1. 질의 확장(Query Augmentation) 시스템 메시지 ---
#   역할: 대화 맥락을 보고 사용자의 의도를 파악해 "검색하기 좋은 한 문장 질문"으로 변환
QUERY_AUGMENTATION_SYSTEM = (
    "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. "
    "대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라.\n"
    "- 질문에 답하지 말고, 변환된 질문 한 문장만 출력하라.\n"
    "- 대화 기록이 없거나 이미 명료한 질문이면 의미를 바꾸지 말고 다듬기만 하라."
)

# --- 4-2. 질의 확장 프롬프트 템플릿 ---
#   MessagesPlaceholder("chat_history") : 과거 대화(사용자 질문/AI 답변 목록)가 통째로 들어가는 동적 빈칸
QUERY_AUGMENTATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QUERY_AUGMENTATION_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# --- 4-3. 답변 생성 시스템 메시지 ---
ANSWER_SYSTEM = (
    "너는 문서에 기반해 답변하는 도시 정책 전문가야.\n\n"
    "[답변 규칙]\n"
    "1. 반드시 아래 <context> 안의 문서 내용만 근거로 답변한다. 추측하거나 외부 지식을 섞지 않는다.\n"
    "2. 근거로 사용한 문장 끝에 [1], [2] 처럼 <context> 의 번호로 출처를 표기한다.\n"
    "3. 문서에서 답을 찾을 수 없으면 '제공된 문서에서 해당 내용을 찾을 수 없습니다.'라고 답한다.\n"
    "4. 한국어로, 핵심부터 간결하고 구조적으로 설명한다.\n\n"
    "<context>\n{context}\n</context>"
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "원래 질문: {question}\n명료화된 질문: {augmented_query}"),
])


def build_query_augmentation_chain():
    """
    질의 확장 체인: 프롬프트 템플릿 | LLM | StrOutputParser
    (StrOutputParser 가 AIMessage 를 깔끔한 일반 문자열로 변환)
    """
    return QUERY_AUGMENTATION_PROMPT | build_llm(temperature=0.0) | StrOutputParser()


def build_answer_chain():
    """답변 생성 체인: 프롬프트 | LLM(스트리밍) | StrOutputParser"""
    return ANSWER_PROMPT | build_llm(temperature=0.3) | StrOutputParser()


def to_lc_messages(history: List["ChatTurn"]) -> List[BaseMessage]:
    """클라이언트가 보낸 대화 기록을 LangChain 메시지 객체(HumanMessage/AIMessage)로 변환."""
    messages: List[BaseMessage] = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        elif turn.role == "assistant":
            messages.append(AIMessage(content=turn.content))
    return messages


def format_context(docs: List[Document]) -> str:
    """검색된 청크를 번호가 붙은 컨텍스트 문자열로 변환 (LLM 이 [n] 으로 출처를 인용할 수 있게)."""
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "알 수 없음")
        page = d.metadata.get("page", "?")
        parts.append(f"[{i}] (출처: {src}, {page}페이지)\n{d.page_content}")
    return "\n\n".join(parts)


# =====================================================================
# 5. PDF 처리
# =====================================================================
def decode_pdf_base64(file_base64: str) -> bytes:
    """
    base64 문자열을 PDF 바이트로 변환하고 형식/크기를 검증한다.
    'data:application/pdf;base64,...' 형태의 Data URL 도 허용.
    """
    data = file_base64.split(",", 1)[1] if file_base64.startswith("data:") else file_base64
    try:
        pdf_bytes = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as e:
        raise RagError("INVALID_BASE64",
                       "전송된 파일 데이터(base64)를 해석할 수 없습니다.",
                       f"파일을 다시 선택해 업로드하세요. (원문: {e})", 400)

    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_MB:
        raise RagError("PDF_TOO_LARGE",
                       f"PDF 크기({size_mb:.1f}MB)가 허용 한도({MAX_PDF_MB}MB)를 초과했습니다.",
                       "파일을 분할하거나 압축한 뒤 다시 시도하세요.", 413)
    if not pdf_bytes.startswith(b"%PDF"):
        raise RagError("NOT_A_PDF",
                       "업로드된 파일이 PDF 형식이 아닙니다. (파일 시그니처 %PDF 없음)",
                       "확장자만 .pdf 로 바꾼 파일이 아닌지 확인하세요.", 415)
    return pdf_bytes


def extract_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """PDF 를 페이지별 텍스트 리스트로 변환: [{page: 1, text: "..."}, ...]"""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except PdfReadError as e:
        raise RagError("PDF_CORRUPTED",
                       "PDF 파일 구조가 손상되어 읽을 수 없습니다.",
                       f"다른 뷰어에서 '다른 이름으로 저장' 후 다시 시도하세요. (원문: {e})", 422)
    except Exception as e:
        raise RagError("PDF_OPEN_FAILED", "PDF 파일을 여는 중 오류가 발생했습니다.",
                       f"원문: {e}", 422)

    # 암호화된 PDF: 빈 비밀번호로 열리는 경우(열람 제한 없음)만 허용
    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:
                raise RagError("PDF_ENCRYPTED",
                               "암호로 보호된 PDF 입니다.",
                               "암호를 해제한 PDF 로 다시 업로드하세요.", 422)
        except RagError:
            raise
        except Exception as e:
            raise RagError("PDF_ENCRYPTED",
                           "암호화된 PDF 를 해독할 수 없습니다.",
                           f"암호를 해제한 PDF 로 다시 업로드하세요. (원문: {e})", 422)

    if len(reader.pages) == 0:
        raise RagError("PDF_EMPTY", "PDF 에 페이지가 없습니다.", "올바른 파일인지 확인하세요.", 422)

    pages = []
    failed_pages = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:  # 특정 페이지만 깨진 경우 나머지는 계속 처리
            logger.warning("%d페이지 텍스트 추출 실패: %s", idx, e)
            failed_pages.append(idx)
            text = ""
        pages.append({"page": idx, "text": text.strip()})

    if not any(p["text"] for p in pages):
        raise RagError("PDF_NO_TEXT",
                       "PDF 에서 추출 가능한 텍스트가 없습니다. 스캔 이미지로 된 PDF 일 가능성이 높습니다.",
                       "OCR(문자 인식) 처리된 PDF 로 변환한 뒤 다시 업로드하세요.", 422)
    if failed_pages:
        logger.warning("텍스트 추출 실패 페이지: %s", failed_pages)
    return pages


# =====================================================================
# 6. 작업(Job) 저장소 - 백그라운드 작업의 진행 현황을 메모리에 보관
# =====================================================================
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _cleanup_jobs() -> None:
    """TTL 이 지난 작업 정보를 제거해 메모리 누수를 방지."""
    now = time.time()
    expired = [jid for jid, j in JOBS.items() if now - j["created_at"] > JOB_TTL_SECONDS]
    for jid in expired:
        JOBS.pop(jid, None)


def new_job(kind: str) -> str:
    """새 작업을 등록하고 job_id 를 반환한다. kind: 'ingest' | 'query'"""
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        _cleanup_jobs()
        JOBS[job_id] = {
            "kind": kind, "status": "queued", "stage": "queued", "message": "대기 중",
            "progress": 0, "logs": [], "result": None, "error": None,
            # query 전용 필드
            "augmented_query": None, "sources": None, "answer": "",
            "created_at": time.time(), "updated_at": time.time(),
        }
    return job_id


def update_job(job_id: str, log: Optional[str] = None, **fields) -> None:
    """작업 상태를 갱신하고, log 가 있으면 진행 로그에 시간과 함께 추가한다."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = time.time()
        if log:
            job["logs"].append({"t": time.strftime("%H:%M:%S"), "msg": log})


def get_job(job_id: str, kind: str) -> Dict[str, Any]:
    """작업을 조회한다. 없으면 만료/재시작 가능성을 안내하는 404 오류."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or job["kind"] != kind:
            raise RagError("JOB_NOT_FOUND",
                           f"작업(job_id={job_id})을 찾을 수 없습니다.",
                           "작업이 만료되었거나 서버 인스턴스가 재시작/변경되었습니다. "
                           "Cloud Run 을 --max-instances=1 로 배포했는지 확인하고 다시 시도하세요.", 404)
        return json.loads(json.dumps(job, ensure_ascii=False))  # 깊은 복사


# =====================================================================
# 7. 벡터 DB 생성 파이프라인 (백그라운드 실행)
# =====================================================================
def run_ingest(job_id: str, filename: str, pdf_bytes: bytes,
               chunk_size: int, chunk_overlap: int) -> None:
    """
    PDF → 텍스트 → 청크 → 임베딩 → Supabase 저장.
    단계마다 progress(0~100)/stage/message 를 갱신해 클라이언트가 현황을 볼 수 있게 한다.
    """
    def step(pct: int, stage: str, msg: str) -> None:
        update_job(job_id, progress=pct, stage=stage, message=msg, log=msg)

    try:
        update_job(job_id, status="running")

        # ① 텍스트 추출
        step(5, "extract", "PDF 텍스트를 추출하고 있습니다.")
        pages = extract_pages(pdf_bytes)
        total_chars = sum(len(p["text"]) for p in pages)
        step(15, "extract", f"텍스트 추출 완료: {len(pages)}페이지, {total_chars:,}자")

        # ② 청크 분할 (페이지 정보를 메타데이터로 유지 → 출처 표기에 사용)
        step(18, "split", f"청크로 분할하고 있습니다. (크기 {chunk_size}, 겹침 {chunk_overlap})")
        page_docs = [
            Document(page_content=p["text"], metadata={"source": filename, "page": p["page"]})
            for p in pages if p["text"]
        ]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "다. ", ". ", " ", ""],  # 한국어 문장 경계 우선
        )
        chunks = splitter.split_documents(page_docs)
        if not chunks:
            raise RagError("NO_CHUNKS", "분할된 청크가 없습니다.",
                           "PDF 에 유효한 텍스트가 있는지 확인하세요.", 422)
        for i, c in enumerate(chunks):
            c.metadata["chunk_index"] = i
            c.metadata["total_chunks"] = len(chunks)
        step(25, "split", f"청크 분할 완료: 총 {len(chunks)}개")

        # ③ 같은 파일명으로 저장된 기존 벡터 삭제 (중복 방지)
        step(27, "cleanup", "같은 파일의 기존 벡터를 정리하고 있습니다.")
        sb = get_supabase()
        embeddings = get_embeddings()
        try:
            res = sb.table(SUPABASE_TABLE).delete().eq("metadata->>source", filename).execute()
            deleted = len(res.data or [])
        except Exception as e:
            raise classify_external_error(e, "SUPABASE")
        step(30, "cleanup", f"기존 벡터 {deleted}개 삭제")

        # ④ 배치 단위로 임베딩 → 저장
        total = len(chunks)
        for start in range(0, total, EMBED_BATCH_SIZE):
            batch = chunks[start:start + EMBED_BATCH_SIZE]
            end = start + len(batch)

            # 4-1. OpenAI 임베딩
            try:
                vectors = embeddings.embed_documents([c.page_content for c in batch])
            except Exception as e:
                raise classify_external_error(e, "OPENAI")

            # 4-2. Supabase 저장 (id 는 DB 에서 gen_random_uuid() 로 자동 생성)
            rows = [
                {"content": c.page_content, "metadata": c.metadata, "embedding": v}
                for c, v in zip(batch, vectors)
            ]
            try:
                sb.table(SUPABASE_TABLE).insert(rows).execute()
            except Exception as e:
                raise classify_external_error(e, "SUPABASE")

            pct = 30 + int(65 * end / total)
            step(pct, "embed", f"임베딩 및 저장: {end}/{total} 청크")

        # ⑤ 완료
        result = {"filename": filename, "pages": len(pages), "chars": total_chars,
                  "chunks": total, "deleted": deleted, "embedding_model": EMBEDDING_MODEL}
        update_job(job_id, status="done", progress=100, stage="done",
                   message="벡터 DB 생성이 완료되었습니다.", result=result,
                   log=f"완료: {total}개 청크 저장")

    except RagError as e:
        logger.error("[ingest %s] %s - %s", job_id, e.code, e.message)
        update_job(job_id, status="error", stage="error", error=e.to_dict(),
                   message=e.message, log=f"오류[{e.code}] {e.message}")
    except Exception as e:
        logger.error("[ingest %s] 예기치 않은 오류\n%s", job_id, traceback.format_exc())
        err = RagError("INGEST_INTERNAL_ERROR",
                       f"벡터 DB 생성 중 예기치 않은 오류가 발생했습니다: {type(e).__name__}",
                       f"원문: {str(e)[:300]}", 500)
        update_job(job_id, status="error", stage="error", error=err.to_dict(),
                   message=err.message, log=f"오류[{err.code}] {err.message}")


# =====================================================================
# 8. 질의 응답 파이프라인 (질의 확장 → 검색 → 스트리밍 답변)
# =====================================================================
def run_query_pipeline(question: str, history: List["ChatTurn"], k: int,
                       source_filter: Optional[str],
                       emit: Callable[[str, Dict[str, Any]], None]) -> None:
    """
    emit(event, data) 로 진행 이벤트를 내보낸다.
      stage     : 단계 변경 {stage, message}
      augmented : 증강된 질의 {augmented_query}
      sources   : 검색된 청크 목록 {sources}
      token     : 답변 토큰 {text}
      done      : 완료
      error     : 오류 {code, message, hint}
    """
    try:
        # ---------- [1~4] 질의 확장 ----------
        emit("stage", {"stage": "augment", "message": "대화 맥락을 반영해 질문을 명료하게 바꾸고 있습니다."})
        chat_messages = to_lc_messages(history)
        query_augmentation_chain = build_query_augmentation_chain()
        try:
            # 현재 질문 + 이전 대화 기록을 딕셔너리로 묶어 주입
            augmented_query = query_augmentation_chain.invoke({
                "question": question,
                "chat_history": chat_messages,
            }).strip()
        except Exception as e:
            raise classify_external_error(e, "GEMINI")
        if not augmented_query:
            augmented_query = question  # 모델이 빈 응답을 주면 원래 질문으로 대체
        emit("augmented", {"augmented_query": augmented_query})

        # ---------- [5] 증강된 질의로 벡터 DB 검색 ----------
        scope = f"'{source_filter}' 문서에서" if source_filter else "전체 문서에서"
        emit("stage", {"stage": "retrieve", "message": f"{scope} 관련 내용을 검색하고 있습니다."})
        retriever = get_retriever(k, source_filter)
        try:
            docs: List[Document] = retriever.invoke(augmented_query)
        except Exception as e:
            raise classify_external_error(e, guess_service(e, "SUPABASE"))

        sources = [{
            "index": i,
            "source": d.metadata.get("source", "알 수 없음"),
            "page": d.metadata.get("page"),
            "chunk_index": d.metadata.get("chunk_index"),
            "content": d.page_content,
        } for i, d in enumerate(docs, start=1)]
        emit("sources", {"sources": sources})

        if not docs:
            emit("token", {"text": "검색된 문서가 없습니다. 먼저 벡터 DB 를 생성했는지, "
                                   "검색 범위(파일 필터)가 올바른지 확인하세요."})
            emit("done", {"warning": "NO_DOCUMENTS"})
            return

        # ---------- 답변 생성 (StreamHandler 로 실시간 토큰 전달) ----------
        emit("stage", {"stage": "generate", "message": f"{LLM_MODEL} 모델이 답변을 작성하고 있습니다."})
        handler = StreamHandler(on_token=lambda t: emit("token", {"text": t}))
        answer_chain = build_answer_chain()
        collected: List[str] = []
        try:
            for chunk in answer_chain.stream(
                {
                    "context": format_context(docs),
                    "chat_history": chat_messages,
                    "question": question,
                    "augmented_query": augmented_query,
                },
                config={"callbacks": [handler]},
            ):
                collected.append(chunk)
        except Exception as e:
            raise classify_external_error(e, "GEMINI")

        # 콜백으로 토큰이 전달되지 않은 경우(모델/버전 차이) 수집한 결과를 한 번에 전송
        if handler.token_count == 0 and collected:
            emit("token", {"text": "".join(collected)})
        if not "".join(collected).strip() and handler.token_count == 0:
            raise RagError("GEMINI_EMPTY_RESPONSE", "언어 모델이 빈 답변을 반환했습니다.",
                           "질문을 바꾸거나 잠시 후 다시 시도하세요.", 502)

        emit("done", {})

    except RagError as e:
        logger.error("[query] %s - %s", e.code, e.message)
        emit("error", e.to_dict())
    except Exception as e:
        logger.error("[query] 예기치 않은 오류\n%s", traceback.format_exc())
        emit("error", RagError("QUERY_INTERNAL_ERROR",
                               f"질의 처리 중 예기치 않은 오류가 발생했습니다: {type(e).__name__}",
                               f"원문: {str(e)[:300]}", 500).to_dict())


def job_emitter(job_id: str) -> Callable[[str, Dict[str, Any]], None]:
    """질의 이벤트를 작업(job) 상태에 반영하는 emit 함수를 만든다 (폴링 방식용)."""
    def emit(event: str, data: Dict[str, Any]) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job["updated_at"] = time.time()
            if event == "token":
                job["answer"] += data["text"]          # StreamHandler 토큰 누적
                return
            if event == "stage":
                job.update(status="running", stage=data["stage"], message=data["message"])
                job["logs"].append({"t": time.strftime("%H:%M:%S"), "msg": data["message"]})
            elif event == "augmented":
                job["augmented_query"] = data["augmented_query"]
            elif event == "sources":
                job["sources"] = data["sources"]
            elif event == "done":
                job.update(status="done", stage="done", message="답변 생성 완료",
                           result=data or None)
            elif event == "error":
                job.update(status="error", stage="error", error=data, message=data["message"])
    return emit


# =====================================================================
# 9. FastAPI 앱 / 요청 모델 / 공통 처리
# =====================================================================
app = FastAPI(title="City Policy RAG Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Apps Script 는 서버 간 호출이라 CORS 영향 없음. 필요 시 도메인 제한
    allow_methods=["*"],
    allow_headers=["*"],
)


class PdfPayload(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255, description="원본 파일명")
    file_base64: str = Field(..., min_length=10, description="PDF base64 문자열")


class IngestPayload(PdfPayload):
    chunk_size: int = Field(1000, ge=200, le=4000)
    chunk_overlap: int = Field(200, ge=0, le=1000)


class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class QueryPayload(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    chat_history: List[ChatTurn] = Field(default_factory=list)
    k: int = Field(4, ge=1, le=10)
    source_filter: Optional[str] = None


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """SERVER_API_KEY 가 설정된 경우에만 X-API-Key 헤더를 검증한다."""
    if not SERVER_API_KEY:
        return
    if not x_api_key:
        raise RagError("API_KEY_MISSING", "요청에 X-API-Key 헤더가 없습니다.",
                       "Apps Script 스크립트 속성 SERVER_API_KEY 를 설정하세요.", 401)
    if not hmac.compare_digest(x_api_key, SERVER_API_KEY):
        raise RagError("API_KEY_INVALID", "X-API-Key 값이 서버 설정과 일치하지 않습니다.",
                       "Apps Script 의 SERVER_API_KEY 와 Cloud Run 의 SERVER_API_KEY 를 같게 맞추세요.", 403)


@app.exception_handler(RagError)
async def rag_error_handler(_: Request, exc: RagError):
    return JSONResponse(status_code=exc.status, content={"error": exc.to_dict()})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    # 어떤 필드가 왜 잘못되었는지 구체적으로 안내
    details = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", []) if x != "body")
        details.append(f"{loc}: {err.get('msg')}")
    return JSONResponse(status_code=422, content={"error": {
        "code": "INVALID_REQUEST",
        "message": "요청 형식이 올바르지 않습니다: " + "; ".join(details),
        "hint": "필수 값 누락, 길이/범위 초과 여부를 확인하세요.",
    }})


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception):
    logger.error("처리되지 않은 오류\n%s", traceback.format_exc())
    return JSONResponse(status_code=500, content={"error": {
        "code": "INTERNAL_SERVER_ERROR",
        "message": f"서버 내부 오류가 발생했습니다: {type(exc).__name__}",
        "hint": "Cloud Run 로그(Logging)에서 상세 스택 트레이스를 확인하세요.",
    }})


# =====================================================================
# 10. 엔드포인트
# =====================================================================
@app.get("/")
def root():
    return {"service": "city-policy-rag", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health(deep: bool = False):
    """
    서버 상태 점검.
      deep=true 이면 Supabase 테이블 접근까지 실제로 확인한다.
    """
    missing = [k for k, v in REQUIRED_ENV.items() if not v]
    info: Dict[str, Any] = {
        "status": "ok" if not missing else "degraded",
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "missing_env": missing,
        "auth_required": bool(SERVER_API_KEY),
        "active_jobs": len(JOBS),
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if deep and not missing:
        try:
            get_supabase().table(SUPABASE_TABLE).select("id").limit(1).execute()
            info["supabase"] = "ok"
        except Exception as e:
            err = classify_external_error(e, "SUPABASE")
            info["status"] = "degraded"
            info["supabase"] = err.to_dict()
    return info


@app.post("/extract", dependencies=[Depends(verify_api_key)])
def extract(payload: PdfPayload):
    """PDF 텍스트 추출 (미리보기 표시용)."""
    pdf_bytes = decode_pdf_base64(payload.file_base64)
    pages = extract_pages(pdf_bytes)
    return {
        "filename": payload.filename,
        "num_pages": len(pages),
        "total_chars": sum(len(p["text"]) for p in pages),
        "empty_pages": [p["page"] for p in pages if not p["text"]],
        "pages": pages,
    }


@app.post("/ingest/start", dependencies=[Depends(verify_api_key)])
def ingest_start(payload: IngestPayload):
    """벡터 DB 생성 작업을 시작하고 job_id 를 즉시 반환한다."""
    if payload.chunk_overlap >= payload.chunk_size:
        raise RagError("INVALID_CHUNK_SETTING",
                       f"청크 겹침({payload.chunk_overlap})은 청크 크기({payload.chunk_size})보다 작아야 합니다.",
                       "겹침 값을 청크 크기의 10~20% 정도로 설정하세요.", 400)
    require_env(["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"])
    pdf_bytes = decode_pdf_base64(payload.file_base64)  # 형식 오류는 작업 시작 전에 바로 알림

    job_id = new_job("ingest")
    threading.Thread(
        target=run_ingest,
        args=(job_id, payload.filename, pdf_bytes, payload.chunk_size, payload.chunk_overlap),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/ingest/status/{job_id}", dependencies=[Depends(verify_api_key)])
def ingest_status(job_id: str):
    """벡터 DB 생성 진행 현황 조회."""
    job = get_job(job_id, "ingest")
    return {k: job[k] for k in ("status", "stage", "message", "progress", "logs", "result", "error")}


@app.post("/query/start", dependencies=[Depends(verify_api_key)])
def query_start(payload: QueryPayload):
    """질의 작업을 시작하고 job_id 를 즉시 반환한다 (답변은 /query/poll 로 실시간 수신)."""
    require_env(list(REQUIRED_ENV.keys()))
    job_id = new_job("query")
    threading.Thread(
        target=run_query_pipeline,
        args=(payload.question, payload.chat_history, payload.k,
              payload.source_filter, job_emitter(job_id)),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/query/poll/{job_id}", dependencies=[Depends(verify_api_key)])
def query_poll(job_id: str, offset: int = 0):
    """
    StreamHandler 가 누적한 답변 중 offset 이후의 새 토큰(delta)만 반환.
    클라이언트는 응답의 offset 을 다음 요청에 그대로 넘기면 된다.
    """
    job = get_job(job_id, "query")
    answer = job["answer"]
    offset = max(0, min(offset, len(answer)))
    return {
        "status": job["status"], "stage": job["stage"], "message": job["message"],
        "augmented_query": job["augmented_query"], "sources": job["sources"],
        "delta": answer[offset:], "offset": len(answer),
        "result": job["result"], "error": job["error"],
    }


@app.post("/query/stream", dependencies=[Depends(verify_api_key)])
def query_stream(payload: QueryPayload):
    """
    NDJSON 실시간 스트리밍. 한 줄에 하나의 이벤트 JSON:
      {"event":"token","text":"..."} 등
    """
    require_env(list(REQUIRED_ENV.keys()))
    q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()

    def emit(event: str, data: Dict[str, Any]) -> None:
        q.put({"event": event, **data})

    def worker() -> None:
        try:
            run_query_pipeline(payload.question, payload.chat_history, payload.k,
                               payload.source_filter, emit)
        finally:
            q.put(None)  # 스트림 종료 신호

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while True:
            item = q.get()
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# 로컬 실행: python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
