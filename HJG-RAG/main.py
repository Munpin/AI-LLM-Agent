"""
================================================================================
 현진건 RAG 서버 (Google Cloud Run + FastAPI + LangChain + Chroma + Google AI)
================================================================================

[전체 흐름]
  1) /ingest        : PDF(base64) 수신 → 텍스트 변환 → 청크 분할 → 임베딩 → Chroma 저장
                      (백그라운드 작업으로 실행하고, 진행 현황은 job 으로 조회)
  2) /query         : 질문 수신 → MultiQueryRetriever 로 질의 확장/검색
                      → 프롬프트 허브(rlm/rag-prompt) + 시스템 역할 → Gemini 답변 생성
                      → StreamHandler 가 토큰을 job 에 실시간 누적
  3) /jobs/{job_id} : 클라이언트(Apps Script)가 offset 기반으로 증분(로그/토큰)을 폴링
  4) /query/stream  : (선택) SSE 로 진짜 실시간 스트리밍을 받는 일반 클라이언트용

[왜 job + 폴링 구조인가?]
  Google Apps Script 의 UrlFetchApp 은 응답이 "끝까지" 도착해야 반환되므로
  SSE/청크 스트리밍을 직접 받을 수 없습니다. 그래서 서버는 StreamHandler 로 받은
  토큰을 메모리의 job 에 계속 쌓고, Apps Script 는 짧은 주기로 새로 쌓인 부분만
  가져가는 방식으로 실시간 스트리밍을 구현합니다.
  ※ job 상태가 메모리에 있으므로 Cloud Run 은 max-instances=1, CPU 항상 할당
    (--no-cpu-throttling) 으로 배포해야 합니다. (README/배포 명령 참고)

[Chroma 사용법 요약]
  - 설치 : pip install chromadb langchain-chroma
  - 클라이언트 : chromadb.PersistentClient(path=...)  → 디스크에 저장되는 로컬 벡터 DB
  - LangChain 래퍼 : Chroma(client=..., collection_name=..., embedding_function=...)
      * add_documents(docs)        : 문서 임베딩 후 저장
      * similarity_search(q, k)    : 유사도 검색
      * as_retriever(search_kwargs): LangChain 검색기(Retriever)로 변환
  - Cloud Run 의 /tmp 는 인스턴스가 내려가면 사라집니다. 영구 보존이 필요하면
    Cloud Storage 볼륨 마운트 경로를 CHROMA_DIR 로 지정하세요.

[환경 변수]
  GOOGLE_API_KEY       (필수) Google AI Studio API 키
  EMBEDDING_MODEL      (선택) 기본값 models/gemini-embedding-001
  CHAT_MODEL           (선택) 기본값 gemini-2.5-flash
  CHROMA_DIR           (선택) 기본값 /tmp/chroma_db
  SERVER_ACCESS_TOKEN  (선택) 설정 시 요청 헤더 X-Access-Token 이 일치해야만 허용
  PROMPT_HUB_ID        (선택) 기본값 rlm/rag-prompt
================================================================================
"""

import base64
import json
import logging
import os
import queue
import tempfile
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# --- LangChain 구성 요소 ---------------------------------------------------------
from langchain.retrievers.multi_query import MultiQueryRetriever  # 다중 질문 검색기
from langchain_chroma import Chroma  # Chroma 벡터 DB 래퍼
from langchain_community.document_loaders import PyPDFLoader  # PDF → Document
from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================================================================
# 1. 설정값 (모두 환경 변수에서 읽음 → 코드에 키를 하드코딩하지 않음)
# ==============================================================================
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
# text-embedding-004 는 지원 종료되어 404 가 나므로 gemini-embedding-001 을 기본값으로 사용
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "models/gemini-embedding-001")
CHAT_MODEL: str = os.environ.get("CHAT_MODEL", "gemini-2.5-flash")
CHROMA_DIR: str = os.environ.get("CHROMA_DIR", "/tmp/chroma_db")
SERVER_ACCESS_TOKEN: str = os.environ.get("SERVER_ACCESS_TOKEN", "")
PROMPT_HUB_ID: str = os.environ.get("PROMPT_HUB_ID", "rlm/rag-prompt")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))        # 청크 하나의 최대 글자 수
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))   # 청크 간 겹치는 글자 수(문맥 유지)
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))  # 임베딩 배치 크기(진행률 단위)
TOP_K = int(os.environ.get("TOP_K", "4"))                     # 질의 1개당 검색할 청크 수
MAX_PDF_MB = int(os.environ.get("MAX_PDF_MB", "30"))          # 업로드 허용 최대 크기

# 요구사항에 명시된 시스템 역할 (문구 그대로 사용)
SYSTEM_ROLE = (
    "당신은 소설 운수 좋은날을 집필한 현진건 작가님입니다. "
    "근대 한국 문학의 현실주의 문제로 사용자의 질문에 답변하세요"
)
# 출처 표기를 위한 보조 지시 (답변이 검색 청크에 근거하도록 강제)
CITATION_RULE = (
    "반드시 한국어로 답하고, 아래 제공된 문맥(context)에 근거해서만 답변하세요. "
    "근거로 사용한 문맥은 문장 끝에 [출처 1], [출처 2]처럼 번호로 표기하세요. "
    "문맥에서 찾을 수 없는 내용은 모른다고 솔직히 말하세요."
)

# Chroma 익명 사용 통계 전송 끄기
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hyunjingun-rag")

# ==============================================================================
# 2. Chroma 클라이언트 (디스크 영구 저장형)
#    - 하나의 PDF = 하나의 컬렉션(collection) 으로 관리합니다.
# ==============================================================================
os.makedirs(CHROMA_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)


def list_collection_names() -> List[str]:
    """chromadb 버전에 따라 list_collections() 가 이름(str) 또는 Collection 객체를
    반환하므로 두 경우 모두 이름 리스트로 통일합니다."""
    return [c if isinstance(c, str) else c.name for c in chroma_client.list_collections()]


# ==============================================================================
# 3. 모델 객체 (지연 생성 + 캐시)
#    - API 키가 없어도 서버 자체는 기동되어 /health 로 상태를 알려줄 수 있게 합니다.
# ==============================================================================
_embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
_model_lock = threading.Lock()


def require_api_key() -> None:
    """API 키가 설정되지 않았으면 명확한 오류를 발생시킵니다."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("환경 변수 GOOGLE_API_KEY 가 설정되지 않았습니다.")


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """구글 임베딩 모델(EMBEDDING_MODEL)을 한 번만 생성해 재사용합니다."""
    global _embeddings
    require_api_key()
    with _model_lock:
        if _embeddings is None:
            _embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=GOOGLE_API_KEY,
            )
    return _embeddings


def make_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Gemini 채팅 모델을 생성합니다.
    - 답변용(temperature 0.3)과 질의 확장용(temperature 0.0)을 따로 만들어 사용합니다."""
    require_api_key()
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )


# ==============================================================================
# 4. 실시간 스트리밍 핸들러 (StreamHandler)
#    - LangChain 콜백 시스템에 연결되어, LLM 이 토큰을 하나 생성할 때마다
#      on_llm_new_token 이 호출됩니다.
#    - 받은 토큰을 "어디로 보낼지"는 생성자에 넘긴 함수(on_token)가 결정합니다.
#        * job 저장소에 누적  → Apps Script 폴링용
#        * Queue 에 넣기      → SSE 스트리밍용
# ==============================================================================
class StreamHandler(BaseCallbackHandler):
    def __init__(self, on_token: Callable[[str], None]):
        self.on_token = on_token  # 토큰 수신 시 실행할 함수
        self.text = ""            # 지금까지 받은 전체 답변(최종 확인용)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """LLM 이 새 토큰을 생성할 때마다 LangChain 이 자동으로 호출합니다."""
        if not token:
            return
        self.text += token
        self.on_token(token)


# ==============================================================================
# 5. Job 저장소 (진행 현황 + 스트리밍 토큰 버퍼)
#    - 스레드 안전성을 위해 Lock 사용
#    - 클라이언트는 log_offset / token_offset 을 넘겨 "새로 생긴 부분만" 받아갑니다.
# ==============================================================================
class JobStore:
    def __init__(self, ttl_seconds: int = 3600):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.ttl = ttl_seconds  # 오래된 job 자동 정리 기준(초)

    def create(self, job_type: str, **extra: Any) -> str:
        """새 job 을 만들고 ID 를 반환합니다."""
        self._cleanup()
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "type": job_type,          # "ingest" 또는 "query"
                "status": "pending",       # pending → running → done / error
                "progress": 0,             # 0~100
                "logs": [],                # 진행 메시지 목록
                "tokens": [],              # 스트리밍된 답변 토큰 목록
                "queries": [],             # MultiQueryRetriever 가 생성한 확장 질의
                "sources": [],             # 검색된 청크(출처) 목록
                "result": None,            # 작업 결과(컬렉션 정보 등)
                "error": None,
                "created_at": time.time(),
                **extra,
            }
        return job_id

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def log(self, job_id: str, message: str, progress: Optional[int] = None) -> None:
        """진행 메시지를 추가하고 필요하면 진행률도 갱신합니다."""
        logger.info("[job %s] %s", job_id[:8], message)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["logs"].append({"t": time.time(), "msg": message})
            if progress is not None:
                job["progress"] = max(0, min(100, int(progress)))

    def append_token(self, job_id: str, token: str) -> None:
        """StreamHandler 가 받은 토큰을 job 에 누적합니다."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job["tokens"].append(token)

    def snapshot(self, job_id: str, log_offset: int = 0, token_offset: int = 0) -> Dict[str, Any]:
        """offset 이후의 증분만 잘라서 반환합니다(네트워크 전송량 최소화)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            new_logs = job["logs"][log_offset:]
            new_tokens = job["tokens"][token_offset:]
            return {
                "id": job["id"],
                "type": job["type"],
                "status": job["status"],
                "progress": job["progress"],
                "logs": [l["msg"] for l in new_logs],
                "log_offset": log_offset + len(new_logs),
                "text": "".join(new_tokens),
                "token_offset": token_offset + len(new_tokens),
                "queries": job["queries"],
                "sources": job["sources"],
                "result": job["result"],
                "error": job["error"],
            }

    def _cleanup(self) -> None:
        """TTL 이 지난 job 을 메모리에서 제거합니다."""
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._jobs.items() if now - v["created_at"] > self.ttl]
            for k in expired:
                del self._jobs[k]


jobs = JobStore()

# ==============================================================================
# 6. 프롬프트 허브 (rlm/rag-prompt) + 시스템 역할 결합
#    - 허브 프롬프트는 {context}, {question} 변수를 가진 human 메시지입니다.
#    - 그 앞에 system 메시지(현진건 작가 역할)를 붙여 최종 프롬프트를 만듭니다.
#    - 네트워크 문제로 허브 접근이 실패하면 동일한 구조의 로컬 프롬프트로 대체합니다.
# ==============================================================================
_rag_prompt: Optional[ChatPromptTemplate] = None
_prompt_source = "unloaded"

FALLBACK_HUMAN_TEMPLATE = (
    "You are an assistant for question-answering tasks. Use the following pieces of "
    "retrieved context to answer the question. If you don't know the answer, just say "
    "that you don't know.\nQuestion: {question}\nContext: {context}\nAnswer:"
)


def get_rag_prompt() -> ChatPromptTemplate:
    global _rag_prompt, _prompt_source
    if _rag_prompt is not None:
        return _rag_prompt
    try:
        from langchain import hub  # langchain 0.3.x 의 프롬프트 허브 클라이언트

        hub_prompt = hub.pull(PROMPT_HUB_ID)
        human_messages = list(hub_prompt.messages)
        _prompt_source = f"hub:{PROMPT_HUB_ID}"
    except Exception as e:  # 허브 접속 실패 시 대체
        logger.warning("프롬프트 허브 로드 실패(%s) → 로컬 프롬프트 사용", e)
        human_messages = [("human", FALLBACK_HUMAN_TEMPLATE)]
        _prompt_source = "local-fallback"

    _rag_prompt = ChatPromptTemplate.from_messages(
        [("system", f"{SYSTEM_ROLE}\n\n{CITATION_RULE}"), *human_messages]
    )
    return _rag_prompt


# ==============================================================================
# 7. 다중 질문 검색기 (MultiQueryRetriever)
#    - 사용자의 질문 1개를 LLM 으로 서로 다른 관점의 질문 여러 개로 바꾼 뒤
#      각각 벡터 검색을 수행하고, 결과를 중복 제거하여 합칩니다.
#    - 생성된 확장 질의를 화면에 보여주기 위해 generate_queries 를 오버라이드합니다.
# ==============================================================================
MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "당신은 벡터 데이터베이스 검색을 돕는 AI 조수입니다.\n"
        "사용자의 질문을 관련 문서를 더 잘 찾을 수 있도록, 서로 다른 관점의 질문 3개로 "
        "다시 작성하세요. 각 질문은 줄바꿈으로만 구분하고 번호나 설명은 붙이지 마세요.\n"
        "원래 질문: {question}"
    ),
)


class TrackingMultiQueryRetriever(MultiQueryRetriever):
    """생성된 확장 질의를 generated_queries 에 기록하는 MultiQueryRetriever."""

    generated_queries: List[str] = Field(default_factory=list)

    def generate_queries(
        self, question: str, run_manager: CallbackManagerForRetrieverRun
    ) -> List[str]:
        queries = super().generate_queries(question, run_manager)
        self.generated_queries = [q.strip() for q in queries if q and q.strip()]
        return queries


def get_vectorstore(collection_id: str) -> Chroma:
    """컬렉션 ID 로 기존 Chroma 벡터 DB 를 엽니다."""
    if collection_id not in list_collection_names():
        raise HTTPException(status_code=404, detail="벡터 DB(컬렉션)를 찾을 수 없습니다. 먼저 생성하세요.")
    return Chroma(
        client=chroma_client,
        collection_name=collection_id,
        embedding_function=get_embeddings(),
    )


def format_docs(docs: List[Document]) -> str:
    """검색된 청크를 LLM 문맥(context) 문자열로 만듭니다.
    번호를 붙여 두어야 LLM 이 [출처 n] 형태로 인용할 수 있습니다."""
    blocks = []
    for i, d in enumerate(docs, start=1):
        name = d.metadata.get("source_name", "문서")
        page = int(d.metadata.get("page", 0)) + 1  # PyPDFLoader 의 page 는 0부터 시작
        blocks.append(f"[출처 {i}] ({name}, {page}쪽)\n{d.page_content}")
    return "\n\n".join(blocks)


def docs_to_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """클라이언트 표시용 출처 정보(번호/파일명/쪽/본문 일부)로 변환합니다."""
    return [
        {
            "index": i,
            "source": d.metadata.get("source_name", "문서"),
            "page": int(d.metadata.get("page", 0)) + 1,
            "chunk": d.metadata.get("chunk_index"),
            "content": d.page_content[:500],
        }
        for i, d in enumerate(docs, start=1)
    ]


def build_chains(collection_id: str):
    """검색 체인과 생성 체인을 만듭니다.

    [질의 확장 - RunnablePassthrough]
      retrieval_chain  : 입력 질문을 RunnablePassthrough 로 그대로 통과시키면서(question),
                         동시에 MultiQueryRetriever 로 확장 검색(docs)을 수행 → dict 로 병합
      generation_chain : RunnablePassthrough.assign 으로 기존 dict 를 유지한 채
                         context(포맷된 청크)를 추가 → 프롬프트 → LLM → 문자열
      rag_chain        : 위 둘을 이어 붙인 전체 파이프라인(참고용, SSE 외 일괄 호출 시 사용)
    """
    vectorstore = get_vectorstore(collection_id)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    retriever = TrackingMultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=make_llm(temperature=0.0),  # 질의 확장은 일관성 있게 temperature 0
        prompt=MULTI_QUERY_PROMPT,
        include_original=True,          # 원래 질문으로도 검색
    )

    retrieval_chain = RunnableParallel(
        question=RunnablePassthrough(),  # 질문 원문 그대로 전달
        docs=retriever,                  # 다중 질의 검색 결과
    )

    generation_chain = (
        RunnablePassthrough.assign(context=RunnableLambda(lambda x: format_docs(x["docs"])))
        | get_rag_prompt()
        | make_llm(temperature=0.3)
        | StrOutputParser()
    )

    rag_chain = retrieval_chain | RunnablePassthrough.assign(answer=generation_chain)
    return retriever, retrieval_chain, generation_chain, rag_chain


# ==============================================================================
# 8. 백그라운드 작업: PDF → 텍스트 → 청크 → 임베딩 → Chroma
# ==============================================================================
def run_ingest_job(job_id: str, filename: str, pdf_bytes: bytes) -> None:
    tmp_path = None
    try:
        jobs.update(job_id, status="running")
        jobs.log(job_id, f"'{filename}' 파일을 받았습니다.", 5)

        # (1) PyPDFLoader 는 파일 경로가 필요하므로 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_path = f.name

        # (2) PDF → 텍스트 (페이지 단위 Document 리스트)
        jobs.log(job_id, "PDF를 텍스트로 변환하는 중…", 10)
        pages = PyPDFLoader(tmp_path).load()
        pages = [p for p in pages if p.page_content.strip()]
        if not pages:
            raise ValueError("텍스트를 추출할 수 없습니다. 스캔 이미지 PDF라면 OCR 처리가 필요합니다.")
        jobs.log(job_id, f"{len(pages)}쪽 분량의 텍스트를 추출했습니다.", 25)

        # (3) 청크 분할: 문단 → 문장 → 단어 순으로 자연스러운 경계를 찾아 자름
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "。", " ", ""],
        )
        chunks = splitter.split_documents(pages)
        for i, c in enumerate(chunks):
            c.metadata["source_name"] = filename  # 출처 표기용 파일명
            c.metadata["chunk_index"] = i
        jobs.log(job_id, f"{len(chunks)}개의 청크로 나눴습니다. (크기 {CHUNK_SIZE}, 겹침 {CHUNK_OVERLAP})", 35)

        # (4) 컬렉션 생성 (이름 규칙: 3~63자, 영문/숫자/_/-)
        collection_id = f"pdf_{uuid.uuid4().hex[:16]}"
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_id,
            embedding_function=get_embeddings(),
            collection_metadata={"hnsw:space": "cosine", "filename": filename},
        )

        # (5) 배치 단위로 임베딩 + 저장 → 배치마다 진행률 갱신
        total = len(chunks)
        jobs.log(job_id, f"임베딩 모델({EMBEDDING_MODEL})로 벡터를 만드는 중…", 36)
        for start in range(0, total, EMBED_BATCH_SIZE):
            batch = chunks[start:start + EMBED_BATCH_SIZE]
            ids = [f"{collection_id}-{start + j}" for j in range(len(batch))]
            vectorstore.add_documents(batch, ids=ids)
            done = start + len(batch)
            jobs.log(job_id, f"벡터 DB 저장 {done}/{total}", 36 + int(60 * done / total))

        result = {
            "collection_id": collection_id,
            "filename": filename,
            "pages": len(pages),
            "chunks": total,
        }
        jobs.update(job_id, status="done", progress=100, result=result)
        jobs.log(job_id, "벡터 DB 생성이 끝났습니다. 이제 질문할 수 있습니다.", 100)
    except Exception as e:
        logger.exception("ingest 실패")
        jobs.update(job_id, status="error", error=str(e))
        jobs.log(job_id, f"오류: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==============================================================================
# 9. 백그라운드 작업: 다중 질의 검색 → 답변 스트리밍 생성
# ==============================================================================
def run_query_job(job_id: str, question: str, collection_id: str) -> None:
    try:
        jobs.update(job_id, status="running")
        jobs.log(job_id, "질문을 여러 관점으로 확장해 검색하는 중…", 10)
        retriever, retrieval_chain, generation_chain, _ = build_chains(collection_id)

        # (1) 검색 단계: {"question": ..., "docs": [...]}
        retrieved = retrieval_chain.invoke(question)
        jobs.update(
            job_id,
            queries=retriever.generated_queries,
            sources=docs_to_sources(retrieved["docs"]),
        )
        jobs.log(job_id, f"관련 청크 {len(retrieved['docs'])}개를 찾았습니다.", 45)

        # (2) 생성 단계: StreamHandler 가 토큰을 job 에 실시간 누적
        handler = StreamHandler(on_token=lambda t: jobs.append_token(job_id, t))
        jobs.log(job_id, "답변을 작성하는 중…", 55)
        collected = []
        for chunk in generation_chain.stream(retrieved, config={"callbacks": [handler]}):
            collected.append(chunk)  # 콜백이 동작하지 않는 환경을 위한 백업

        # 콜백으로 토큰이 하나도 안 들어왔다면 결과 전체를 한 번에 넣어줌
        if not handler.text and collected:
            jobs.append_token(job_id, "".join(collected))

        jobs.update(job_id, status="done", progress=100)
        jobs.log(job_id, "답변을 완료했습니다.", 100)
    except HTTPException as e:
        jobs.update(job_id, status="error", error=e.detail)
        jobs.log(job_id, f"오류: {e.detail}")
    except Exception as e:
        logger.exception("query 실패")
        jobs.update(job_id, status="error", error=str(e))
        jobs.log(job_id, f"오류: {e}")


# ==============================================================================
# 10. FastAPI 앱 & 엔드포인트
# ==============================================================================
app = FastAPI(title="현진건 RAG 서버", version="1.0.0")

# Apps Script 는 서버 측(UrlFetchApp)에서 호출하므로 CORS 가 필수는 아니지만,
# 브라우저에서 직접 /query/stream 을 쓰는 클라이언트를 위해 허용해 둡니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_token(x_access_token: Optional[str] = Header(default=None)) -> None:
    """SERVER_ACCESS_TOKEN 이 설정된 경우에만 헤더 토큰을 검사합니다."""
    if SERVER_ACCESS_TOKEN and x_access_token != SERVER_ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="접근 토큰이 올바르지 않습니다.")


# ---- 요청 바디 스키마 ----------------------------------------------------------
class IngestRequest(BaseModel):
    filename: str = Field(..., description="원본 PDF 파일명(출처 표기용)")
    file_base64: str = Field(..., description="PDF 파일의 base64 문자열")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection_id: str = Field(..., description="/ingest 결과로 받은 컬렉션 ID")


# ---- 상태 확인 ----------------------------------------------------------------
@app.get("/")
@app.get("/health")
def health() -> Dict[str, Any]:
    """클라이언트의 '서버 연동 상태' 표시에 사용됩니다."""
    return {
        "status": "ok",
        "api_key_configured": bool(GOOGLE_API_KEY),
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
        "prompt_source": _prompt_source,
        "collections": len(list_collection_names()),
        "time": time.time(),
    }


# ---- 벡터 DB 목록 -------------------------------------------------------------
@app.get("/collections", dependencies=[Depends(verify_token)])
def collections() -> List[Dict[str, Any]]:
    out = []
    for name in list_collection_names():
        meta = chroma_client.get_collection(name).metadata or {}
        out.append({"collection_id": name, "filename": meta.get("filename", "")})
    return out


# ---- 벡터 DB 생성 (비동기 job) -------------------------------------------------
@app.post("/ingest", dependencies=[Depends(verify_token)])
def ingest(req: IngestRequest, background: BackgroundTasks) -> Dict[str, Any]:
    try:
        pdf_bytes = base64.b64decode(req.file_base64, validate=False)
    except Exception:
        raise HTTPException(status_code=400, detail="base64 디코딩에 실패했습니다.")
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="PDF 파일이 아닙니다.")
    if len(pdf_bytes) > MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"PDF는 {MAX_PDF_MB}MB 이하만 가능합니다.")

    job_id = jobs.create("ingest", filename=req.filename)
    background.add_task(run_ingest_job, job_id, req.filename, pdf_bytes)
    return {"job_id": job_id}


# ---- 질의 (비동기 job, Apps Script 폴링용) -------------------------------------
@app.post("/query", dependencies=[Depends(verify_token)])
def query(req: QueryRequest, background: BackgroundTasks) -> Dict[str, Any]:
    if req.collection_id not in list_collection_names():
        raise HTTPException(status_code=404, detail="벡터 DB(컬렉션)를 찾을 수 없습니다.")
    job_id = jobs.create("query", question=req.question)
    background.add_task(run_query_job, job_id, req.question, req.collection_id)
    return {"job_id": job_id}


# ---- job 증분 조회 ------------------------------------------------------------
@app.get("/jobs/{job_id}", dependencies=[Depends(verify_token)])
def job_status(job_id: str, log_offset: int = 0, token_offset: int = 0) -> Dict[str, Any]:
    try:
        return jobs.snapshot(job_id, log_offset, token_offset)
    except KeyError:
        raise HTTPException(status_code=404, detail="작업(job)을 찾을 수 없습니다.")


# ---- (선택) SSE 실시간 스트리밍 ------------------------------------------------
@app.post("/query/stream", dependencies=[Depends(verify_token)])
def query_stream(req: QueryRequest) -> StreamingResponse:
    """브라우저 fetch/EventSource 등 스트리밍을 직접 받을 수 있는 클라이언트용.
    이벤트 종류: queries → sources → token(여러 번) → done | error"""
    q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()

    def worker() -> None:
        try:
            retriever, retrieval_chain, generation_chain, _ = build_chains(req.collection_id)
            retrieved = retrieval_chain.invoke(req.question)
            q.put({"event": "queries", "data": retriever.generated_queries})
            q.put({"event": "sources", "data": docs_to_sources(retrieved["docs"])})
            handler = StreamHandler(on_token=lambda t: q.put({"event": "token", "data": t}))
            for _ in generation_chain.stream(retrieved, config={"callbacks": [handler]}):
                pass
            q.put({"event": "done", "data": ""})
        except HTTPException as e:
            q.put({"event": "error", "data": e.detail})
        except Exception as e:
            q.put({"event": "error", "data": str(e)})
        finally:
            q.put(None)  # 스트림 종료 신호

    threading.Thread(target=worker, daemon=True).start()

    def event_gen():
        while True:
            item = q.get()
            if item is None:
                break
            payload = json.dumps(item["data"], ensure_ascii=False)
            yield f"event: {item['event']}\ndata: {payload}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# 로컬 실행: python main.py  (Cloud Run 에서는 Dockerfile 의 uvicorn 명령 사용)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
