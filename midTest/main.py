"""
서울 도시계획 RAG 챗봇 서버 (Google Cloud Run)
- PDF 로딩: PyPDFLoader
- 임베딩: Google Generative AI Embeddings
- 벡터 DB: Chroma
- 검색기: MultiQueryRetriever (다중 질문 검색기)
- 질의 확장: RunnablePassthrough.assign 기반 체인
"""
import os
import base64
import logging
import tempfile
import threading
import uuid
from operator import itemgetter
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seoul-rag")

# ─────────────────────────── 설정 ───────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "/tmp/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "seoul_urban_planning")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))
NUM_QUERIES = int(os.getenv("NUM_QUERIES", "3"))

SYSTEM_ROLE = "당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요"

# ─────────────────────────── 프롬프트 ───────────────────────────
# 1) 질의 확장 프롬프트
EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 서울시 도시계획 검색 전문가입니다. 사용자의 질문을 벡터 검색에 유리하도록 확장하세요. "
     "질문의 의도는 유지하되 관련 도시계획 용어(예: 용도지역, 지구단위계획, 도시기본계획, "
     "정비사업, 용적률, 건폐율, 생활권계획 등)와 동의어를 보완하세요. "
     "설명 없이 확장된 질문 한 문단만 출력하세요."),
    ("human", "원래 질문: {question}"),
])

# 2) 다중 질문 생성 프롬프트
MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "당신은 서울시 도시계획 문서를 검색하는 AI 어시스턴트입니다.\n"
        f"아래 질문에 대해 벡터 DB에서 관련 문서를 폭넓게 찾을 수 있도록 서로 다른 관점의 질문 {NUM_QUERIES}개를 생성하세요.\n"
        "각 질문은 한 줄에 하나씩, 번호나 기호 없이 작성하세요.\n\n"
        "원래 질문: {question}"
    ),
)

# 3) 최종 답변 프롬프트
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     SYSTEM_ROLE + "\n\n"
     "아래 [참고 문서]의 내용을 근거로 정확하게 답변하세요. "
     "문서에 없는 내용은 일반적인 도시계획 지식임을 밝히고 추측을 사실처럼 말하지 마세요. "
     "가능하면 근거가 된 문서명과 페이지를 함께 제시하세요.\n\n"
     "[참고 문서]\n{context}"),
    ("human", "원래 질문: {question}\n(검색용 확장 질의: {expanded_question})"),
])

# ─────────────────────────── 전역 상태 ───────────────────────────
class RAGState:
    def __init__(self):
        self.lock = threading.Lock()
        self.vectorstore = None
        self.embeddings = None
        self.llm = None
        self.documents: List[dict] = []

    def init(self):
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
        if self.embeddings is None:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY
            )
        if self.llm is None:
            self.llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL, temperature=LLM_TEMPERATURE, google_api_key=GOOGLE_API_KEY
            )
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_DIR,
                collection_metadata={"hnsw:space": "cosine"},
            )

    def vector_count(self) -> int:
        try:
            return self.vectorstore._collection.count() if self.vectorstore else 0
        except Exception:
            return 0


state = RAGState()

# ─────────────────────────── FastAPI ───────────────────────────
app = FastAPI(title="Seoul Urban Planning RAG Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class UploadRequest(BaseModel):
    filename: str
    data: str  # base64 인코딩된 PDF


class QueryRequest(BaseModel):
    question: str


def format_docs(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source_name", "unknown")
        page = d.metadata.get("page", 0) + 1
        parts.append(f"[{i}] ({src}, p.{page})\n{d.page_content}")
    return "\n\n".join(parts)


def build_multi_query_retriever():
    base_retriever = state.vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K}
    )
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=state.llm,
        prompt=MULTI_QUERY_PROMPT,
        include_original=True,
    )


def build_rag_chain(mq_retriever):
    """
    RunnablePassthrough.assign으로 입력 dict를 그대로 통과시키며 단계별 결과를 누적:
    {question} → +expanded_question → +generated_queries → +docs → +context → +answer
    """
    expansion_chain = EXPANSION_PROMPT | state.llm |
