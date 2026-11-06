"""
================================================================================
Advanced RAG Backend Server (main.py)
- 아키텍처: FAISS Index + Multi-Query + Hybrid Search (Dense+BM25) + RAG Fusion (RRF)
- 연동 대상: Google Apps Script (Code.gs / index.html) & Google Cloud Run
================================================================================
"""

import base64
import hashlib
import io
import logging
import os
import re
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal

import faiss
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# Google Gemini SDK (선택적 로드 및 예외 방어)
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# ============================================================================
# 1. 실시간 StreamHandler 및 로깅 설정
# ============================================================================
class RealtimeStreamHandler(logging.StreamHandler):
    """
    서버 콘솔(stdout) 출력과 동시에 최근 진행 현황 로그를 인메모리 버퍼에 기록하여
    클라이언트(GAS)가 실시간 진행 상태 및 단계별 복잡도를 조회할 수 있도록 지원하는 핸들러.
    """

    def __init__(self, max_buffer_size: int = 200):
        super().__init__(stream=sys.stdout)
        self.log_buffer: deque = deque(maxlen=max_buffer_size)
        self.current_progress: Dict[str, Any] = {
            "stage": "IDLE",
            "percent": 0,
            "message": "서버 대기 중 (인덱스 미생성)",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            formatted = self.format(record)
            self.log_buffer.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "formatted": formatted,
                }
            )
        except Exception:
            self.handleError(record)

    def update_progress(self, stage: str, percent: int, message: str) -> None:
        """RAG 생성 및 검색 파이프라인 진행률을 업데이트하고 로그를 남김."""
        self.current_progress = {
            "stage": stage,
            "percent": max(0, min(100, percent)),
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"[{stage} | {percent}%] {message}")


stream_handler = RealtimeStreamHandler()
stream_handler.setFormatter(
    logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)

logger = logging.getLogger("AdvancedRAG")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(stream_handler)


# ============================================================================
# 2. 알고리즘 복잡도 상수 및 메타데이터 정의
# ============================================================================
ALGORITHM_COMPLEXITY = {
    "FAISS_IndexFlatL2": {
        "build_time": "O(N * d)",
        "search_time": "O(N * d)",
        "space": "O(N * d)",
        "description": "모든 벡터와 정확한 L2 유클리드 거리를 전수 계산 (100% 재현율 보장)",
    },
    "FAISS_IndexHNSWFlat": {
        "build_time": "O(N * log(N) * d)",
        "search_time": "O(log(N) * d)",
        "space": "O(N * (d + M_hnsw))",
        "description": "계층적 탐색 가능한 소세계 그래프(HNSW) 기반 초고속 근사 최근접 이웃(ANN) 검색",
    },
    "Multi_Query_Expansion": {
        "time": "O(M * T_LLM)",
        "space": "O(M * |q|)",
        "description": "단일 쿼리를 M개의 다각도 파생 쿼리로 확장하여 어휘 불일치 및 검색 누락 완화",
    },
    "Hybrid_Search_BM25": {
        "build_time": "O(N * L_avg)",
        "search_time": "O(M * N * |q|)",
        "space": "O(V + N * L_avg)",
        "description": "역색인 기반 TF-IDF 포화도 및 문서 길이 정규화를 적용한 희소(Sparse) 키워드 검색",
    },
    "RAG_Fusion_RRF": {
        "time": "O(M * K * log(K))",
        "space": "O(M * K)",
        "description": "Reciprocal Rank Fusion 공식 Σ 1/(k + rank_i(d))으로 이종 검색기 순위를 정규화 없이 융합",
    },
}


# ============================================================================
# 3. 인메모리 Advanced RAG 엔진 (FAISS + BM25 + Multi-Query + RRF)
# ============================================================================
class DocumentChunk(BaseModel):
    """벡터 DB 및 BM25 인덱스에 적재되는 단일 문서 청크 스키마"""
    chunk_id: int
    source: str
    page: int
    text: str
    char_count: int


class AdvancedRAGEngine:
    """
    FAISS(Dense) + BM25(Sparse) + Multi-Query + RRF(RAG Fusion) 통합 파이프라인 엔진
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim: int = embedding_dim
        self.index_type: Literal["IndexFlatL2", "IndexHNSWFlat"] = "IndexFlatL2"
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.loaded_files: List[Dict[str, Any]] = []
        self.last_build_metrics: Dict[str, Any] = {}

    def _get_gemini_client(self) -> Optional[Any]:
        """환경변수 GEMINI_API_KEY를 확인하여 Gemini 클라이언트를 반환"""
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if HAS_GENAI and api_key:
            return genai.Client(api_key=api_key)
        return None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        한국어/영문/숫자 혼합 텍스트를 BM25 검색용 토큰 리스트로 분리.
        어절 단위 토큰과 2-gram 음절 조각을 함께 생성하여 한국어 조사 결합에 강건하게 동작.
        시간 복잡도: O(|text|)
        """
        cleaned = re.sub(r"[^\w\s가-힣]", " ", text.lower())
        words = [w for w in cleaned.split() if len(w) >= 1]
        tokens = list(words)
        for w in words:
            if len(w) >= 3:
                # 복합명사/조사 분리를 위한 바이그램(bi-gram) 추가
                tokens.extend([w[i : i + 2] for i in range(len(w) - 1)])
        return tokens if tokens else ["empty"]

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        텍스트 배열을 고차원 정규화 밀집 벡터(Dense Vector, float32)로 변환.
        - Gemini API 키가 설정된 경우: text-embedding-004 모델 사용
        - 미설정 또는 오프라인 테스트 환경인 경우: 토큰 해시 투영(Feature Hashing) 기반 결정론적 임베딩 생성
        """
        client = self._get_gemini_client()
        if client is not None:
            try:
                vectors = []
                batch_size = 32
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    resp = client.models.embed_content(
                        model="text-embedding-004",
                        contents=batch,
                    )
                    for emb in resp.embeddings:
                        vec = np.array(emb.values, dtype=np.float32)
                        if vec.shape[0] != self.embedding_dim:
                            # 차원 자동 동기화
                            self.embedding_dim = int(vec.shape[0])
                        norm = np.linalg.norm(vec)
                        vectors.append(vec / (norm + 1e-10))
                return np.vstack(vectors).astype(np.float32)
            except Exception as e:
                logger.warning(f"Gemini 임베딩 호출 실패, 로컬 해시 임베딩으로 폴백합니다: {e}")

        # 결정론적 Feature Hashing 임베딩 (O(N * |tokens|))
        mat = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = self._tokenize(text)
            for tok in tokens:
                digest = hashlib.md5(tok.encode("utf-8")).digest()
                dim_idx = int.from_bytes(digest[:4], "little") % self.embedding_dim
                sign = 1.0 if (digest[4] % 2 == 0) else -1.0
                mat[row_idx, dim_idx] += sign
            norm = np.linalg.norm(mat[row_idx])
            if norm > 0:
                mat[row_idx] /= norm
        return mat

    def build_index_from_pages(
        self,
        filename: str,
        pages_text: List[Dict[str, Any]],
        index_type: Literal["IndexFlatL2", "IndexHNSWFlat"] = "IndexFlatL2",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> Dict[str, Any]:
        """
        추출된 PDF 페이지 텍스트들을 청킹(Chunking)한 뒤,
        FAISS 인덱스(IndexFlatL2 또는 IndexHNSWFlat)와 BM25 희소 인덱스를 동시에 구축.
        """
        t_start = time.perf_counter()
        stream_handler.update_progress("CHUNKING", 20, f"'{filename}' 문서 청킹 진행 중...")

        new_chunks: List[DocumentChunk] = []
        chunk_id_counter = 0

        for page_info in pages_text:
            page_num = page_info["page"]
            raw_text = re.sub(r"\s+", " ", page_info["text"]).strip()
            if not raw_text:
                continue

            # 슬라이딩 윈도우 기반 청킹
            step = max(1, chunk_size - chunk_overlap)
            for start in range(0, len(raw_text), step):
                segment = raw_text[start : start + chunk_size].strip()
                if len(segment) < 20 and len(raw_text) >= 20:
                    continue
                new_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id_counter,
                        source=filename,
                        page=page_num,
                        text=segment,
                        char_count=len(segment),
                    )
                )
                chunk_id_counter += 1

        if not new_chunks:
            raise ValueError("PDF에서 추출 가능한 텍스트가 없습니다. 스캔 이미지 PDF인지 확인하세요.")

        stream_handler.update_progress(
            "EMBEDDING", 50, f"총 {len(new_chunks)}개 청크 고차원 임베딩 생성 중..."
        )
        texts = [c.text for c in new_chunks]
        embeddings = self._embed_texts(texts)
        num_vectors, dim = embeddings.shape

        # FAISS 인덱스 생성 (IndexFlatL2 또는 IndexHNSWFlat)
        stream_handler.update_progress(
            "FAISS_BUILD", 75, f"FAISS [{index_type}] 인덱스 구축 및 벡터 적재 중 (차원={dim})..."
        )
        self.index_type = index_type
        if index_type == "IndexHNSWFlat":
            # HNSW 그래프 연결 수 M=32 설정
            hnsw_m = 32
            index = faiss.IndexHNSWFlat(dim, hnsw_m)
            index.hnsw.efConstruction = 64
            index.hnsw.efSearch = 32
        else:
            index = faiss.IndexFlatL2(dim)

        index.add(embeddings)
        self.faiss_index = index

        # BM25 Sparse 인덱스 생성
        stream_handler.update_progress(
            "BM25_BUILD", 90, "BM25(Sparse) 키워드 역색인 구축 중..."
        )
        self.tokenized_corpus = [self._tokenize(t) for t in texts]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        self.chunks = new_chunks

        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        faiss_algo_key = f"FAISS_{index_type}"

        file_record = {
            "filename": filename,
            "pages": len(pages_text),
            "chunks": len(new_chunks),
            "index_type": index_type,
            "embedding_dim": dim,
            "build_time_ms": elapsed_ms,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.loaded_files = [file_record]
        self.last_build_metrics = {
            **file_record,
            "complexity": {
                "faiss": ALGORITHM_COMPLEXITY[faiss_algo_key],
                "bm25": ALGORITHM_COMPLEXITY["Hybrid_Search_BM25"],
            },
        }

        stream_handler.update_progress(
            "COMPLETED",
            100,
            f"벡터 DB 구축 완료: {filename} ({len(new_chunks)}개 청크, {index_type}, {elapsed_ms}ms)",
        )
        return self.last_build_metrics

    def expand_multi_queries(self, original_query: str, num_queries: int = 3) -> List[str]:
        """
        [Multi-Query Layer]
        사용자의 단일 입력 쿼리를 서로 다른 관점(의미적 동의어, 핵심 키워드 중심, 문맥적 배경)의
        파생 쿼리들로 확장하여 단일 쿼리 검색의 어휘 불일치(Lexical Gap)와 검색 누락을 방지.
        """
        queries = [original_query.strip()]
        client = self._get_gemini_client()

        if client is not None:
            try:
                prompt = (
                    "당신은 RAG 검색 최적화 전문가입니다. 아래 사용자 질문을 바탕으로 "
                    f"벡터 검색과 키워드 검색의 재현율(Recall)을 높일 수 있는 서로 다른 관점의 파생 검색 쿼리 {num_queries}개를 생성하세요.\n"
                    "- 번호나 불릿 없이 한 줄에 하나씩만 출력하세요.\n"
                    f"사용자 질문: {original_query}"
                )
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                if resp and resp.text:
                    lines = [
                        re.sub(r"^[\d\-\*\.\)\s]+", "", line).strip()
                        for line in resp.text.strip().splitlines()
                        if line.strip()
                    ]
                    for q in lines:
                        if q and q not in queries and len(queries) < num_queries + 1:
                            queries.append(q)
            except Exception as e:
                logger.warning(f"LLM Multi-Query 확장 중 폴백 전환: {e}")

        # LLM 미사용 환경이거나 파생 쿼리가 부족할 때 규칙 기반 다각도 쿼리 보강
        tokens = [w for w in re.sub(r"[^\w\s가-힣]", " ", original_query).split() if len(w) >= 2]
        keyword_focused = " ".join(tokens) if tokens else original_query
        perspective_candidates = [
            f"{keyword_focused} 핵심 개념 및 정의",
            f"{original_query} 관련 주요 특징 및 근거 데이터",
            f"{keyword_focused} 상세 절차 및 사례 분석",
        ]
        for cand in perspective_candidates:
            if len(queries) >= num_queries + 1:
                break
            if cand not in queries:
                queries.append(cand)

        return queries[: num_queries + 1]

    def hybrid_search_and_rrf(
        self,
        original_query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """
        [Hybrid Search + RAG Fusion (RRF) Pipeline]
        1단계 (Multi-Query): 단일 쿼리를 M개의 파생 쿼리로 확장
        2단계 (Hybrid Search): 각 쿼리에 대해 FAISS(Dense) 검색 + BM25(Sparse) 검색 병행 수행
        3단계 (RRF Fusion): 수집된 2*M개의 순위 리스트를 Reciprocal Rank Fusion 공식으로 통합 및 재정렬
        """
        if self.faiss_index is None or self.bm25_index is None or not self.chunks:
            raise RuntimeError("벡터 DB가 아직 생성되지 않았습니다. 먼저 PDF 파일을 업로드하여 벡터 DB를 생성하세요.")

        t0 = time.perf_counter()
        pipeline_trace: List[str] = []

        # 1. Multi-Query 확장
        t_mq_start = time.perf_counter()
        expanded_queries = self.expand_multi_queries(original_query, num_queries=3)
        t_mq_ms = round((time.perf_counter() - t_mq_start) * 1000, 2)
        pipeline_trace.append(
            f"[Multi-Query] 1개 원본 쿼리 -> {len(expanded_queries)}개 다각도 쿼리로 확장 ({t_mq_ms}ms)"
        )

        # 2. 각 확장 쿼리별 FAISS Dense 검색 & BM25 Sparse 검색 수행
        t_hybrid_start = time.perf_counter()
        num_chunks = len(self.chunks)
        candidate_k = min(num_chunks, max(top_k * 3, 10))

        # 쿼리 벡터 일괄 임베딩
        query_embeddings = self._embed_texts(expanded_queries)
        distances_mat, indices_mat = self.faiss_index.search(query_embeddings, candidate_k)

        # 문서별 RRF 점수 및 순위 메타데이터 누적 구조체
        rrf_scores: Dict[int, float] = {}
        doc_rank_details: Dict[int, Dict[str, Any]] = {}

        for q_idx, q_text in enumerate(expanded_queries):
            # (A) FAISS Dense 검색 결과 순위 반영
            for rank_zero_based in range(candidate_k):
                doc_id = int(indices_mat[q_idx][rank_zero_based])
                if doc_id < 0 or doc_id >= num_chunks:
                    continue
                l2_dist = float(distances_mat[q_idx][rank_zero_based])
                dense_sim = round(1.0 / (1.0 + max(0.0, l2_dist)), 4)
                rank_1based = rank_zero_based + 1

                # RRF 공식: weight / (rrf_k + rank)
                rrf_contrib = dense_weight / (rrf_k + rank_1based)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_contrib

                info = doc_rank_details.setdefault(
                    doc_id,
                    {
                        "dense_best_rank": None,
                        "sparse_best_rank": None,
                        "dense_similarity": 0.0,
                        "bm25_score": 0.0,
                        "matched_queries": set(),
                        "rank_contributions": [],
                    },
                )
                if info["dense_best_rank"] is None or rank_1based < info["dense_best_rank"]:
                    info["dense_best_rank"] = rank_1based
                if dense_sim > info["dense_similarity"]:
                    info["dense_similarity"] = dense_sim
                info["matched_queries"].add(q_text)
                info["rank_contributions"].append(
                    f"Q{q_idx+1}-Dense(#{rank_1based}, +{rrf_contrib:.5f})"
                )

            # (B) BM25 Sparse 키워드 검색 결과 순위 반영
            q_tokens = self._tokenize(q_text)
            bm25_scores = self.bm25_index.get_scores(q_tokens)
            sorted_bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]

            for rank_zero_based, doc_id_np in enumerate(sorted_bm25_indices):
                doc_id = int(doc_id_np)
                b_score = float(bm25_scores[doc_id])
                if b_score <= 0 and rank_zero_based > 0:
                    continue
                rank_1based = rank_zero_based + 1

                rrf_contrib = sparse_weight / (rrf_k + rank_1based)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_contrib

                info = doc_rank_details.setdefault(
                    doc_id,
                    {
                        "dense_best_rank": None,
                        "sparse_best_rank": None,
                        "dense_similarity": 0.0,
                        "bm25_score": 0.0,
                        "matched_queries": set(),
                        "rank_contributions": [],
                    },
                )
                if info["sparse_best_rank"] is None or rank_1based < info["sparse_best_rank"]:
                    info["sparse_best_rank"] = rank_1based
                if b_score > info["bm25_score"]:
                    info["bm25_score"] = round(b_score, 4)
                info["matched_queries"].add(q_text)
                info["rank_contributions"].append(
                    f"Q{q_idx+1}-BM25(#{rank_1based}, +{rrf_contrib:.5f})"
                )

        t_hybrid_ms = round((time.perf_counter() - t_hybrid_start) * 1000, 2)
        pipeline_trace.append(
            f"[Hybrid Search] FAISS({self.index_type}) + BM25 병행 검색 완료 (후보군 {len(rrf_scores)}개, {t_hybrid_ms}ms)"
        )

        # 3. RRF 순위 융합 및 상위 top_k 재정렬 (Re-ranking)
        t_rrf_start = time.perf_counter()
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:top_k]

        retrieved_docs: List[Dict[str, Any]] = []
        for final_rank, doc_id in enumerate(sorted_doc_ids, start=1):
            chunk = self.chunks[doc_id]
            detail = doc_rank_details[doc_id]
            retrieved_docs.append(
                {
                    "final_rank": final_rank,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    "snippet": chunk.text,
                    "rrf_score": round(rrf_scores[doc_id], 6),
                    "dense_best_rank": detail["dense_best_rank"],
                    "sparse_best_rank": detail["sparse_best_rank"],
                    "dense_similarity": detail["dense_similarity"],
                    "bm25_score": detail["bm25_score"],
                    "matched_queries": list(detail["matched_queries"]),
                    "rank_contributions": detail["rank_contributions"][:6],
                }
            )

        t_rrf_ms = round((time.perf_counter() - t_rrf_start) * 1000, 2)
        pipeline_trace.append(
            f"[RAG Fusion (RRF)] k={rrf_k} 기준 상위 {len(retrieved_docs)}개 최적 문서 재정렬 완료 ({t_rrf_ms}ms)"
        )

        # 4. LLM 답변 생성 (출처 인용 포함)
        t_llm_start = time.perf_counter()
        llm_answer = self._generate_grounded_answer(original_query, retrieved_docs)
        t_llm_ms = round((time.perf_counter() - t_llm_start) * 1000, 2)
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        pipeline_trace.append(f"[LLM Answer] 문맥 기반 답변 및 출처 생성 완료 ({t_llm_ms}ms, 총 {total_ms}ms)")

        for log_msg in pipeline_trace:
            logger.info(log_msg)

        faiss_algo_key = f"FAISS_{self.index_type}"
        return {
            "query": original_query,
            "expanded_queries": expanded_queries,
            "retrieved_documents": retrieved_docs,
            "llm_answer": llm_answer,
            "timing_ms": {
                "multi_query_ms": t_mq_ms,
                "hybrid_search_ms": t_hybrid_ms,
                "rrf_fusion_ms": t_rrf_ms,
                "llm_generation_ms": t_llm_ms,
                "total_ms": total_ms,
            },
            "complexity_analysis": {
                "parameters": {
                    "N_chunks": num_chunks,
                    "d_dim": self.embedding_dim,
                    "M_queries": len(expanded_queries),
                    "K_candidates": candidate_k,
                    "rrf_k": rrf_k,
                    "index_type": self.index_type,
                },
                "layers": {
                    "multi_query": ALGORITHM_COMPLEXITY["Multi_Query_Expansion"],
                    "faiss_dense": ALGORITHM_COMPLEXITY[faiss_algo_key],
                    "bm25_sparse": ALGORITHM_COMPLEXITY["Hybrid_Search_BM25"],
                    "rrf_fusion": ALGORITHM_COMPLEXITY["RAG_Fusion_RRF"],
                },
            },
            "pipeline_trace": pipeline_trace,
        }

    def _generate_grounded_answer(
        self, query: str, retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """
        RRF로 재정렬된 상위 문서 스니펫만을 근거로 LLM 답변을 생성하며,
        각 문단 끝에 [출처: 파일명 p.페이지 #청크ID] 형태의 명확한 출처를 표기.
        """
        if not retrieved_docs:
            return "검색된 관련 문서가 없어 답변을 생성할 수 없습니다."

        context_blocks = []
        for doc in retrieved_docs:
            citation = f"[출처: {doc['source']} p.{doc['page']} #Chunk-{doc['chunk_id']} | RRF={doc['rrf_score']}]"
            context_blocks.append(f"문서 #{doc['final_rank']} {citation}\n내용: {doc['snippet']}")

        joined_context = "\n\n".join(context_blocks)
        client = self._get_gemini_client()

        if client is not None:
            try:
                prompt = (
                    "당신은 신뢰할 수 있는 Advanced RAG 어시스턴트입니다.\n"
                    "반드시 아래 제공된 [RRF 상위 검색 문서]의 내용만을 근거로 사용자의 질문에 한국어로 명확하고 체계적으로 답변하세요.\n"
                    "답변의 각 핵심 근거마다 반드시 [출처: 파일명 p.페이지 #Chunk-번호] 형식으로 출처를 명시하세요.\n\n"
                    f"[RRF 상위 검색 문서]\n{joined_context}\n\n"
                    f"[사용자 질문]\n{query}"
                )
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                if resp and resp.text:
                    return resp.text.strip()
            except Exception as e:
                logger.warning(f"Gemini 답변 생성 실패, 구조화 요약 모드로 폴백합니다: {e}")

        # 로컬/폴백 구조화 답변 생성 (검색된 RRF 상위 근거 문맥 + 출처 명시)
        summary_lines = [
            f"질문하신 **\"{query}\"**에 대해 RRF(Reciprocal Rank Fusion) 순위 융합으로 선별된 상위 {len(retrieved_docs)}개 핵심 문맥 기반 답변입니다.\n"
        ]
        for doc in retrieved_docs[:3]:
            citation = f"[출처: {doc['source']} p.{doc['page']} #Chunk-{doc['chunk_id']}]"
            summary_lines.append(
                f"• **핵심 근거 #{doc['final_rank']} (RRF 점수: `{doc['rrf_score']}`)**: "
                f"{doc['snippet']} {citation}"
            )
        summary_lines.append(
            "\n※ 위 답변은 FAISS 고차원 벡터 검색과 BM25 키워드 검색을 RRF 공식으로 융합하여 추출한 최적 문맥에 기반합니다."
        )
        return "\n\n".join(summary_lines)


# 싱글톤 엔진 인스턴스 생성
rag_engine = AdvancedRAGEngine()


# ============================================================================
# 4. FastAPI 애플리케이션 및 요청/응답 스키마 정의
# ============================================================================
app = FastAPI(
    title="Advanced RAG Server (FAISS + Multi-Query + Hybrid Search + RRF)",
    description="FAISS Dense Vector + BM25 Sparse + Multi-Query Expansion + Reciprocal Rank Fusion API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestBase64Request(BaseModel):
    """GAS(UrlFetchApp)에서 PDF 파일(Base64) 또는 텍스트로 벡터 DB 생성을 요청할 때 사용하는 스키마"""
    filename: str = Field(..., description="업로드할 PDF 파일명")
    pdf_base64: Optional[str] = Field(None, description="Base64 인코딩된 PDF 바이너리 데이터")
    raw_text: Optional[str] = Field(None, description="PDF 텍스트 직접 전달 시 사용 (샘플 테스트 호환)")
    index_type: Literal["IndexFlatL2", "IndexHNSWFlat"] = Field(
        "IndexFlatL2", description="FAISS 인덱스 알고리즘 종류"
    )
    chunk_size: int = Field(500, ge=100, le=2000, description="청크 당 최대 글자 수")
    chunk_overlap: int = Field(100, ge=0, le=500, description="청크 간 중첩 글자 수")


class QueryRequest(BaseModel):
    """사용자 질의 및 하이브리드 RRF 검색 파라미터 스키마"""
    query: str = Field(..., min_length=1, description="사용자의 검색 질의")
    top_k: int = Field(5, ge=1, le=20, description="RRF 재정렬 후 반환할 최종 상위 문서 수")
    rrf_k: int = Field(60, ge=1, le=200, description="RRF 수식의 평활화 상수 k (기본값 60)")
    dense_weight: float = Field(1.0, ge=0.0, le=5.0, description="FAISS Dense 검색 가중치")
    sparse_weight: float = Field(1.0, ge=0.0, le=5.0, description="BM25 Sparse 검색 가중치")


def build_error_response(
    status_code: int,
    error_code: str,
    stage: str,
    message: str,
    hint: str,
    detail: Optional[str] = None,
) -> JSONResponse:
    """클라이언트(GAS)에서 에러 원인과 해결 방법을 정확히 렌더링할 수 있도록 표준화된 에러 JSON 반환"""
    logger.error(f"[{error_code} @ {stage}] {message} | Detail: {detail}")
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "error": {
                "code": error_code,
                "http_status": status_code,
                "stage": stage,
                "message": message,
                "hint": hint,
                "detail": detail or message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        },
    )


# ============================================================================
# 5. API 엔드포인트 구현
# ============================================================================
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """서버 동작 상태, 인덱스 적재 현황, 알고리즘 복잡도 메타데이터 반환"""
    return {
        "status": "ok",
        "server_state": "READY" if rag_engine.faiss_index is not None else "EMPTY_INDEX",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "index_info": {
            "is_ready": rag_engine.faiss_index is not None,
            "index_type": rag_engine.index_type if rag_engine.faiss_index else None,
            "total_chunks": len(rag_engine.chunks),
            "embedding_dim": rag_engine.embedding_dim,
            "loaded_files": rag_engine.loaded_files,
        },
        "progress": stream_handler.current_progress,
        "recent_logs": list(stream_handler.log_buffer)[-15:],
        "algorithm_complexity": ALGORITHM_COMPLEXITY,
    }


@app.get("/progress")
async def get_progress() -> Dict[str, Any]:
    """RAG 벡터 DB 생성 진행률 및 실시간 StreamHandler 로그 버퍼 조회"""
    return {
        "status": "ok",
        "progress": stream_handler.current_progress,
        "logs": list(stream_handler.log_buffer)[-30:],
    }


@app.get("/stream-logs")
async def stream_logs() -> StreamingResponse:
    """Server-Sent Events(SSE) 방식의 실시간 로그 스트리밍 엔드포인트"""
    def event_generator():
        for entry in list(stream_handler.log_buffer):
            yield f"data: {entry['formatted']}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/ingest")
async def ingest_document(payload: IngestBase64Request) -> Any:
    """
    [벡터 DB 생성 엔드포인트]
    GAS에서 전달받은 PDF(Base64) 또는 텍스트를 파싱하여 FAISS 인덱스 및 BM25 인덱스를 생성.
    """
    try:
        stream_handler.update_progress(
            "RECEIVING", 10, f"'{payload.filename}' 파일 수신 및 유효성 검증 중..."
        )
        pages_text: List[Dict[str, Any]] = []

        if payload.pdf_base64:
            try:
                raw_b64 = payload.pdf_base64.split(",")[-1].strip()
                pdf_bytes = base64.b64decode(raw_b64)
                reader = PdfReader(io.BytesIO(pdf_bytes))
                for idx, page in enumerate(reader.pages, start=1):
                    extracted = page.extract_text() or ""
                    if extracted.strip():
                        pages_text.append({"page": idx, "text": extracted})
            except Exception as pdf_err:
                return build_error_response(
                    status_code=400,
                    error_code="PDF_PARSE_ERROR",
                    stage="PDF_EXTRACTION",
                    message=f"PDF 파일을 읽는 중 오류가 발생했습니다: {str(pdf_err)}",
                    hint="파일이 손상되었거나 암호가 걸려 있는지 확인하세요.",
                    detail=traceback.format_exc(),
                )
        elif payload.raw_text:
            # 페이지 구분자 또는 단락 기준으로 페이지 분할
            paragraphs = [p.strip() for p in payload.raw_text.split("\n\n") if p.strip()]
            for idx, para in enumerate(paragraphs, start=1):
                pages_text.append({"page": idx, "text": para})
        else:
            return build_error_response(
                status_code=400,
                error_code="EMPTY_DOCUMENT_PAYLOAD",
                stage="VALIDATION",
                message="PDF 데이터(pdf_base64) 또는 텍스트(raw_text)가 전달되지 않았습니다.",
                hint="먼저 PDF 파일을 선택한 후 벡터 DB 생성 버튼을 클릭하세요.",
            )

        metrics = rag_engine.build_index_from_pages(
            filename=payload.filename,
            pages_text=pages_text,
            index_type=payload.index_type,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
        )

        return {
            "status": "ok",
            "message": f"'{payload.filename}' 기반 FAISS({payload.index_type}) + BM25 하이브리드 벡터 DB 생성이 완료되었습니다.",
            "metrics": metrics,
            "progress": stream_handler.current_progress,
            "logs": list(stream_handler.log_buffer)[-20:],
        }

    except ValueError as ve:
        stream_handler.update_progress("ERROR", 0, str(ve))
        return build_error_response(
            status_code=422,
            error_code="EMPTY_EXTRACTED_TEXT",
            stage="CHUNKING",
            message=str(ve),
            hint="텍스트가 포함된 PDF 파일인지 확인하세요 (스캔본 이미지는 OCR 처리가 필요합니다).",
        )
    except Exception as e:
        stream_handler.update_progress("ERROR", 0, f"벡터 DB 생성 실패: {str(e)}")
        return build_error_response(
            status_code=500,
            error_code="INDEX_BUILD_FAILURE",
            stage="FAISS_BM25_BUILD",
            message=f"벡터 DB 인덱스 생성 중 서버 내부 오류가 발생했습니다: {str(e)}",
            hint="서버 메모리 상태 또는 입력 파일 형식을 점검하세요.",
            detail=traceback.format_exc(),
        )


@app.post("/query")
async def query_rag(payload: QueryRequest) -> Any:
    """
    [Multi-Query + Hybrid Search + RAG Fusion (RRF) 질의 엔드포인트]
    정제된 상위 문서 스니펫, RRF 점수/순위 메타데이터, LLM 답변, 알고리즘 복잡도를 JSON으로 반환.
    """
    try:
        result = rag_engine.hybrid_search_and_rrf(
            original_query=payload.query,
            top_k=payload.top_k,
            rrf_k=payload.rrf_k,
            dense_weight=payload.dense_weight,
            sparse_weight=payload.sparse_weight,
        )
        return {
            "status": "ok",
            "data": result,
            "logs": list(stream_handler.log_buffer)[-15:],
        }
    except RuntimeError as re_err:
        return build_error_response(
            status_code=400,
            error_code="INDEX_NOT_INITIALIZED",
            stage="PRE_SEARCH_CHECK",
            message=str(re_err),
            hint="질의하기 전에 먼저 [1. PDF 파일 요청/선택] 후 [2. 벡터 DB 생성] 버튼을 눌러 인덱스를 구축하세요.",
        )
    except Exception as e:
        return build_error_response(
            status_code=500,
            error_code="RAG_PIPELINE_ERROR",
            stage="HYBRID_RRF_SEARCH",
            message=f"RAG Fusion 검색 파이프라인 실행 중 오류가 발생했습니다: {str(e)}",
            hint="질의 문자열이나 서버 로그를 확인하세요.",
            detail=traceback.format_exc(),
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

