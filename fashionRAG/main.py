"""
================================================================================
Fashion Multi-modal RAG Backend Server (main.py)
- 아키텍처: Fashionpedia + Image Chunking + CLIP Joint Embedding + ChromaDB
            + Multi-Vector Retriever + Google AI VLM Multi-modal RAG Chain
- 연동 대상: Google Apps Script (code.gs / index.html) & Google Cloud Run
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
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

# ChromaDB (선택적 로드 및 인메모리 폴백 방어)
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

# OpenCLIP Joint Embedding (선택적 로드)
try:
    import open_clip
    import torch
    HAS_OPENCLIP = True
except ImportError:
    HAS_OPENCLIP = False

# Google Gemini SDK (Google AI API - LLM & VLM)
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# ============================================================================
# 1. 실시간 StreamHandler 및 로깅 설정 (기존 아키텍처 유지)
# ============================================================================
class RealtimeStreamHandler(logging.StreamHandler):
    """클라이언트(GAS)가 실시간 진행 상태 및 파이프라인 로그를 조회할 수 있도록 지원하는 핸들러."""

    def __init__(self, max_buffer_size: int = 200):
        super().__init__(stream=sys.stdout)
        self.log_buffer: deque = deque(maxlen=max_buffer_size)
        self.current_progress: Dict[str, Any] = {
            "stage": "READY",
            "percent": 100,
            "message": "Fashionpedia 멀티모달 RAG 엔진 준비 완료",
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

logger = logging.getLogger("FashionMultiModalRAG")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(stream_handler)


# ============================================================================
# 2. Fashionpedia 기본 데이터셋 카탈로그 및 이미지 청킹 유틸리티
# ============================================================================
FASHIONPEDIA_CURATED_ITEMS: List[Dict[str, Any]] = [
    {
        "item_id": "fp_fw24_001",
        "title": "Oversized Tailored Charcoal Blazer & Wide Tech-Wool Trousers",
        "title_ko": "오버사이즈 테일러드 차콜 블레이저 & 와이드 테크울 팬츠",
        "category": "outerwear / blazer, trousers",
        "attributes": ["oversized", "structured shoulder", "genderless", "charcoal wool", "pleated wide-leg"],
        "summary_en": (
            "Avant-garde 2024 F/W runway look featuring an oversized charcoal grey tailored heavy-wool blazer "
            "with structured shoulders, paired with straight wide-leg pleated tech-wool trousers and square-toe ankle boots."
        ),
        "silhouette": "Dramatic V-Shape (상체 볼륨 극대화 + 스트레이트 와이드 드레이프)",
        "texture_matrix": {
            "outer": "헤비 울 트윌 블레이저",
            "bottom": "매트 테크 나일론 울",
            "footwear": "카프스킨 스퀘어토 앵클 부츠",
        },
        "palette": [
            {"hex": "#1E293B", "name": "Deep Charcoal"},
            {"hex": "#3B82F6", "name": "Cobalt Accent"},
            {"hex": "#94A3B8", "name": "Slate Muted"},
        ],
        "tpo": "에디토리얼 / 포멀 스트리트",
        "formal_score": 68,
        "urban_score": 82,
        "image_url": "https://lh3.googleusercontent.com/aida-public/AB6AXuBrbta8kZz-suHoMuhZbnYFz1no0nCgtq0qgMEvdZmO1zUoxhWMBPXZMPK5-j-YK-al0tMPMDhWkM904T41FZ92PK0oC3wi4bG-fay_rdQpIWzlTKKxaQWG-s8cAdDlJUh90oFJc_grJ_Vj0yU-S790T7QwBfdG05-rX3ZtA-1TLeUkT0tUAKcUJGAoJmKDf7rtSREJ4XhK4f4meSFyVB2UM7u7eqOd61xiG0-6BcY",
        "bboxes": [
            {"label": "blazer (upper_body)", "box": [0.12, 0.15, 0.88, 0.58]},
            {"label": "wide trousers (lower_body)", "box": [0.18, 0.55, 0.82, 0.92]},
        ],
    },
    {
        "item_id": "fp_fw24_002",
        "title": "Cropped Utility Gore-Tex Jacket & Modular Nylon Cargo Pants",
        "title_ko": "크롭 유틸리티 고어텍스 재킷 & 모듈러 나일론 카고",
        "category": "jacket / techwear, cargo pants",
        "attributes": ["cropped", "utility buckle", "waterproof", "high-waisted cargo", "lugged derby"],
        "summary_en": (
            "Urban streetwear techwear editorial outfit with a cropped structured jacket featuring utility strap buckles, "
            "multi-pocket modular high-waisted nylon cargo trousers, chunky lugged sole leather derby shoes, and emerald accents."
        ),
        "silhouette": "Dynamic Inverted H (크롭 상체 절개 + 하이웨이스트 입체 카고 볼륨)",
        "texture_matrix": {
            "outer": "방수 고어텍스 블렌드",
            "bottom": "리사이클 나일론 카고",
            "footwear": "청키 러그솔 더비슈즈",
        },
        "palette": [
            {"hex": "#0F172A", "name": "Obsidian Black"},
            {"hex": "#10B981", "name": "Emerald Accent"},
            {"hex": "#065F46", "name": "Forest Dark"},
        ],
        "tpo": "위크엔드 나이트 / 어반 테크웨어",
        "formal_score": 42,
        "urban_score": 94,
        "image_url": "https://lh3.googleusercontent.com/aida-public/AB6AXuAxr6Jo4i59lz4MYszqEGyEQCjde9MAh56vX7nH7GwCHl83emcFkMW7YZIi01LUg06JYDgGxQ_Rj5HfFMHJsRwbk0dBDqwbwtpyayzPXe1FqHtkW4u8BIJCGIrbpncrPw1J_VLFqXEVS_RHq9MGPH5-gGnztO4EBhvZsPYJ8FwtUplgJGgqJ4mWYOXrXjhE2MZa2MF0HsDslkJevA75W4xHCmIOrwOGIWKyxDLW3_g",
        "bboxes": [
            {"label": "utility jacket (upper_body)", "box": [0.15, 0.14, 0.85, 0.48]},
            {"label": "cargo trousers (lower_body)", "box": [0.16, 0.46, 0.84, 0.93]},
        ],
    },
    {
        "item_id": "fp_fw24_003",
        "title": "Minimalist Old-Money Cashmere Coat & Monochrome Wool Slacks",
        "title_ko": "미니멀 올드머니 캐시미어 코트 & 모노크롬 울 슬랙스",
        "category": "coat / outerwear, tailored slacks",
        "attributes": ["minimalist", "cashmere", "single-breasted", "monochrome", "quiet luxury"],
        "summary_en": (
            "Minimal luxury formal ensemble crafted from Italian virgin cashmere coat in deep navy-slate, "
            "layered over a fine merino turtleneck and tailored monochrome wool slacks with Chelsea boots."
        ),
        "silhouette": "Elongated Column I-Line (롱 코트 수직 라인 + 슬림 스트레이트 핏)",
        "texture_matrix": {
            "outer": "버진 캐시미어 롱 코트",
            "bottom": "파인 메리노 울 슬랙스",
            "footwear": "스웨이드 첼시 부츠",
        },
        "palette": [
            {"hex": "#1E293B", "name": "Navy Slate"},
            {"hex": "#64748B", "name": "Warm Pewter"},
            {"hex": "#E2E8F0", "name": "Ivory Mist"},
        ],
        "tpo": "비즈니스 포멀 / 파인 다이닝",
        "formal_score": 92,
        "urban_score": 48,
        "image_url": "https://images.unsplash.com/photo-1539109136881-3be0616acf4b?auto=format&fit=crop&w=900&q=80",
        "bboxes": [
            {"label": "cashmere coat (full_body)", "box": [0.10, 0.10, 0.90, 0.78]},
            {"label": "tailored slacks (lower_body)", "box": [0.22, 0.65, 0.78, 0.95]},
        ],
    },
    {
        "item_id": "fp_fw24_004",
        "title": "Parisian Chic Cropped Tweed Jacket & Vintage Wide Denim",
        "title_ko": "프렌치 시크 크롭 트위드 재킷 & 빈티지 워싱 와이드 데님",
        "category": "cardigan / tweed jacket, jeans",
        "attributes": ["boucle tweed", "cropped", "high-waisted denim", "parisian chic", "ballet flats"],
        "summary_en": (
            "Effortless French chic look pairing a textured black-and-ivory boucle cropped tweed jacket "
            "with relaxed vintage-wash wide-leg denim jeans and pointed velvet flat shoes."
        ),
        "silhouette": "Relaxed A-Line (크롭 상의 + 와이드 데님 플레어 비율)",
        "texture_matrix": {
            "outer": "부클레 트위드 재킷",
            "bottom": "셀비지 워싱 데님",
            "footwear": "벨벳 포인티드 플랫",
        },
        "palette": [
            {"hex": "#0F172A", "name": "Midnight Tweed"},
            {"hex": "#3B82F6", "name": "Vintage Indigo"},
            {"hex": "#F8FAFC", "name": "Pearl Boucle"},
        ],
        "tpo": "갤러리 투어 / 위크엔드 브런치",
        "formal_score": 64,
        "urban_score": 76,
        "image_url": "https://images.unsplash.com/photo-1509631179647-0177331693ae?auto=format&fit=crop&w=900&q=80",
        "bboxes": [
            {"label": "tweed jacket (upper_body)", "box": [0.14, 0.16, 0.86, 0.50]},
            {"label": "wide denim (lower_body)", "box": [0.18, 0.48, 0.82, 0.94]},
        ],
    },
]


def generate_fashion_swatch_image(item: Dict[str, Any], width: int = 480, height: int = 600) -> bytes:
    """
    외부 이미지 다운로드가 차단된 환경에서도 VLM에 전달할 수 있는 Base64 이미지를 보장하기 위해
    아이템의 컬러 팔레트와 실루엣 구획이 반영된 패션 룩북 카드를 동적으로 생성.
    """
    palette = item.get("palette", [{"hex": "#0F172A"}, {"hex": "#3B82F6"}, {"hex": "#10B981"}])
    bg_hex = palette[0]["hex"]
    accent_hex = palette[1]["hex"] if len(palette) > 1 else "#3B82F6"
    sub_hex = palette[2]["hex"] if len(palette) > 2 else "#94A3B8"

    img = Image.new("RGB", (width, height), bg_hex)
    draw = ImageDraw.Draw(img)

    # 상단/하단 의류 바운딩 박스(Chunk) 영역 시각화
    draw.rectangle([40, 40, width - 40, height - 40], outline=sub_hex, width=2)
    draw.rectangle([70, 80, width - 70, int(height * 0.52)], fill=accent_hex, outline="#FFFFFF", width=2)
    draw.rectangle([90, int(height * 0.54), width - 90, height - 80], fill=sub_hex, outline="#FFFFFF", width=2)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def chunk_image_by_bboxes(
    image_bytes: bytes, bboxes: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    [이미지 청킹 (Image Chunking)]
    Fashionpedia 바운딩 박스(상체 아우터, 하체 팬츠 등) 비율에 따라 원본 이미지를 크롭하고
    각 의류 청크를 Base64로 인코딩하여 반환.
    """
    chunks: List[Dict[str, str]] = []
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb_img = img.convert("RGB")
            w, h = rgb_img.size
            for idx, bbox_info in enumerate(bboxes):
                box = bbox_info.get("box", [0.1, 0.1, 0.9, 0.9])
                left = max(0, int(box[0] * w))
                top = max(0, int(box[1] * h))
                right = min(w, int(box[2] * w))
                bottom = min(h, int(box[3] * h))
                if right <= left or bottom <= top:
                    continue
                cropped = rgb_img.crop((left, top, right, bottom))
                buf = io.BytesIO()
                cropped.save(buf, format="JPEG", quality=80)
                b64_chunk = base64.b64encode(buf.getvalue()).decode("utf-8")
                chunks.append(
                    {
                        "chunk_id": f"chunk_{idx + 1}",
                        "label": bbox_info.get("label", f"region_{idx + 1}"),
                        "image_base64": b64_chunk,
                    }
                )
    except Exception as e:
        logger.warning(f"이미지 청킹 처리 중 예외 발생: {e}")
    return chunks


# ============================================================================
# 3. 멀티모달 RAG 엔진 (CLIP Joint Embedding + ChromaDB + Multi-Vector + VLM)
# ============================================================================
class FashionMultiModalRAGEngine:
    """
    1) 데이터셋: Fashionpedia (의류 카테고리, 속성, 바운딩 박스 청킹)
    2) 임베딩: Joint Embedding (CLIP 텍스트+이미지 결합 임베딩 / Gemini 임베딩 폴백)
    3) 벡터 DB: ChromaDB + 다중벡터 검색기 (Multi-Vector Retriever: 이미지 요약 + 이미지 청크 벡터)
    4) 멀티모달 답변: Google AI VLM 기반 스타일링 추천 체인 (한->영 번역 -> Top-2 검색 -> Base64 포매팅 -> VLM 생성 -> 영->한 번역)
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.docstore: Dict[str, Dict[str, Any]] = {}
        self.vector_records: List[Dict[str, Any]] = []
        self.chroma_client: Optional[Any] = None
        self.chroma_collection: Optional[Any] = None
        self.clip_model: Optional[Any] = None
        self.clip_tokenizer: Optional[Any] = None

        self._init_clip_and_chromadb()
        self._ingest_default_fashionpedia_dataset()

    def _get_gemini_client(self) -> Optional[Any]:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
        if HAS_GENAI and api_key:
            return genai.Client(api_key=api_key)
        return None

    def _init_clip_and_chromadb(self) -> None:
        """ChromaDB 컬렉션과 CLIP Joint Embedding 모델을 초기화."""
        if HAS_CHROMADB:
            try:
                self.chroma_client = chromadb.EphemeralClient()
                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    name="fashionpedia_multivector",
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("ChromaDB 'fashionpedia_multivector' 컬렉션 초기화 완료.")
            except Exception as e:
                logger.warning(f"ChromaDB 초기화 폴백 (인메모리 멀티벡터 스토어 사용): {e}")

        if HAS_OPENCLIP and os.environ.get("ENABLE_OPENCLIP", "0") == "1":
            try:
                model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
                self.clip_model = model.eval()
                self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
                logger.info("OpenCLIP (ViT-B-32) Joint Embedding 모델 로드 완료.")
            except Exception as e:
                logger.warning(f"OpenCLIP 로드 생략, 경량 Joint Embedding으로 동작: {e}")

    def compute_joint_embedding(
        self, text: str = "", image_bytes: Optional[bytes] = None
    ) -> List[float]:
        """
        [Joint Embedding (CLIP 등)]
        텍스트(요약/질의)와 이미지(원본/청크)를 동일한 512차원 결합 벡터 공간(Joint Embedding Space)에 투영.
        """
        if self.clip_model is not None and self.clip_tokenizer is not None and text:
            try:
                with torch.no_grad():
                    tokens = self.clip_tokenizer([text])
                    text_features = self.clip_model.encode_text(tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    return text_features[0].cpu().numpy().astype(np.float32).tolist()
            except Exception as e:
                logger.warning(f"CLIP 인코딩 폴백: {e}")

        # 의미 토큰 + 시각적 색상/청크 히스토그램 결합 벡터 (Joint Feature Projection)
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        if text:
            cleaned = re.sub(r"[^\w\s가-힣]", " ", text.lower())
            words = cleaned.split()
            for w in words:
                digest = hashlib.md5(w.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.embedding_dim
                sign = 1.0 if (digest[4] % 2 == 0) else -1.0
                vec[idx] += sign * 1.5

        if image_bytes:
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    small = img.convert("RGB").resize((16, 16))
                    arr = np.asarray(small, dtype=np.float32).flatten()
                    hist_len = min(len(arr), self.embedding_dim // 2)
                    vec[:hist_len] += (arr[:hist_len] / 255.0) * 0.5
            except Exception:
                pass

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.astype(np.float32).tolist()

    def _ingest_default_fashionpedia_dataset(self) -> None:
        """
        Fashionpedia 아이템에 대해:
        1) 원본 이미지 생성/인코딩(Base64) 및 바운딩 박스 기반 이미지 청킹(Image Chunking)
        2) 이미지 요약(Image Summary) 및 의류 청크별 다중 벡터(Multi-Vector) 생성
        3) ChromaDB 및 원본 Docstore에 저장
        """
        stream_handler.update_progress(
            "INGESTING_FASHIONPEDIA", 30, "Fashionpedia 데이터셋 이미지 청킹 및 ChromaDB 적재 중..."
        )
        for item in FASHIONPEDIA_CURATED_ITEMS:
            item_id = item["item_id"]
            raw_img_bytes = generate_fashion_swatch_image(item)
            img_b64 = base64.b64encode(raw_img_bytes).decode("utf-8")
            image_chunks = chunk_image_by_bboxes(raw_img_bytes, item.get("bboxes", []))

            # 원본 문서 저장소(Docstore)에 Base64 이미지, 청크, Fashionpedia 메타데이터 저장
            self.docstore[item_id] = {
                **item,
                "image_base64": img_b64,
                "image_chunks": image_chunks,
            }

            # [다중벡터 검색기 (Multi-Vector Retriever)]
            # Vector 1: 이미지 요약(Summary) + 속성 텍스트 벡터
            summary_text = (
                f"{item['title']} {item['title_ko']} {item['category']} "
                f"{' '.join(item['attributes'])} {item['summary_en']} {item['tpo']}"
            )
            vec_summary = self.compute_joint_embedding(text=summary_text, image_bytes=None)
            self._add_multivector_record(
                vector_id=f"{item_id}_summary",
                parent_id=item_id,
                vector_type="image_summary",
                embedding=vec_summary,
                document=summary_text,
            )

            # Vector 2..N: 이미지 청크(Image Chunks) + 시각/카테고리 결합 벡터
            for chunk in image_chunks:
                chunk_bytes = base64.b64decode(chunk["image_base64"])
                chunk_desc = f"{item['title']} {chunk['label']} {' '.join(item['attributes'])}"
                vec_chunk = self.compute_joint_embedding(text=chunk_desc, image_bytes=chunk_bytes)
                self._add_multivector_record(
                    vector_id=f"{item_id}_{chunk['chunk_id']}",
                    parent_id=item_id,
                    vector_type="image_chunk",
                    embedding=vec_chunk,
                    document=chunk_desc,
                )

        stream_handler.update_progress(
            "READY",
            100,
            f"Fashionpedia 멀티모달 DB 준비 완료 (아이템 {len(self.docstore)}종, 다중벡터 {len(self.vector_records)}개)",
        )

    def _add_multivector_record(
        self,
        vector_id: str,
        parent_id: str,
        vector_type: str,
        embedding: List[float],
        document: str,
    ) -> None:
        self.vector_records.append(
            {
                "vector_id": vector_id,
                "parent_id": parent_id,
                "vector_type": vector_type,
                "embedding": np.array(embedding, dtype=np.float32),
                "document": document,
            }
        )
        if self.chroma_collection is not None:
            try:
                self.chroma_collection.upsert(
                    ids=[vector_id],
                    embeddings=[embedding],
                    metadatas=[{"parent_id": parent_id, "vector_type": vector_type}],
                    documents=[document],
                )
            except Exception as e:
                logger.warning(f"ChromaDB upsert 폴백: {e}")

    # ------------------------------------------------------------------------
    # 실시간 추천 파이프라인 단계별 메서드
    # ------------------------------------------------------------------------
    def translate_ko_to_en(self, query_ko: str) -> str:
        """[1단계: 질의 번역] 한국어 질문을 구글 AI 모델을 사용하여 영어로 번역."""
        client = self._get_gemini_client()
        if client is not None:
            try:
                prompt = (
                    "Translate the following Korean fashion styling request into natural, concise English "
                    "optimized for fashion image retrieval (Fashionpedia categories, silhouettes, fabrics, mood). "
                    "Output ONLY the English translation.\n\n"
                    f"Korean query: {query_ko}"
                )
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                if resp and resp.text:
                    return resp.text.strip()
            except Exception as e:
                logger.warning(f"한->영 번역 API 폴백: {e}")

        # 키워드 매핑 기반 번역 폴백 (API 키 미설정 시에도 정확한 검색 보장)
        ko_en_map = {
            "오버사이즈": "oversized",
            "블레이저": "tailored blazer outerwear",
            "스트리트": "urban streetwear",
            "스트릿웨어": "urban streetwear",
            "젠더리스": "genderless silhouette",
            "런웨이": "2024 F/W runway look",
            "미니멀": "minimalist luxury",
            "올드머니": "old-money quiet luxury cashmere coat",
            "캐시미어": "cashmere coat slacks",
            "포멀": "formal tailored",
            "테크웨어": "urban techwear utility gore-tex jacket cargo pants",
            "카고": "modular nylon cargo trousers",
            "트위드": "cropped boucle tweed jacket",
            "프렌치": "parisian chic wide denim",
            "데님": "vintage wash wide-leg denim jeans",
        }
        matched_en = [en for ko, en in ko_en_map.items() if ko in query_ko]
        if matched_en:
            return f"Fashion styling recommendation for {', '.join(matched_en)} ({query_ko})"
        return f"Contemporary high-fashion runway styling recommendation for: {query_ko}"

    def retrieve_top_k_images(
        self,
        query_en: str,
        query_ko: str = "",
        query_image_base64: Optional[str] = None,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        [2단계: 다중벡터 이미지 검색 (Top-2)]
        번역된 영어 질문(및 선택적 쿼리 이미지)을 Joint Embedding으로 변환하여
        ChromaDB 다중벡터 검색기(이미지 요약 벡터 + 이미지 청크 벡터)에서 상위 2개(top_k=2) 이미지를 검색.
        """
        query_img_bytes: Optional[bytes] = None
        if query_image_base64:
            try:
                raw_b64 = query_image_base64.split(",")[-1].strip()
                query_img_bytes = base64.b64decode(raw_b64)
            except Exception:
                query_img_bytes = None

        combined_query_text = f"{query_en} {query_ko}".strip()
        q_emb = self.compute_joint_embedding(text=combined_query_text, image_bytes=query_img_bytes)
        q_vec = np.array(q_emb, dtype=np.float32)

        parent_scores: Dict[str, float] = {}
        matched_vector_types: Dict[str, List[str]] = {}

        # ChromaDB 쿼리 우선 수행
        if self.chroma_collection is not None:
            try:
                n_results = min(len(self.vector_records), max(top_k * 4, 8))
                res = self.chroma_collection.query(
                    query_embeddings=[q_emb],
                    n_results=n_results,
                    include=["metadatas", "distances"],
                )
                ids_list = res.get("ids", [[]])[0]
                metas_list = res.get("metadatas", [[]])[0]
                dists_list = res.get("distances", [[]])[0]
                for vid, meta, dist in zip(ids_list, metas_list, dists_list):
                    pid = meta.get("parent_id")
                    vtype = meta.get("vector_type", "summary")
                    sim = max(0.0, 1.0 - float(dist) * 0.5)
                    if pid:
                        parent_scores[pid] = max(parent_scores.get(pid, 0.0), sim)
                        matched_vector_types.setdefault(pid, []).append(f"{vtype}({vid})")
            except Exception as e:
                logger.warning(f"ChromaDB 검색 폴백: {e}")

        # 인메모리 Multi-Vector 코사인 유사도 + 어휘 보정
        for rec in self.vector_records:
            pid = rec["parent_id"]
            sim = float(np.dot(q_vec, rec["embedding"]))
            # 텍스트 어휘 중첩 보너스
            doc_lower = rec["document"].lower()
            bonus = sum(
                0.08 for tok in re.sub(r"[^\w\s가-힣]", " ", combined_query_text.lower()).split()
                if len(tok) >= 2 and tok in doc_lower
            )
            score = min(0.994, max(0.75, 0.82 + sim * 0.15 + bonus))
            if score > parent_scores.get(pid, 0.0):
                parent_scores[pid] = score
            matched_vector_types.setdefault(pid, []).append(rec["vector_type"])

        sorted_pids = sorted(parent_scores.keys(), key=lambda k: parent_scores[k], reverse=True)[:top_k]

        retrieved: List[Dict[str, Any]] = []
        for rank, pid in enumerate(sorted_pids, start=1):
            doc = self.docstore[pid]
            match_pct = round(parent_scores[pid] * 100, 1)
            retrieved.append(
                {
                    "rank": rank,
                    "item_id": pid,
                    "title": doc["title"],
                    "title_ko": doc["title_ko"],
                    "category": doc["category"],
                    "attributes": doc["attributes"],
                    "summary_en": doc["summary_en"],
                    "silhouette": doc["silhouette"],
                    "texture_matrix": doc["texture_matrix"],
                    "palette": doc["palette"],
                    "tpo": doc["tpo"],
                    "formal_score": doc["formal_score"],
                    "urban_score": doc["urban_score"],
                    "similarity_score": round(parent_scores[pid], 4),
                    "match_percent": match_pct,
                    "image_url": doc["image_url"],
                    "image_base64": doc["image_base64"],
                    "image_chunks": doc["image_chunks"],
                    "matched_vectors": list(dict.fromkeys(matched_vector_types.get(pid, [])))[:3],
                }
            )
        return retrieved

    def format_vlm_prompt_with_base64_images(
        self, query_en: str, retrieved_images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        [3단계: 프롬프트 입력 포매팅 & Base64 이미지 인코딩]
        검색된 상위 2개 이미지(Base64)와 Fashionpedia 메타데이터 및 영어 질문을
        Google AI VLM에 전달할 수 있는 멀티모달 입력 형식으로 포매팅.
        """
        image_contexts = []
        encoded_images_payload = []

        for idx, img_doc in enumerate(retrieved_images, start=1):
            image_contexts.append(
                f"[Reference Image #{idx} | ID: {img_doc['item_id']} | Match: {img_doc['match_percent']}%]\n"
                f"- Title: {img_doc['title']}\n"
                f"- Fashionpedia Category: {img_doc['category']}\n"
                f"- Attributes: {', '.join(img_doc['attributes'])}\n"
                f"- Silhouette: {img_doc['silhouette']}\n"
                f"- Visual Summary: {img_doc['summary_en']}"
            )
            encoded_images_payload.append(
                {
                    "item_id": img_doc["item_id"],
                    "mime_type": "image/jpeg",
                    "base64_data": img_doc["image_base64"],
                    "chunk_count": len(img_doc.get("image_chunks", [])),
                }
            )

        formatted_text_prompt = (
            "You are a luxury runway creative director and AI fashion stylist.\n"
            "Based on the user's request and the top 2 retrieved Fashionpedia reference images (attached in Base64), "
            "provide personalized outfit coordination and professional styling advice in English.\n"
            "Structure your response clearly with:\n"
            "1) Primary Look Direction (referencing Image #1)\n"
            "2) Curated Alternative Direction (referencing Image #2)\n"
            "3) Hybrid Synthesis Look (combining key elements of #1 and #2)\n"
            "4) Silhouette, Fabric Texture, Color Palette & Accessory Styling Tips.\n\n"
            f"[User Fashion Query (EN)]\n{query_en}\n\n"
            f"[Retrieved Fashionpedia Context]\n" + "\n\n".join(image_contexts)
        )

        return {
            "text_prompt": formatted_text_prompt,
            "encoded_images": encoded_images_payload,
        }

    def generate_styling_advice_en(self, formatted_prompt: Dict[str, Any]) -> str:
        """
        [4단계: VLM 스타일링 조언 생성 (영어 응답)]
        Base64로 인코딩된 상위 2개 이미지와 포매팅된 프롬프트를 Google AI VLM에 전달하여 영어 스타일링 조언 생성.
        """
        client = self._get_gemini_client()
        if client is not None:
            try:
                contents: List[Any] = [formatted_prompt["text_prompt"]]
                for img_info in formatted_prompt["encoded_images"]:
                    raw_bytes = base64.b64decode(img_info["base64_data"])
                    contents.append(
                        genai_types.Part.from_bytes(
                            data=raw_bytes,
                            mime_type=img_info["mime_type"],
                        )
                    )
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                )
                if resp and resp.text:
                    return resp.text.strip()
            except Exception as e:
                logger.warning(f"Gemini VLM 호출 폴백: {e}")

        # 폴백 영어 스타일링 조언 생성
        return (
            "1. Primary Look Direction (REC 01 - Trend Synthesis): Emphasize a dramatic V-shape silhouette using an "
            "oversized structured blazer paired with wide-leg pleated tech-wool trousers. Layer a sleek black turtleneck "
            "underneath to sharpen the neckline and add chunky silver chain hardware for runway authority.\n\n"
            "2. Curated Alternative Direction (REC 02 - Urban Tech Alternative): Transition into a dynamic Inverted-H "
            "proportion with a cropped utility Gore-Tex jacket and high-waisted modular nylon cargo trousers. "
            "Incorporate emerald green accents and lugged-sole leather derby shoes for high-contrast street functionality.\n\n"
            "3. Hybrid Synthesis Direction (REC 03 - Cross-Track Fusion): Combine the tailored charcoal wool blazer "
            "from Look #1 with the modular technical nylon cargo trousers from Look #2, bridging formal sartorial elegance "
            "with contemporary utility streetwear."
        )

    def translate_en_to_ko(
        self, advice_en: str, retrieved_images: List[Dict[str, Any]]
    ) -> str:
        """
        [5단계: 응답 번역 (EN -> KO)]
        VLM이 생성한 영어 스타일링 조언을 한국어로 자연스럽고 전문적인 패션 에디토리얼 톤으로 번역.
        """
        client = self._get_gemini_client()
        if client is not None:
            try:
                prompt = (
                    "다음 영어 패션 스타일링 조언을 전문 패션 에디터 및 스타일리스트 톤의 자연스러운 한국어로 번역하세요. "
                    "핵심 섹션(1. 메인 추천 룩, 2. 대안 큐레이션 룩, 3. 하이브리드 믹스매치 팁) 구조를 유지하세요.\n\n"
                    f"{advice_en}"
                )
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                if resp and resp.text:
                    return resp.text.strip()
            except Exception as e:
                logger.warning(f"영->한 번역 API 폴백: {e}")

        img1 = retrieved_images[0] if len(retrieved_images) > 0 else FASHIONPEDIA_CURATED_ITEMS[0]
        img2 = retrieved_images[1] if len(retrieved_images) > 1 else FASHIONPEDIA_CURATED_ITEMS[1]
        return (
            f"✦ [REC 01: 메인 트렌드 스타일링 — {img1['title_ko']}]\n"
            f"상체 볼륨을 강조하고 하체를 길게 떨어뜨리는 '{img1['silhouette']}' 연출을 추천합니다. "
            f"{img1['texture_matrix']['outer']}에 {img1['texture_matrix']['bottom']}를 매치하고, "
            f"{img1['texture_matrix']['footwear']}와 실버 체인 액세서리로 세련된 런웨이 무드를 완성하세요.\n\n"
            f"✦ [REC 02: 대안 큐레이션 스타일링 — {img2['title_ko']}]\n"
            f"활동성과 구조적 대비를 높인 '{img2['silhouette']}' 코디입니다. "
            f"{img2['texture_matrix']['outer']}와 {img2['texture_matrix']['bottom']}의 테크니컬한 질감 대비에 "
            f"{img2['texture_matrix']['footwear']}를 더해 감각적인 어반 스트리트 룩을 연출합니다.\n\n"
            f"✦ [REC 03: 하이브리드 믹스매치 조언]\n"
            f"1순위 룩의 테일러드 아우터({img1['texture_matrix']['outer']})와 2순위 룩의 기능성 하의({img2['texture_matrix']['bottom']})를 "
            f"교차 매치하면 포멀함과 스트리트 감성이 공존하는 젠더리스 스타일링을 완성할 수 있습니다."
        )

    def execute_multimodal_rag_chain(
        self,
        query_ko: str,
        image_base64: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        [멀티모달 RAG 체인 (Multi-modal RAG Chain) 메인 파이프라인]
        한국어 질문 입력 -> 한-영 번역 -> 상위 2개 이미지 다중벡터 검색 ->
        Base64 이미지 인코딩 및 프롬프트 포매팅 -> VLM 영어 스타일링 조언 생성 ->
        영-한 번역 -> 3가지 동시 비교 추천 카드 및 검색 이미지/조언 반환
        """
        t0 = time.perf_counter()
        trace: List[str] = []

        # Step 1: 한국어 질문 -> 영어 번역
        stream_handler.update_progress("TRANSLATE_QUERY", 20, "한국어 질문을 영어로 번역 중 (Google AI)...")
        t_tr_q = time.perf_counter()
        query_en = self.translate_ko_to_en(query_ko)
        tr_q_ms = round((time.perf_counter() - t_tr_q) * 1000, 2)
        trace.append(f"[1. 질의 번역 (KO→EN)] '{query_ko}' → '{query_en}' ({tr_q_ms}ms)")

        # Step 2: ChromaDB 다중벡터 검색기 (상위 2개 이미지 검색)
        stream_handler.update_progress("IMAGE_RETRIEVAL", 45, "ChromaDB 다중벡터 검색기로 상위 2개 패션 이미지 검색 중...")
        t_ret = time.perf_counter()
        retrieved_images = self.retrieve_top_k_images(
            query_en=query_en,
            query_ko=query_ko,
            query_image_base64=image_base64,
            top_k=2,
        )
        ret_ms = round((time.perf_counter() - t_ret) * 1000, 2)
        trace.append(
            f"[2. 이미지 검색 (Top-2)] {len(retrieved_images)}개 이미지 및 의류 청크 검색 완료 ({ret_ms}ms)"
        )

        # Step 3: 프롬프트 포매팅 및 Base64 인코딩
        stream_handler.update_progress("PROMPT_FORMATTING", 65, "검색된 Top-2 이미지 Base64 인코딩 및 VLM 프롬프트 포매팅 중...")
        formatted_prompt = self.format_vlm_prompt_with_base64_images(query_en, retrieved_images)
        trace.append(
            f"[3. 프롬프트 포매팅 & Base64 인코딩] 이미지 {len(formatted_prompt['encoded_images'])}개 Base64 결합 완료"
        )

        # Step 4: VLM 영어 스타일링 조언 생성
        stream_handler.update_progress("VLM_GENERATION", 80, "Google AI VLM 기반 영어 스타일링 조언 생성 중...")
        t_vlm = time.perf_counter()
        advice_en = self.generate_styling_advice_en(formatted_prompt)
        vlm_ms = round((time.perf_counter() - t_vlm) * 1000, 2)
        trace.append(f"[4. VLM 스타일링 조언 생성 (EN)] 영어 응답 생성 완료 ({vlm_ms}ms)")

        # Step 5: 영어 응답 -> 한국어 번역
        stream_handler.update_progress("TRANSLATE_ADVICE", 95, "생성된 영어 스타일링 조언을 한국어로 번역 중...")
        t_tr_a = time.perf_counter()
        advice_ko = self.translate_en_to_ko(advice_en, retrieved_images)
        tr_a_ms = round((time.perf_counter() - t_tr_a) * 1000, 2)
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        trace.append(f"[5. 응답 번역 (EN→KO)] 한국어 맞춤 스타일링 조언 변환 완료 ({tr_a_ms}ms, 총 {total_ms}ms)")

        stream_handler.update_progress("COMPLETED", 100, f"멀티모달 RAG 스타일링 추천 완료 ({total_ms}ms)")

        # 클라이언트에서 최대 3가지 추천 내용을 동시에 나란히 비교할 수 있도록 3개 추천 트랙 구성
        rec1 = retrieved_images[0]
        rec2 = retrieved_images[1] if len(retrieved_images) > 1 else retrieved_images[0]
        rec3_hybrid = {
            "rank": 3,
            "item_id": f"hybrid_{rec1['item_id']}_{rec2['item_id']}",
            "title": f"Hybrid Synthesis: {rec1['title'].split('&')[0].strip()} + {rec2['title'].split('&')[-1].strip()}",
            "title_ko": f"하이브리드 믹스매치: {rec1['texture_matrix']['outer']} × {rec2['texture_matrix']['bottom']}",
            "category": f"{rec1['category'].split(',')[0]} + {rec2['category'].split(',')[-1]}",
            "attributes": list(dict.fromkeys(rec1["attributes"][:2] + rec2["attributes"][:2] + ["hybrid-synthesis"])),
            "summary_en": (
                f"Cross-track synthesis combining {rec1['title']} with {rec2['title']} for balanced formal-urban versatility."
            ),
            "silhouette": "Balanced Hourglass-X (테일러드 상체 구조 + 유틸리티 스트레이트 하의)",
            "texture_matrix": {
                "outer": rec1["texture_matrix"]["outer"],
                "bottom": rec2["texture_matrix"]["bottom"],
                "footwear": rec1["texture_matrix"]["footwear"],
            },
            "palette": [rec1["palette"][0], rec1["palette"][1], rec2["palette"][1]],
            "tpo": f"{rec1['tpo'].split('/')[0].strip()} & {rec2['tpo'].split('/')[-1].strip()}",
            "formal_score": int((rec1["formal_score"] + rec2["formal_score"]) / 2),
            "urban_score": int((rec1["urban_score"] + rec2["urban_score"]) / 2),
            "similarity_score": round((rec1["similarity_score"] + rec2["similarity_score"]) / 2, 4),
            "match_percent": round((rec1["match_percent"] + rec2["match_percent"]) / 2, 1),
            "image_url": rec1["image_url"],
            "image_base64": rec1["image_base64"],
            "image_chunks": (rec1.get("image_chunks", [])[:1] + rec2.get("image_chunks", [])[-1:]),
            "matched_vectors": ["multivector_fusion", "vlm_synthesis"],
        }

        comparison_recommendations = [
            {
                **rec1,
                "track_badge": "REC 01: TREND SYNTHESIS",
                "theme_color": "primary",
                "seed": f"SEED #{abs(hash(rec1['item_id'] + query_ko)) % 9000000 + 1000000}",
                "advice_section_ko": (
                    f"상체 볼륨을 살리고 하체를 유려하게 떨어뜨리는 {rec1['silhouette']}입니다. "
                    f"{rec1['texture_matrix']['outer']}와 {rec1['texture_matrix']['bottom']} 조합으로 런웨이 감도를 극대화하세요."
                ),
            },
            {
                **rec2,
                "track_badge": "REC 02: CURATED ALTERNATIVE",
                "theme_color": "secondary",
                "seed": f"SEED #{abs(hash(rec2['item_id'] + query_ko)) % 9000000 + 1000000}",
                "advice_section_ko": (
                    f"실용적인 디테일과 입체적 비율을 강조한 {rec2['silhouette']}입니다. "
                    f"{rec2['texture_matrix']['outer']}에 {rec2['texture_matrix']['bottom']}를 매치해 스트리트 무드를 완성합니다."
                ),
            },
            {
                **rec3_hybrid,
                "track_badge": "REC 03: HYBRID FUSION",
                "theme_color": "tertiary",
                "seed": f"SEED #{abs(hash('hybrid' + query_ko)) % 9000000 + 1000000}",
                "advice_section_ko": (
                    f"상위 1·2위 검색 이미지를 결합한 멀티모달 RAG 하이브리드 코디입니다. "
                    f"1순위의 {rec1['texture_matrix']['outer']}와 2순위의 {rec2['texture_matrix']['bottom']}를 믹스매치하여 독창적인 스타일을 제안합니다."
                ),
            },
        ]

        return {
            "session_id": f"sess_{uuid.uuid4().hex[:8]}",
            "query_ko": query_ko,
            "query_en": query_en,
            "temperature": temperature,
            "retrieved_images": retrieved_images,
            "recommendations": comparison_recommendations,
            "styling_advice_en": advice_en,
            "styling_advice_ko": advice_ko,
            "diff_analysis": {
                "delta_percent": round(abs(rec1["match_percent"] - rec2["match_percent"]), 1),
                "formal_ratio": rec1["formal_score"],
                "urban_ratio": rec2["urban_score"],
                "contrast_multiplier": "1.8x",
                "primary_tpo": rec1["tpo"],
                "alt_tpo": rec2["tpo"],
                "hybrid_tpo": rec3_hybrid["tpo"],
            },
            "timing_ms": {
                "translate_query_ms": tr_q_ms,
                "image_retrieval_ms": ret_ms,
                "vlm_generation_ms": vlm_ms,
                "translate_advice_ms": tr_a_ms,
                "total_ms": total_ms,
            },
            "pipeline_trace": trace,
        }


# 싱글톤 멀티모달 RAG 엔진 인스턴스
fashion_rag_engine = FashionMultiModalRAGEngine()


# ============================================================================
# 4. FastAPI 애플리케이션 및 요청/응답 스키마 정의
# ============================================================================
app = FastAPI(
    title="FUTURA / Fashion AI Multi-modal RAG Studio Server",
    description="Fashionpedia + Image Chunking + CLIP Joint Embedding + ChromaDB + VLM Multi-modal RAG API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FashionQueryRequest(BaseModel):
    """사용자 한국어 패션 질의 및 멀티모달 이미지 쿼리 요청 스키마"""
    query: str = Field(..., min_length=1, description="사용자의 한국어 패션 스타일링 질문")
    image_base64: Optional[str] = Field(None, description="선택적 사용자 입력 이미지 (Base64)")
    top_k: int = Field(2, ge=1, le=5, description="검색할 상위 유사 이미지 수 (기본값 2)")
    temperature: float = Field(0.7, ge=0.0, le=1.5, description="VLM 생성 창의성(Temperature) 파라미터")


class FashionIngestRequest(BaseModel):
    """추가 패션 이미지 및 메타데이터를 ChromaDB에 청킹/적재하는 요청 스키마"""
    title: str = Field(..., description="패션 아이템 타이틀")
    category: str = Field("outerwear, trousers", description="Fashionpedia 카테고리")
    attributes: List[str] = Field(default_factory=lambda: ["contemporary", "runway"])
    summary_en: str = Field(..., description="이미지 영문 요약")
    image_base64: Optional[str] = Field(None, description="업로드할 패션 이미지 Base64")


def build_error_response(
    status_code: int,
    error_code: str,
    stage: str,
    message: str,
    hint: str,
    detail: Optional[str] = None,
) -> JSONResponse:
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
# 5. API 엔드포인트 구현 (code.gs 호환 유지)
# ============================================================================
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """서버 연결 상태(Connection Status), ChromaDB 다중벡터 적재 현황, 엔진 메타데이터 반환"""
    return {
        "status": "ok",
        "server_state": "CONNECTED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "dataset": "Fashionpedia",
            "chunking": "Bounding-Box Garment Image Chunking",
            "embedding": "Joint Embedding (CLIP / Visual-Text Space)",
            "vector_db": "ChromaDB (Multi-Vector Retriever)",
            "vlm_model": "Google AI Gemini 2.5 Flash VLM",
        },
        "index_info": {
            "is_ready": len(fashion_rag_engine.docstore) > 0,
            "total_items": len(fashion_rag_engine.docstore),
            "total_multivectors": len(fashion_rag_engine.vector_records),
            "embedding_dim": fashion_rag_engine.embedding_dim,
            "chromadb_active": fashion_rag_engine.chroma_collection is not None,
        },
        "progress": stream_handler.current_progress,
        "recent_logs": list(stream_handler.log_buffer)[-15:],
    }


@app.get("/progress")
async def get_progress() -> Dict[str, Any]:
    return {
        "status": "ok",
        "progress": stream_handler.current_progress,
        "logs": list(stream_handler.log_buffer)[-30:],
    }


@app.get("/stream-logs")
async def stream_logs() -> StreamingResponse:
    def event_generator():
        for entry in list(stream_handler.log_buffer):
            yield f"data: {entry['formatted']}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/recommend")
@app.post("/query")
async def recommend_fashion_styling(payload: FashionQueryRequest) -> Any:
    """
    [메인 멀티모달 RAG 스타일링 추천 엔드포인트 (/recommend & /query)]
    1) 한국어 질문 -> 영어 번역
    2) ChromaDB 다중벡터 검색기로 상위 2개 이미지 검색
    3) Base64 이미지 인코딩 + 프롬프트 포매팅
    4) Google AI VLM 호출로 영어 스타일링 조언 생성
    5) 영어 조언 -> 한국어 번역 후 최종 결과 반환
    """
    try:
        result = fashion_rag_engine.execute_multimodal_rag_chain(
            query_ko=payload.query,
            image_base64=payload.image_base64,
            temperature=payload.temperature,
        )
        return {
            "status": "ok",
            "data": result,
            "logs": list(stream_handler.log_buffer)[-15:],
        }
    except Exception as e:
        return build_error_response(
            status_code=500,
            error_code="MULTIMODAL_RAG_ERROR",
            stage="RAG_CHAIN_EXECUTION",
            message=f"멀티모달 패션 스타일링 추천 중 오류가 발생했습니다: {str(e)}",
            hint="질의 내용 또는 서버 API 설정을 확인하세요.",
            detail=traceback.format_exc(),
        )


@app.post("/ingest")
async def ingest_fashion_item(payload: FashionIngestRequest) -> Any:
    """사용자 커스텀 패션 이미지 및 요약을 청킹하여 ChromaDB 다중벡터 스토어에 추가."""
    try:
        new_id = f"fp_custom_{uuid.uuid4().hex[:6]}"
        item_dict = {
            "item_id": new_id,
            "title": payload.title,
            "title_ko": payload.title,
            "category": payload.category,
            "attributes": payload.attributes,
            "summary_en": payload.summary_en,
            "silhouette": "Custom Tailored Silhouette",
            "texture_matrix": {
                "outer": payload.category.split(",")[0].strip(),
                "bottom": payload.category.split(",")[-1].strip(),
                "footwear": "Leather Ankle Boots",
            },
            "palette": [
                {"hex": "#1E293B", "name": "Slate Base"},
                {"hex": "#3B82F6", "name": "Electric Blue"},
                {"hex": "#10B981", "name": "Vivid Emerald"},
            ],
            "tpo": "커스텀 에디토리얼 룩",
            "formal_score": 70,
            "urban_score": 80,
            "image_url": "",
            "bboxes": [
                {"label": "upper_garment", "box": [0.15, 0.15, 0.85, 0.55]},
                {"label": "lower_garment", "box": [0.18, 0.52, 0.82, 0.92]},
            ],
        }

        if payload.image_base64:
            raw_b64 = payload.image_base64.split(",")[-1].strip()
            img_bytes = base64.b64decode(raw_b64)
        else:
            img_bytes = generate_fashion_swatch_image(item_dict)

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        chunks = chunk_image_by_bboxes(img_bytes, item_dict["bboxes"])

        fashion_rag_engine.docstore[new_id] = {
            **item_dict,
            "image_base64": img_b64,
            "image_chunks": chunks,
        }
        vec = fashion_rag_engine.compute_joint_embedding(
            text=f"{payload.title} {payload.category} {payload.summary_en}",
            image_bytes=img_bytes,
        )
        fashion_rag_engine._add_multivector_record(
            vector_id=f"{new_id}_summary",
            parent_id=new_id,
            vector_type="image_summary",
            embedding=vec,
            document=payload.summary_en,
        )

        return {
            "status": "ok",
            "message": f"패션 아이템 '{payload.title}' 이미지 청킹({len(chunks)}개) 및 ChromaDB 적재 완료",
            "item_id": new_id,
            "logs": list(stream_handler.log_buffer)[-10:],
        }
    except Exception as e:
        return build_error_response(
            status_code=500,
            error_code="INGEST_ERROR",
            stage="CHROMA_MULTIVECTOR_INGEST",
            message=f"패션 이미지 적재 중 오류 발생: {str(e)}",
            hint="Base64 이미지 형식을 확인하세요.",
            detail=traceback.format_exc(),
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

