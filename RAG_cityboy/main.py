import os
import io
import json
import logging
import asyncio
from typing import List, AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# PDF 및 텍스트 분할 관련
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 임베딩, 벡터DB, LLM 관련
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain LCEL & Prompt 관련
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import AsyncCallbackHandler

# 로깅 설정 (Cloud Run 수집용 상세 로깅)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RAG_Server")

app = FastAPI(title="City Policy RAG Server", version="1.0.0")

# CORS 허용 (Google Apps Script 통신용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. 환경 변수 검증 및 초기화
# ---------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not all([GEMINI_API_KEY, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    logger.critical(" 필수 환경변수가 설정되지 않았습니다. GEMINI_API_KEY, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY를 확인하세요.")

# Supabase 클라이언트 및 임베딩 초기화
try:
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    vector_store = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
except Exception as e:
    logger.error(f" Supabase/Embedding 초기화 오류: {str(e)}")

# ---------------------------------------------------------
# 2. 커스텀 스트리밍 콜백 핸들러
# ---------------------------------------------------------
class StreamHandler(AsyncCallbackHandler):
    """LLM 스트리밍 토큰을 캡처하여 비동기 큐로 전송하는 핸들러"""
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            await self.queue.put(token)

    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        logger.error(f" LLM 스트리밍 중 오류 발생: {str(error)}")
        await self.queue.put(f"\n[오류 발생: {str(error)}]\n")

# ---------------------------------------------------------
# 3. 데이터 모델 정의
# ---------------------------------------------------------
class ChatMessage(BaseModel):
    role: str # "user" 또는 "assistant"
    content: str

class QueryRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage] = []

# ---------------------------------------------------------
# 4. API 엔드포인트
# ---------------------------------------------------------

@app.get("/health")
async def health_check():
    """서버 통신 상태 확인용 엔드포인트"""
    return {"status": "ok", "message": "Google Cloud Run RAG Server is running properly."}


@app.post("/api/extract-text")
async def extract_pdf_text(file: UploadFile = File(...)):
    """PDF 파일 선택 시 텍스트를 추출하여 미리보기용으로 반환하는 엔드포인트"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="INVALID_FILE_TYPE: PDF 파일만 업로드할 수 있습니다.")
    
    try:
        pdf_bytes = await file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        extracted_text = ""
        
        for idx, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                extracted_text += f"--- Page {idx + 1} ---\n{text}\n\n"
        
        if not extracted_text.strip():
            raise HTTPException(status_code=422, detail="PDF_PARSE_EMPTY: PDF에서 텍스트를 추출할 수 없습니다. (스캔본 이미지 여부 확인 필요)")

        return {"filename": file.filename, "text": extracted_text.strip()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF 텍스트 추출 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF_PROCESSING_ERROR: PDF를 읽는 도중 서버 오류가 발생했습니다 ({str(e)})")


@app.post("/api/build-rag")
async def build_rag_index(file: UploadFile = File(...)):
    """PDF 텍스트 청킹 후 Supabase pgvector에 임베딩 저장"""
    try:
        pdf_bytes = await file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        
        documents_to_insert = []
        
        # 1. 텍스트 분할기 설정 (청크 크기 800, 오버랩 100)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        
        for idx, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                chunks = text_splitter.split_text(page_text)
                for chunk in chunks:
                    documents_to_insert.append({
                        "page_content": chunk,
                        "metadata": {"source": file.filename, "page": idx + 1}
                    })

        if not documents_to_insert:
            raise HTTPException(status_code=400, detail="BUILD_RAG_EMPTY: RAG를 생성할 유효한 텍스트가 없습니다.")

        # 2. Supabase Vector store에 청크 및 OpenAI 임베딩 등록
        texts = [doc["page_content"] for doc in documents_to_insert]
        metadatas = [doc["metadata"] for doc in documents_to_insert]
        
        # 이전 동일 파일 데이터 삭제 처리 (옵션)
        supabase_client.table("documents").delete().eq("metadata->>source", file.filename).execute()
        
        vector_store.add_texts(texts=texts, metadatas=metadatas)
        logger.info(f" RAG 구축 완료: {file.filename} ({len(texts)} 개 청크 생성 및 저장됨)")
        
        return {
            "status": "success", 
            "message": f"성공적으로 '{file.filename}' 기반 Vector DB가 생성되었습니다.",
            "total_chunks": len(texts)
        }

    except Exception as e:
        logger.error(f"RAG Vector DB 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"VECTOR_DB_BUILD_FAILED: RAG 벡터 생성 실패 ({str(e)})")


@app.post("/api/chat")
async def chat_rag_stream(request: QueryRequest):
    """
    질의 확장(Query Augmentation) 및 검색, Gemini 2.5 Flash 실시간 스트리밍 답변
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # -----------------------------------------------------
            # 1. 대화 기록 (Chat History) 파싱
            # -----------------------------------------------------
            formatted_history = []
            for msg in request.chat_history:
                if msg.role == "user":
                    formatted_history.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    formatted_history.append(AIMessage(content=msg.content))

            # -----------------------------------------------------
            # 2. Query Augmentation Chain (질의 확장 구현)
            # -----------------------------------------------------
            query_aug_prompt = ChatPromptTemplate.from_messages([
                ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])

            # 증강 체인용 고속 Gemini 모델
            aug_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GEMINI_API_KEY,
                temperature=0.1
            )
            
            query_aug_chain = query_aug_prompt | aug_llm | StrOutputParser()
            
            # 4. 체인 실행 (데이터 주입)
            augmented_query = await query_aug_chain.ainvoke({
                "chat_history": formatted_history,
                "question": request.query
            })
            logger.info(f" 원본 질의: '{request.query}' -> 증강된 질의: '{augmented_query}'")

            # -----------------------------------------------------
            # 5. 증강된 질의로 벡터 DB 검색 (Retriever)
            # -----------------------------------------------------
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            retrieved_docs = await retriever.ainvoke(augmented_query)

            # 클라이언트에 검색된 청크/출처 정보 먼저 전달 (SSE format)
            sources_payload = []
            context_str = ""
            for idx, doc in enumerate(retrieved_docs):
                source_info = f"출처: {doc.metadata.get('source', '알수없음')} (p.{doc.metadata.get('page', '?')})"
                sources_payload.append({
                    "content": doc.page_content,
                    "source": source_info
                })
                context_str += f"[문서 {idx+1}] ({source_info})\n{doc.page_content}\n\n"

            # SSE JSON 프로토콜 전송: 검색 결과 이벤트
            yield f"event: sources\ndata: {json.dumps({'augmented_query': augmented_query, 'sources': sources_payload}, ensure_ascii=False)}\n\n"

            # -----------------------------------------------------
            # 6. 도시 정책 전문가 프롬프트 & StreamHandler 적용 답변 생성
            # -----------------------------------------------------
            stream_queue = asyncio.Queue()
            stream_handler = StreamHandler(stream_queue)

            chat_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GEMINI_API_KEY,
                temperature=0.3,
                streaming=True,
                callbacks=[stream_handler]
            )

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야. 주어진 [참고 문서] 내용만을 바탕으로 사용자의 질문에 전문적이고 명확하게 답변해라. 문서에 없는 내용은 모른다고 솔직히 답해라."),
                ("human", " [참고 문서]\n{context}\n\n질문: {question}")
            ])

            qa_chain = qa_prompt | chat_llm | StrOutputParser()

            # 백그라운드 태스크로 체인 실행
            task = asyncio.create_task(qa_chain.ainvoke({
                "context": context_str,
                "question": augmented_query
            }))

            # Queue에서 토큰을 꺼내 클라이언트로 스트리밍 출력
            while not task.done() or not stream_queue.empty():
                try:
                    token = await asyncio.wait_for(stream_queue.get(), timeout=0.1)
                    yield f"event: token\ndata: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                    stream_queue.task_done()
                except asyncio.TimeoutError:
                    continue

            # 태스크 예외 확인
            if task.exception():
                raise task.exception()

            yield f"event: end\ndata: {json.dumps({'status': 'completed'})}\n\n"

        except Exception as e:
            logger.error(f" RAG 스트리밍 처리 실패: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'error': f'RAG_STREAM_ERROR: {str(e)}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
