try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="Seoul Urban Planning RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 상태 관리
vectorstore = None
rag_chain = None

RAG_CONFIG = {
    "vector_db": "ChromaDB (In-Memory/Ephemeral)",
    "embedding_model": "models/text-embedding-004",
    "llm_model": "gemini-2.5-flash",
    "retriever": "MultiQueryRetriever (다중 질문 생성 검색기)",
    "query_expansion": "RunnablePassthrough.assign (질의 확장 및 컨텍스트 병합)",
    "system_role": "당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요"
}

class ChatRequest(BaseModel):
    question: str

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@app.get("/status")
async def get_status():
    return {
        "status": "online",
        "pdf_loaded": rag_chain is not None,
        "config": RAG_CONFIG
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, rag_chain
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 1. PDF 파일 읽어오기 및 분할
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 2. 구글 임베딩 모델 및 Chroma DB 구성
        embeddings = GoogleGenerativeAIEmbeddings(model=RAG_CONFIG["embedding_model"])
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            collection_name="seoul_urban_plan"
        )

        # 3. LLM 및 다중 질문 검색기(MultiQueryRetriever) 설정
        llm = ChatGoogleGenerativeAI(model=RAG_CONFIG["llm_model"], temperature=0.2)
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            llm=llm
        )

        # 4. 질의 확장 (RunnablePassthrough 사용)
        expansion_prompt = ChatPromptTemplate.from_template(
            "다음 질문을 서울시 도시계획, 토지이용, 지구단위계획 및 건축 법규 관점에서 "
            "더 구체적이고 전문적인 검색 질의로 확장하세요. 오직 확장된 질의 문장만 출력하세요:\n"
            "질문: {question}"
        )
        query_expansion_chain = expansion_prompt | llm | StrOutputParser()

        # 5. 시스템 역할 정의 및 프롬프트 구성
        system_prompt = (
            f"{RAG_CONFIG['system_role']}\n\n"
            "아래 제공된 [컨텍스트]를 바탕으로 답변하세요. "
            "컨텍스트에 없는 내용은 추측하지 말고 모른다고 답변하세요.\n\n"
            "[확장된 질의]: {expanded_query}\n"
            "[컨텍스트]:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        # 6. RAG 체인 구성 (RunnablePassthrough를 활용한 질의 확장 및 검색 연결)
        rag_chain = (
            RunnablePassthrough.assign(expanded_query=query_expansion_chain)
            | RunnablePassthrough.assign(
                context=lambda x: format_docs(multi_query_retriever.invoke(x["expanded_query"]))
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        os.remove(tmp_path)
        return {
            "status": "success", 
            "message": f"'{file.filename}' 처리 완료 ({len(splits)}개 청크 생성)",
            "config": RAG_CONFIG
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    global rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=400, detail="먼저 PDF 파일을 업로드하여 학습시켜 주세요.")
    
    try:
        answer = rag_chain.invoke({"question": request.question})
        return {
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
