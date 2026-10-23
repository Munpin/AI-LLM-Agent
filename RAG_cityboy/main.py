import os
import io
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# PyPDF module for PDF text extraction
import pypdf

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Supabase Client
from supabase.client import Client, create_client

# Configure logging format and level for Cloud Run stdout parsing
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("rag_server")

# Initialize FastAPI Application
app = FastAPI(
    title="City Policy AI Assistant API",
    description="RAG Engine with Query Augmentation powered by Google Gemini and Supabase",
    version="1.0.1"
)

# CORS middleware enabling interaction from frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Environment Variables & Clients Verification
# ---------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning("Supabase credentials (SUPABASE_URL/SUPABASE_KEY) are missing in environment variables.")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is missing. OpenAI Embedding functionality will fail if invoked.")

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY is missing. Gemini LLM calls will fail if invoked.")

# Global Supabase client
supabase_client: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Successfully initialized Supabase Client.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase Client: {str(e)}")

# Initialize OpenAI Embeddings for Vector Search
def get_embeddings() -> OpenAIEmbeddings:
    """Returns OpenAI Embeddings instance."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server Misconfiguration: OPENAI_API_KEY environment variable is missing."
        )
    return OpenAIEmbeddings(model="text-embedding-3-small")


# ---------------------------------------------------------
# Data Transfer Objects (DTOs)
# ---------------------------------------------------------
class ChatHistoryItem(BaseModel):
    role: str = Field(..., description="Role of the speaker: 'user' or 'assistant'")
    content: str = Field(..., description="Content of the message")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Current question asked by the user")
    chat_history: List[ChatHistoryItem] = Field(default=[], description="Previous conversation turn messages")


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def convert_chat_history(history: List[ChatHistoryItem]) -> List[BaseMessage]:
    """
    Converts list of user/assistant JSON objects into LangChain BaseMessage instances.
    """
    messages: List[BaseMessage] = []
    for item in history:
        if item.role.lower() == "user":
            messages.append(HumanMessage(content=item.content))
        elif item.role.lower() in ["assistant", "ai"]:
            messages.append(AIMessage(content=item.content))
    return messages


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/health", tags=["Health Check"])
async def health_check():
    """Endpoint for Google Cloud Run health checks."""
    return {"status": "ok", "service": "City Policy RAG Server"}


@app.post("/upload_pdf", tags=["Ingestion"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Extracts text from uploaded PDF, splits it into recursive chunks, 
    embeds them using OpenAI Embeddings, and stores vectors into Supabase vector database.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files (.pdf) are supported."
        )

    if not supabase_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase client is not initialized. Please verify SUPABASE_URL and SUPABASE_KEY."
        )

    try:
        logger.info(f"Processing PDF file uploaded: {file.filename}")
        pdf_bytes = await file.read()
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        
        extracted_text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"

        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract readable text from the provided PDF. It might be scanned or image-based."
            )

        # Split text into overlapping chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        documents = text_splitter.create_documents(
            texts=[extracted_text],
            metadatas=[{"source": file.filename}]
        )

        logger.info(f"Generated {len(documents)} text chunks from PDF.")

        # Embed and insert into Supabase Vector Store
        embeddings = get_embeddings()
        
        vector_store = SupabaseVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            client=supabase_client,
            table_name="documents",
            query_name="match_documents"
        )

        return {
            "status": "success",
            "message": f"Successfully processed '{file.filename}'. Integrated {len(documents)} text chunks into Supabase.",
            "chunks_count": len(documents)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error occurred while processing PDF upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while parsing and storing PDF contents: {str(e)}"
        )


@app.post("/chat/stream", tags=["RAG Stream"])
async def chat_stream(request: QueryRequest):
    """
    RAG Endpoint featuring:
    1. Query Augmentation using Gemini Model based on chat history.
    2. Document Retrieval from Supabase using augmented query.
    3. Final Streamed Answer generation with City Policy Expert persona.
    """
    if not supabase_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection missing: Supabase client is not initialized."
        )

    try:
        # Convert incoming chat history into LangChain Message format
        formatted_history = convert_chat_history(request.chat_history)

        # ---------------------------------------------------------
        # Step 1: Query Augmentation Chain Setup
        # ---------------------------------------------------------
        augmentation_system_prompt = (
            "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. "
            "대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라. 오직 변환된 질문만 출력하라."
        )

        augmentation_prompt = ChatPromptTemplate.from_messages([
            ("system", augmentation_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        augmentation_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.0,
            google_api_key=GOOGLE_API_KEY
        )

        query_augmentation_chain = augmentation_prompt | augmentation_llm | StrOutputParser()

        logger.info(f"Executing Query Augmentation for input: '{request.query}'")
        augmented_query = await query_augmentation_chain.ainvoke({
            "question": request.query,
            "chat_history": formatted_history
        })
        
        logger.info(f"Augmented Query generated: '{augmented_query}'")

        # ---------------------------------------------------------
        # Step 2: Vector DB Retriever
        # ---------------------------------------------------------
        embeddings = get_embeddings()
        vector_store = SupabaseVectorStore(
            client=supabase_client,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = await retriever.ainvoke(augmented_query)

        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        if not context_text:
            context_text = "관련된 도시 정책 문서를 찾지 못했습니다."

        # ---------------------------------------------------------
        # Step 3: Streamed Main RAG Chain Execution
        # ---------------------------------------------------------
        city_expert_system_prompt = (
            "너는 문서에 기반해 답변하는 도시 정책 전문가야. "
            "제공된 다음 문맥(Context)정보를 바탕으로 사용자의 질문에 전문적이고 친절하며 명확하게 답변해줘.\n\n"
            "[참고 문서 내용]\n{context}"
        )

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", city_expert_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        streaming_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            streaming=True,
            google_api_key=GOOGLE_API_KEY
        )

        rag_chain = rag_prompt | streaming_llm | StrOutputParser()

        async def generate_response_stream() -> AsyncGenerator[str, None]:
            """Async generator driving real-time response streaming using native astream."""
            try:
                async for chunk in rag_chain.astream({
                    "context": context_text,
                    "chat_history": formatted_history,
                    "question": request.query
                }):
                    if chunk:
                        yield chunk
            except Exception as stream_err:
                logger.error(f"Streaming error: {str(stream_err)}")
                yield f"\n[오류 발생]: {str(stream_err)}"

        return StreamingResponse(generate_response_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error encountered during RAG Chat processing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred during query generation: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
