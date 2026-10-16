import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import requests
from bs4 import BeautifulSoup

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

# 💡 환경 변수에서 API 키를 가져오거나, 코드에 직접 넣을 수 있습니다.
# os.environ["GOOGLE_API_KEY"] = "여러분의_실제_API_키"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

def get_article_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        article = soup.find('article', class_='story-news article')
        if article: return article.get_text(strip=True)
        
        if soup.find('article'):
            return soup.find('article').get_text(strip=True)
        
        cm_ad = soup.find('div', id="CmAdContent")
        if cm_ad:
            return cm_ad.get_text(strip=True)
            
        return "기사 내용을 찾을 수 없습니다."
    except Exception as e:
        return f"URL을 가져오는 중 오류 발생: {e}"

def get_search_context(query: str) -> str:
    wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time="w")
    search = DuckDuckGoSearchResults(
        api_wrapper=wrapper, 
        source="news", 
        results_separator=';\n'
    )
    
    try:
        docs = search.invoke(query)
    except Exception as e:
        return "검색 중 오류가 발생했습니다."
    
    links = []
    if docs:
        for doc in docs.split(";\n"):
            if "link:" in doc:
                link = doc.split("link:")[1].strip()
                links.append(link)
    
    articles = []
    for link in links[:2]:
        text = get_article_text(link)
        articles.append(f"[출처: {link}]\n{text[:1500]}")
    
    return "\n\n".join(articles) if articles else "관련 기사를 찾을 수 없습니다."

@app.post("/api/chat")
def chat_with_bot(req: ChatRequest):
    try:
        # 💡 1. 올바른 모델명(gemini-1.5-flash)으로 수정
        # API 키가 환경 변수에 없다면 google_api_key="키값"을 직접 전달할 수도 있습니다.
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.2
        )
        
        # LCEL 프롬프트 설정
        prompt = ChatPromptTemplate.from_messages([
            ("system", "사용자의 질문에 대해 아래 context에 기반하여 답변하라.:\n\n{context}"),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{question}")
        ])
        
        # 체인 구성
        chain = prompt | llm | StrOutputParser()
        
        # 웹 스크래핑을 통한 컨텍스트 수집
        context = get_search_context(req.message)
        
        # 💡 2. 히스토리 역할(Role) 변환 (user/assistant -> human/ai)
        formatted_history = []
        for h in req.history:
            role = h.get("role", "user")
            content = h.get("content", "")
            
            if role in ["user", "human"]:
                formatted_history.append(("human", content))
            elif role in ["assistant", "ai"]:
                formatted_history.append(("ai", content))
            else:
                formatted_history.append(("human", content))
                
        # 결과 생성
        result = chain.invoke({
            "context": context,
            "messages": formatted_history,
            "question": req.message
        })
        
        return {"reply": result}
        
    except Exception as e:
        return {"reply": f"⚠️ 서버 처리 중 오류가 발생했습니다: {str(e)}"}
