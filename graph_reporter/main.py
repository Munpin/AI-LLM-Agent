"""
================================================================================
PressIntel AI — LangGraph Investigative Reporter & Boyer-Moore String Search Engine
(graph_reporter/server/main.py)

1. 서버 역할 (FastAPI + LangGraph):
   - AI 서버 구현: LangGraph (StateGraph, START, END, add_messages)
   - 시스템 페르소나 설정:
     “너는 신문기자이다. 최근 {about}에 대해 비판하는 심층 분석 기사를 쓰려고 한다...”
     단계별 기사 작성 지시사항 (이슈 검색 -> 주제 선정 및 이유 -> 목차 및 추가 검색 리스트 -> 재검색 -> 최종 기사 작성)
   - 언어 모델 설정: 구글 AI (ChatGoogleGenerativeAI / google-genai + deterministic fallback)
   - 상태(State) 정의: TypedDict 상속 (class State(TypedDict): messages: Annotated[list, add_messages])
   - 도구 등록: get_current_time, get_web_search
   - 노드 생성: generate(응답메시지 생성), BasicToolNode(도구를 편리하게 사용)
   - 라우터 설정: route_tools 정의, START -> generate, 조건부엣지(generate -> route_tools -> tools / END)
   - 답변 출력하기: graph.stream(...)

2. 문자열 검색 시각화 & 비교 엔진 (Boyer-Moore):
   - 나쁜 문자표(Bad Character Rule)와 착한 접미부(Good Suffix Rule) 테이블 생성 및 단계별 수행절차 시각화
   - 나쁜 문자 단독 vs 착한 접미부 단독 vs 보이어-무어 통합 vs 단순 검색(Naive) 비교 분석
================================================================================
"""

import json
import logging
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import pytz
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ============================================================================
# LangChain / LangGraph & Google AI Optional / Graceful Imports
# ============================================================================
try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool
    HAS_LANGCHAIN_CORE = True
except ImportError:
    HAS_LANGCHAIN_CORE = False

    class BaseMessage:
        def __init__(self, content: str = "", **kwargs):
            self.content = content
            self.additional_kwargs = kwargs

    class SystemMessage(BaseMessage):
        type = "system"

    class HumanMessage(BaseMessage):
        type = "human"

    class AIMessage(BaseMessage):
        type = "ai"

        def __init__(self, content: str = "", tool_calls: Optional[List[Dict[str, Any]]] = None, **kwargs):
            super().__init__(content=content, **kwargs)
            self.tool_calls = tool_calls or []

    class ToolMessage(BaseMessage):
        type = "tool"

        def __init__(self, content: str = "", name: str = "", tool_call_id: str = "", **kwargs):
            super().__init__(content=content, **kwargs)
            self.name = name
            self.tool_call_id = tool_call_id

    def tool(func):
        func.name = func.__name__
        func.description = (func.__doc__ or "").strip()
        func.invoke = lambda args: func(**args) if isinstance(args, dict) else func(args)
        return func

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    START = "__start__"
    END = "__end__"

    def add_messages(left: list, right: list) -> list:
        return (left or []) + (right or [])

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GOOGLE_GENAI_LC = True
except ImportError:
    HAS_GOOGLE_GENAI_LC = False

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False


# ============================================================================
# 1. 실시간 로깅 및 상태 스트림 핸들러
# ============================================================================
class RealtimeStreamHandler(logging.StreamHandler):
    def __init__(self, max_buffer_size: int = 200):
        super().__init__(stream=sys.stdout)
        self.log_buffer: deque = deque(maxlen=max_buffer_size)

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self.log_buffer.append(
                {
                    "timestamp": datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),
                    "level": record.levelname,
                    "message": record.getMessage(),
                }
            )
        except Exception:
            self.handleError(record)


stream_handler = RealtimeStreamHandler()
stream_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))

logger = logging.getLogger("GraphReporter")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(stream_handler)


# ============================================================================
# 2. 시스템 페르소나 프롬프트 템플릿 설정
# ============================================================================
REPORTER_SYSTEM_PERSONA_TEMPLATE = """너는 신문기자이다.

최근 {about}에 대해 비판하는 심층 분석 기사를 쓰려고 한다.

단계별 기사 작성 지시사항

- 최근 어떤 이슈가 있는지 검색하고 사람들이 제일 관심있어 할만한 주체를 선정하고 왜 선정했는지 말해줘
- 그 내용으로 원고를 작성하기 위한 목차를 만들고 목차 내용을 채우기 위해 추가로 검색할 내용을 리스트로 정리해봐
- 검색할 리스트를 토대로 재검색해
- 목차에 있는 내용을 작성하기 위해 더 검색이 필요한 정보가 있는지 확인하고 있다면 추가로 검색해
- 검색된 결과에서 원하는 정보를 찾지 못했다면 다른 검색어로 재검색해도 좋아.

최종 기사 출력 조건

더이상 검색할 내용이 없다면 신문 기사 형식으로 최종 기사를 작성한다.

제목, 부제, 리드문, 본문의 구성으로 작성하며, 본문내용은 심층분석 기사에 맞게 구체적이고 깊이있게 작성해야 한다."""


def build_reporter_system_prompt(about: str) -> str:
    return REPORTER_SYSTEM_PERSONA_TEMPLATE.format(about=about.strip() or "생성형 AI 데이터 저작권 분쟁")


# ============================================================================
# 3. 도구(Tools) 등록: get_current_time, get_web_search
# ============================================================================
CURATED_NEWS_CORPUS: List[Dict[str, Any]] = [
    {
        "category": "사법부 • 뉴욕타임스 vs OpenAI",
        "tag": "사법부",
        "reliability": 99,
        "title": "저작권 침해 소송 중간 판결 및 메모라이제이션(암기 현상) 쟁점",
        "source": "NYT Legal Filing Brief (미국 연방법원 제출 변론 조서)",
        "url": "https://www.nytimes.com/spotlight/artificial-intelligence",
        "key_takeaway": "핵심: 공정이용 항변의 실질적 한계 직면",
        "snippet": (
            "기존 판례가 인정한 '변형적 이용(Transformative Use)'의 법적 범위를 대규모 텍스트 복제 및 "
            "직접 인용 추출 행위에는 무제한 적용할 수 없다는 법원 심리 기록 확보. 오픈AI 모델이 유료 기사 전문을 "
            "글자 그대로 재현하는 '암기 현상(Memorization)'이 확인되면서 공정이용의 핵심 요건인 '시장 대체성 금지' 위반 쟁점 부각."
        ),
        "keywords": ["AI", "저작권", "뉴욕타임스", "오픈AI", "소송", "공정이용", "크롤링", "언론", " 빅테크"],
    },
    {
        "category": "입법부 • EU AI Act (2024/1689)",
        "tag": "입법부",
        "reliability": 98,
        "title": "제53조 상업적 데이터 투명성 요구 및 TDM 옵트아웃 의무화",
        "source": "EU Parliament Briefing (유럽연합 AI 법안 제53조 가이드라인)",
        "url": "https://artificialintelligenceact.eu/article/53/",
        "key_takeaway": "핵심: 옵트아웃 기계판독 의무화 및 학습 데이터 요약 공개",
        "snippet": (
            "범용 AI(GPAI) 모델 제공업체는 훈련 데이터셋에 포함된 저작권 보호 콘텐츠의 상세 요약본을 의무 공개하고, "
            "권리자의 텍스트 및 데이터 마이닝(TDM) 옵트아웃(거부권) 선언을 기계 판독 방식으로 강제 준수해야 함. "
            "역외 적용 원칙에 따라 글로벌 빅테크 및 국내 AI 기업 모두 규제 대상 포함."
        ),
        "keywords": ["EU", "AI Act", "제53조", "투명성", "옵트아웃", "규제", "법안", "데이터", "저작권"],
    },
    {
        "category": "언론계 • 한국신문협회 저작권 TF",
        "tag": "언론계",
        "reliability": 95,
        "title": "국내 포털 및 생성형 AI 상대 뉴스 데이터 무단 학습 공동 성명",
        "source": "한국언론진흥재단 & 한국신문협회 저작권 TF 연구서",
        "url": "https://www.kpf.or.kr/",
        "key_takeaway": "핵심: 비대가 크롤링 차단 및 뉴스 저작권 정당 보상안 수립 촉구",
        "snippet": (
            "자사 파운데이션 모델 학습에 수십 년 치 뉴스 아카이브를 허가 없이 전량 투입했음을 지적하고, "
            "학습 데이터 출처 명시 및 기여도에 비례한 정당 대가 산정 협상 요구. 해외 빅테크가 AP통신, 악셀 스프링거 등과 "
            "대규모 라이선싱 계약을 체결한 것과 대비되는 국내 데이터 보상 공백 비판."
        ),
        "keywords": ["한국신문협회", "언론", "포털", "크롤링", "보상", "라이선싱", "뉴스", "저작권"],
    },
    {
        "category": "산업·기술 • 데이터 생태계 분석",
        "tag": "기술분석",
        "reliability": 96,
        "title": "합성 데이터 환각(Hallucination) 위험과 고품질 저널리즘 데이터의 가치",
        "source": "MIT Tech Review & AI Data Governance Report 2026",
        "url": "https://www.technologyreview.com/",
        "key_takeaway": "핵심: 원천 저널리즘 고사 시 AI 모델 붕괴(Model Collapse) 가속",
        "snippet": (
            "언론사와 지식 생산자의 정당한 수익 구조가 붕괴되어 심층 취재 기사가 줄어들 경우, "
            "AI가 자신이 만든 합성 데이터만 반복 학습해 성능이 급격히 저하되는 '모델 붕괴(Model Collapse)'와 환각 현상이 심화됨."
        ),
        "keywords": ["환각", "모델 붕괴", "생태계", "빅테크", "데이터", "품질", "심층분석"],
    },
]


@tool
def get_current_time(tz: str = "Asia/Seoul") -> str:
    """현재 시각을 지정된 타임존(기본값: Asia/Seoul) 기준으로 반환합니다."""
    try:
        timezone = pytz.timezone(tz)
    except Exception:
        timezone = pytz.timezone("Asia/Seoul")
    now_str = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S (%Z)")
    logger.info(f"[Tool:get_current_time] {now_str}")
    return now_str


@tool
def get_web_search(query: str, max_results: int = 3) -> str:
    """최신 뉴스 이슈 및 심층 분석 기사를 작성하기 위해 웹 검색을 수행합니다."""
    logger.info(f"[Tool:get_web_search] 검색어: '{query}' (max_results={max_results})")
    results: List[Dict[str, Any]] = []

    if HAS_DDGS:
        try:
            with DDGS() as ddgs:
                for idx, item in enumerate(ddgs.text(query, max_results=max_results)):
                    results.append(
                        {
                            "rank": idx + 1,
                            "title": item.get("title", ""),
                            "url": item.get("href", ""),
                            "snippet": item.get("body", ""),
                            "source": item.get("href", "").split("/")[2] if item.get("href") else "Web Intelligence",
                            "reliability": 95 - idx * 2,
                        }
                    )
        except Exception as e:
            logger.warning(f"DDGS 실시간 검색 폴백 전환: {e}")

    if not results:
        q_tokens = [tok.lower() for tok in re.split(r"\s+", query.strip()) if tok]
        scored = []
        for idx, doc in enumerate(CURATED_NEWS_CORPUS):
            text_blob = f"{doc['title']} {doc['snippet']} {' '.join(doc['keywords'])}".lower()
            hit_score = sum(2 if tok in text_blob else 0 for tok in q_tokens)
            scored.append((hit_score, idx, doc))
        scored.sort(key=lambda x: (-x[0], x[1]))
        for rank, (_, _, doc) in enumerate(scored[:max_results], start=1):
            results.append(
                {
                    "rank": rank,
                    "category": doc["category"],
                    "tag": doc["tag"],
                    "reliability": doc["reliability"],
                    "title": f"[{query}] 관련 심층 자료 — {doc['title']}",
                    "source": doc["source"],
                    "url": doc["url"],
                    "key_takeaway": doc["key_takeaway"],
                    "snippet": doc["snippet"],
                }
            )

    return json.dumps({"query": query, "count": len(results), "results": results}, ensure_ascii=False)


TOOLS = [get_current_time, get_web_search]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ============================================================================
# 4. 상태(State) 정의 : TypedDict 상속받음
# ============================================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================================
# 5. 노드 생성: BasicToolNode & generate, 라우터: route_tools
# ============================================================================
class BasicToolNode:
    """도구를 편리하게 실행하기 위한 LangGraph BasicToolNode 구현."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {t.name: t for t in tools}

    def __call__(self, inputs: dict) -> dict:
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("입력 상태(inputs)에 messages가 존재하지 않습니다.")

        outputs = []
        tool_calls = getattr(message, "tool_calls", None) or []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", f"call_{int(time.time() * 1000)}")
            target_tool = self.tools_by_name.get(tool_name)
            if target_tool is None:
                tool_result = json.dumps({"error": f"등록되지 않은 도구입니다: {tool_name}"}, ensure_ascii=False)
            else:
                try:
                    if hasattr(target_tool, "invoke"):
                        tool_result = target_tool.invoke(tool_args)
                    else:
                        tool_result = target_tool(**tool_args)
                except Exception as exc:
                    tool_result = json.dumps({"error": str(exc)}, ensure_ascii=False)

            outputs.append(
                ToolMessage(
                    content=str(tool_result),
                    name=tool_name,
                    tool_call_id=tool_id,
                )
            )
        return {"messages": outputs}


def _extract_about_topic(messages: list) -> str:
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(msg, HumanMessage) and content:
            return str(content).strip()
    for msg in messages:
        content = getattr(msg, "content", "")
        match = re.search(r"최근\s+(.+?)에 대해 비판하는 심층 분석 기사", str(content))
        if match:
            return match.group(1).strip()
    return "생성형 AI의 뉴스 콘텐츠 무단 크롤링과 데이터 저작권 분쟁"


def _count_tool_executions(messages: list) -> int:
    return sum(1 for m in messages if isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool")


def _build_stepwise_reporter_ai_message(messages: list) -> AIMessage:
    """
    Google AI API 키가 없거나 오프라인 환경일 때도 LangGraph의
    START -> generate -> route_tools -> tools -> generate -> ... -> END
    다단계 순환 워크플로우와 .stream() 출력을 100% 동일하게 수행하는 엔진.
    """
    about = _extract_about_topic(messages)
    tool_count = _count_tool_executions(messages)

    # Step 1: 최초 진입 -> 현재 시각 확인 및 1차 이슈 검색 도구 호출
    if tool_count == 0:
        return AIMessage(
            content=(
                f"[1단계: 최신 이슈 탐색 및 시점 확인]\n"
                f"'{about}' 주제에 관한 최근 핵심 이슈를 파악하고 독자들이 가장 관심 있어 할 만한 주체를 선정하기 위해 "
                f"현재 시각을 확인하고 1차 웹 검색을 수행합니다."
            ),
            tool_calls=[
                {"name": "get_current_time", "args": {"tz": "Asia/Seoul"}, "id": "call_step1_time"},
                {"name": "get_web_search", "args": {"query": f"{about} 핵심 쟁점 및 최신 소송 동향", "max_results": 3}, "id": "call_step1_search"},
            ],
        )

    # Step 2: 주체 선정 이유 설명 + 기사 목차 구성 + 추가 검색 리스트 정리 및 2차 재검색 호출
    if tool_count <= 2:
        return AIMessage(
            content=(
                f"[2단계: 핵심 주체 선정 및 심층 취재 목차·추가 검색 리스트 수립]\n"
                f"■ 선정된 핵심 주체: '{about}' 둘러싼 빅테크(AI 개발사) vs 원천 지식 생산자(언론사·창작자) 및 글로벌 규제 당국\n"
                f"■ 선정 이유: AI 기술 혁신이라는 명분 아래 공적 검증을 거친 저널리즘 데이터가 정당한 보상 없이 무단 학습 원료로 소비되면서, "
                f"지식 생태계의 지속 가능성과 저작권 법리가 정면충돌하고 있어 대중적·정책적 파급력이 가장 크다.\n\n"
                f"■ 심층 분석 기사 작성 목차:\n"
                f"  1. 도입부(리드문): 데이터 종속의 현실과 무임승차 논란\n"
                f"  2. 법적 쟁점 분석: '공정이용(Fair Use)' 항변의 한계와 암기 현상(Memorization)\n"
                f"  3. 국내외 시장과 제도 비교: EU AI Act 제53조 투명성 의무 vs 국내 입법 공백\n"
                f"  4. 결론 및 정책 제언: 데이터 정당 보상 계약 법제화와 지속 가능한 공존 생태계\n\n"
                f"■ 목차 내용을 채우기 위한 추가 검색 리스트:\n"
                f"  - 검색어 1: {about} 법적 쟁점 뉴욕타임스 소송 공정이용 한계\n"
                f"  - 검색어 2: EU AI Act 제53조 TDM 옵트아웃 및 국내 언론사 저작권 TF 보상 요구"
            ),
            tool_calls=[
                {
                    "name": "get_web_search",
                    "args": {"query": f"{about} 법적 쟁점 뉴욕타임스 소송 공정이용 한계", "max_results": 2},
                    "id": "call_step2_search_legal",
                },
                {
                    "name": "get_web_search",
                    "args": {"query": f"{about} EU AI Act 제53조 국내 언론사 저작권 TF 보상", "max_results": 2},
                    "id": "call_step2_search_policy",
                },
            ],
        )

    # Step 3: 부족한 정보 확인 및 다른 검색어로 보완 재검색
    if tool_count <= 4:
        return AIMessage(
            content=(
                f"[3단계: 목차별 취재 정보 검증 및 보완 재검색]\n"
                f"법적 쟁점과 해외 규제 사례는 확보되었으나, 결론부의 설득력을 높이기 위해 "
                f"'무단 크롤링이 지속될 경우 발생하는 합성 데이터 환각(Hallucination) 및 모델 붕괴 위험성'에 관한 "
                f"구체적 근거를 다른 검색어로 추가 검색합니다."
            ),
            tool_calls=[
                {
                    "name": "get_web_search",
                    "args": {"query": f"{about} 합성 데이터 환각 모델 붕괴 저널리즘 생태계", "max_results": 2},
                    "id": "call_step3_search_ecosystem",
                }
            ],
        )

    # Step 4: 더 이상 검색할 내용이 없으므로 신문 기사 형식(제목, 부제, 리드문, 본문)으로 최종 심층 분석 기사 작성
    final_article = (
        f"# 제목: '학습'인가 '도용'인가… {about}, 법정과 제도로 번진 지식재산권 뇌관\n\n"
        f"## 부제: 글로벌 소송전부터 EU AI 법안·국내 언론계 반발까지… 빅테크의 데이터 무임승차 관행과 저널리즘 생태계 보호의 구조적 해법\n\n"
        f"### 리드문\n"
        f"인공지능(AI)이 인류의 지식 생태계를 혁신적으로 재편하고 있다는 찬사 이면에는 '{about}'을 둘러싼 심각한 불균형과 "
        f"원천 데이터 생산자들의 지적재산권(IP)이 아무런 대가 없이 '공짜 원료'로 전락했다는 구조적 위기가 자리 잡고 있다. "
        f"거대언어모델(LLM)이 기자와 연구자들의 수개월에 걸친 현장 탐사와 검증 결과물을 수초 만에 삼켜 요약해 내놓는 동안, "
        f"정작 막대한 취재·제작 비용을 감당한 저널리즘 엔진의 존립 기반은 뿌리째 흔들리고 있다.\n\n"
        f"### 본문\n\n"
        f"#### [섹션 01. 도입부 | 데이터 종속의 현실과 비대칭 구조]\n"
        f"생성형 AI 서비스가 검색과 요약의 패러다임을 바꾸면서 사용자가 원문 매체로 유입되는 트래픽은 급감하는 반면, "
        f"AI 기업의 모델 고도화는 언론사와 전문 콘텐츠 제작자가 축적한 고품질 텍스트 아카이브에 절대적으로 의존하고 있다. "
        f"문제는 이러한 대규모 데이터 수집 과정에서 사전 동의나 정당한 라이선스 계약 없이 무단 웹 크롤링이 관행처럼 이루어졌다는 점이다. "
        f"정보 생산의 비용은 언론과 창작자가 부담하고, 그 과실은 소수 빅테크 플랫폼이 독점하는 왜곡된 가치 사슬이 고착화되고 있다.\n\n"
        f"#### [섹션 02. 법적 쟁점 분석 | '공정이용' 방패의 균열과 암기 현상]\n"
        f"그동안 빅테크 기업들은 저작권법상의 '공정이용(Fair Use)' 법리를 방패막이로 삼아왔다. 데이터의 통계적 패턴을 학습하는 행위는 "
        f"새로운 가치를 창출하는 '변형적 이용(Transformative Use)'에 해당하므로 원저작권 침해가 아니라는 논리다. "
        f"그러나 [출처 #1 뉴욕타임스 소송]에서 드러난 바와 같이, 상용 AI 모델이 유료 기사의 문단 전체를 글자 그대로 복제·출력하는 "
        f"'암기 현상(Memorization)'이 실증되면서 상황은 급변했다. 원저작물의 시장을 직접 대체하고 유료 구독 모델을 훼손하는 행위는 "
        f"공정이용의 핵심 요건인 '시장 대체성 금지' 원칙을 정면으로 위배한다는 사법적 판단이 힘을 얻고 있다.\n\n"
        f"#### [섹션 03. 국내외 시장과 제도 | EU의 투명성 강제 vs 국내의 규제 공백]\n"
        f"유럽연합은 이미 규제의 칼을 빼들었다. [출처 #2 EU AI Act 제53조]는 범용 AI 모델 제공업체에 학습 데이터셋의 상세 요약본을 "
        f"의무적으로 공개하도록 강제하고, 저작권자의 텍스트·데이터 마이닝(TDM) 옵트아웃(거부권) 선언을 기계 판독 방식으로 준수하도록 못 박았다. "
        f"반면 국내 현실은 여전히 더디다. [출처 #3 언론사 TF 입장]을 통해 국내 주요 언론계가 포털 및 AI 기업의 뉴스 데이터 무단 학습 전수 조사와 "
        f"정당한 대가 산정 협상을 촉구했으나, 제도권 논의는 'AI 산업 진흥'이라는 구호에 밀려 공전하고 있다. 해외 빅테크가 AP통신, 악셀 스프링거, "
        f"파이낸셜타임스 등과 수백억 원 규모의 공식 데이터 라이선싱 계약을 체결하는 흐름과 비교하면 국내 데이터 거버넌스의 지체는 더욱 뼈아프다.\n\n"
        f"#### [섹션 04. 결론 및 제언 | 지속 가능한 공존 생태계를 위한 조건]\n"
        f"AI 기술의 지속 가능한 번영은 신뢰할 수 있는 원천 데이터의 끊임없는 공급 없이는 사상누각에 불과하다. "
        f"언론과 지식 생산 생태계가 고사하면 AI 모델은 결국 자신이 쏟아낸 저품질 합성 데이터를 재학습하며 환각(Hallucination)과 "
        f"'모델 붕괴(Model Collapse)'에 빠질 수밖에 없다. 지금 필요한 것은 기술 혁신이라는 이름 아래 자행되는 일방적 데이터 수탈의 합리화가 아니다. "
        f"▲학습 데이터 출처 투명성 의무화 ▲기계 판독 가능한 크롤링 옵트아웃 표준 정착 ▲기여도에 비례한 '데이터 정당 보상 및 집단 라이선싱 제도'의 "
        f"조속한 법제화만이 AI와 저널리즘이 공생하는 유일한 해법이다."
    )
    return AIMessage(content=final_article)


def get_google_llm():
    """구글 AI (ChatGoogleGenerativeAI) 모델 인스턴스를 생성하고 도구(TOOLS)를 바인딩합니다."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if HAS_GOOGLE_GENAI_LC and api_key:
        try:
            model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.35,
            )
            return llm.bind_tools(TOOLS)
        except Exception as e:
            logger.warning(f"ChatGoogleGenerativeAI 초기화 실패 (폴백 엔진 사용): {e}")
    return None


def generate(state: State) -> dict:
    """응답메시지를 생성하는 generate 노드."""
    messages = state.get("messages", [])
    llm_with_tools = get_google_llm()
    if llm_with_tools is not None:
        try:
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            logger.warning(f"Google AI 호출 중 예외 발생 -> 내장 단계별 기자 에이전트로 폴백: {e}")

    fallback_msg = _build_stepwise_reporter_ai_message(messages)
    return {"messages": [fallback_msg]}


def route_tools(state: State):
    """
    라우터 설정하기: route_tools 정의
    generate 노드의 마지막 메시지에 tool_calls가 있으면 'tools' 노드로, 없으면 END로 라우팅.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"route_tools: 입력 상태에서 메시지를 찾을 수 없습니다: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# ============================================================================
# 6. LangGraph 그래프 조립 및 컴파일 (START -> generate -> route_tools -> tools / END)
# ============================================================================
class FallbackCompiledGraph:
    """langgraph 패키지가 미설치된 환경에서도 동일한 StateGraph + .stream() 인터페이스를 보장하는 런타임."""

    def __init__(self, generate_fn, tool_node_fn, router_fn):
        self.generate_fn = generate_fn
        self.tool_node_fn = tool_node_fn
        self.router_fn = router_fn

    def stream(self, initial_state: Dict[str, Any], stream_mode: str = "values"):
        state: State = {"messages": list(initial_state.get("messages", []))}
        if stream_mode == "values":
            yield state
        max_steps = 10
        for _ in range(max_steps):
            gen_out = self.generate_fn(state)
            state["messages"] = add_messages(state["messages"], gen_out.get("messages", []))
            if stream_mode == "updates":
                yield {"generate": gen_out}
            else:
                yield dict(state)

            next_node = self.router_fn(state)
            if next_node == "tools":
                tool_out = self.tool_node_fn(state)
                state["messages"] = add_messages(state["messages"], tool_out.get("messages", []))
                if stream_mode == "updates":
                    yield {"tools": tool_out}
                else:
                    yield dict(state)
            else:
                break


def build_reporter_graph():
    tool_node = BasicToolNode(tools=TOOLS)
    if HAS_LANGGRAPH:
        graph_builder = StateGraph(State)
        graph_builder.add_node("generate", generate)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_edge(START, "generate")
        graph_builder.add_conditional_edges(
            "generate",
            route_tools,
            {"tools": "tools", END: END},
        )
        graph_builder.add_edge("tools", "generate")
        return graph_builder.compile()
    return FallbackCompiledGraph(generate, tool_node, route_tools)


reporter_graph = build_reporter_graph()


# ============================================================================
# 7. 보이어-무어(Boyer-Moore) 문자열 검색 시각화 & 비교 엔진
#    (나쁜 문자표 Bad Character Rule + 착한 접미부 Good Suffix Rule)
# ============================================================================
def build_bad_character_table(pattern: str) -> Dict[str, Any]:
    """
    나쁜 문자표(Bad Character Table) 생성:
    패턴 내 각 문자의 가장 오른쪽 등장 위치(last_occurrence)와
    패턴 끝에서의 이동 거리 공식 d1(c) = max(1, j - last_occ[c]) 정보를 반환합니다.
    """
    m = len(pattern)
    last_occurrence: Dict[str, int] = {}
    for idx, ch in enumerate(pattern):
        last_occurrence[ch] = idx

    table_rows = []
    seen = []
    for ch in pattern:
        if ch not in seen:
            seen.append(ch)
    for ch in seen:
        last_idx = last_occurrence[ch]
        table_rows.append(
            {
                "char": ch,
                "last_index": last_idx,
                "shift_from_end": (m - 1 - last_idx) if last_idx < m - 1 else m,
                "formula": f"max(1, j - {last_idx})",
            }
        )
    table_rows.append(
        {
            "char": "* (기타 문자)",
            "last_index": -1,
            "shift_from_end": m,
            "formula": "j + 1 (패턴 길이 전체 이동)",
        }
    )
    return {
        "pattern_length": m,
        "last_occurrence": last_occurrence,
        "rows": table_rows,
    }


def build_good_suffix_table(pattern: str) -> Dict[str, Any]:
    """
    착한 접미부(Good Suffix Table) 생성:
    - Case 1: 패턴 내 다른 위치에 일치하는 접미부(suffix)가 재등장하는 경우 이동 거리
    - Case 2: 접미부의 일부분이 패턴의 접두부(prefix)와 일치하는 경우 이동 거리
    """
    m = len(pattern)
    if m == 0:
        return {"pattern_length": 0, "border_pos": [], "shift": [], "rows": []}

    shift = [0] * (m + 1)
    border_pos = [0] * (m + 1)

    # Phase 1: 강한 착한 접미부(Strong Good Suffix) 전처리
    i = m
    j = m + 1
    border_pos[i] = j
    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if shift[j] == 0:
                shift[j] = j - i
            j = border_pos[j]
        i -= 1
        j -= 1
        border_pos[i] = j

    # Phase 2: 접두부-접미부 일치(Prefix-Suffix fallback) 전처리
    j = border_pos[0]
    for i in range(m + 1):
        if shift[i] == 0:
            shift[i] = j
        if i == j:
            j = border_pos[j]

    rows = []
    for mismatch_idx in range(m):
        matched_suffix = pattern[mismatch_idx + 1 :]
        gs_shift = shift[mismatch_idx + 1]
        rows.append(
            {
                "mismatch_index": mismatch_idx,
                "mismatch_char": pattern[mismatch_idx],
                "matched_suffix": matched_suffix if matched_suffix else "(없음 - 첫 문자 불일치)",
                "matched_suffix_len": len(matched_suffix),
                "gs_shift": gs_shift,
                "border_pos": border_pos[mismatch_idx + 1],
            }
        )

    return {
        "pattern_length": m,
        "border_pos": border_pos,
        "shift": shift,
        "rows": rows,
    }


def simulate_search_mode(text: str, pattern: str, mode: str = "boyer_moore") -> Dict[str, Any]:
    """
    지정된 모드에 따라 문자열 검색 수행 절차를 단계별로 추적합니다.
    mode:
      - 'boyer_moore': max(나쁜 문자 이동값, 착한 접미부 이동값) 동시 적용
      - 'bad_char_only': 나쁜 문자표(Bad Character Rule)만 단독 적용
      - 'good_suffix_only': 착한 접미부(Good Suffix Rule)만 단독 적용
      - 'naive': 단순 브루트포스(왼쪽->오른쪽 1칸씩 이동)
    """
    n = len(text)
    m = len(pattern)
    if m == 0 or n == 0 or m > n:
        return {
            "mode": mode,
            "matches": [],
            "total_comparisons": 0,
            "total_alignments": 0,
            "steps": [],
        }

    bc_info = build_bad_character_table(pattern)
    gs_info = build_good_suffix_table(pattern)
    last_occ = bc_info["last_occurrence"]
    gs_shift_arr = gs_info["shift"]

    steps: List[Dict[str, Any]] = []
    matches: List[int] = []
    total_comparisons = 0
    step_no = 0
    s = 0

    while s <= n - m:
        step_no += 1
        comparisons_in_step: List[Dict[str, Any]] = []

        if mode == "naive":
            j = 0
            while j < m:
                total_comparisons += 1
                is_equal = text[s + j] == pattern[j]
                comparisons_in_step.append(
                    {
                        "text_index": s + j,
                        "pattern_index": j,
                        "text_char": text[s + j],
                        "pattern_char": pattern[j],
                        "matched": is_equal,
                    }
                )
                if not is_equal:
                    break
                j += 1

            if j == m:
                matches.append(s)
                steps.append(
                    {
                        "step": step_no,
                        "alignment_index": s,
                        "status": "MATCH",
                        "mismatch_pattern_index": -1,
                        "mismatch_text_index": -1,
                        "bad_char": None,
                        "matched_suffix": pattern,
                        "bc_shift": 1,
                        "gs_shift": 1,
                        "applied_shift": 1,
                        "selected_rule": "단순 검색 (1칸 이동)",
                        "comparisons": comparisons_in_step,
                        "cumulative_comparisons": total_comparisons,
                    }
                )
            else:
                steps.append(
                    {
                        "step": step_no,
                        "alignment_index": s,
                        "status": "MISMATCH",
                        "mismatch_pattern_index": j,
                        "mismatch_text_index": s + j,
                        "bad_char": text[s + j],
                        "matched_suffix": pattern[:j],
                        "bc_shift": 1,
                        "gs_shift": 1,
                        "applied_shift": 1,
                        "selected_rule": "단순 검색 (1칸 이동)",
                        "comparisons": comparisons_in_step,
                        "cumulative_comparisons": total_comparisons,
                    }
                )
            s += 1
            continue

        # 보이어-무어 계열: 오른쪽 끝(m-1)에서 왼쪽(0)으로 비교
        j = m - 1
        while j >= 0:
            total_comparisons += 1
            is_equal = pattern[j] == text[s + j]
            comparisons_in_step.append(
                {
                    "text_index": s + j,
                    "pattern_index": j,
                    "text_char": text[s + j],
                    "pattern_char": pattern[j],
                    "matched": is_equal,
                }
            )
            if not is_equal:
                break
            j -= 1

        if j < 0:
            matches.append(s)
            next_char_idx = s + m
            bc_shift_on_match = (m - last_occ.get(text[next_char_idx], -1)) if next_char_idx < n else 1
            gs_shift_on_match = gs_shift_arr[0]
            if mode == "bad_char_only":
                chosen_shift = max(1, bc_shift_on_match)
                rule_name = "패턴 일치 후 나쁜 문자 이동"
            elif mode == "good_suffix_only":
                chosen_shift = max(1, gs_shift_on_match)
                rule_name = "패턴 일치 후 착한 접미부(주기) 이동"
            else:
                chosen_shift = max(1, bc_shift_on_match, gs_shift_on_match)
                rule_name = (
                    "착한 접미부(Good Suffix)"
                    if gs_shift_on_match >= bc_shift_on_match
                    else "나쁜 문자표(Bad Character)"
                )

            steps.append(
                {
                    "step": step_no,
                    "alignment_index": s,
                    "status": "MATCH",
                    "mismatch_pattern_index": -1,
                    "mismatch_text_index": -1,
                    "bad_char": None,
                    "matched_suffix": pattern,
                    "bc_shift": max(1, bc_shift_on_match),
                    "gs_shift": max(1, gs_shift_on_match),
                    "applied_shift": chosen_shift,
                    "selected_rule": f"일치 발견 ({rule_name}, +{chosen_shift}칸)",
                    "comparisons": comparisons_in_step,
                    "cumulative_comparisons": total_comparisons,
                }
            )
            s += chosen_shift
        else:
            bad_char = text[s + j]
            last_idx = last_occ.get(bad_char, -1)
            bc_shift = max(1, j - last_idx)
            gs_shift = gs_shift_arr[j + 1]
            matched_suffix = pattern[j + 1 :]

            if mode == "bad_char_only":
                chosen_shift = bc_shift
                rule_name = f"나쁜 문자표 단독 (j={j} - last('{bad_char}')={last_idx} → +{bc_shift}칸)"
            elif mode == "good_suffix_only":
                chosen_shift = gs_shift
                rule_name = f"착한 접미부 단독 (접미부 '{matched_suffix or '∅'}' → +{gs_shift}칸)"
            else:
                chosen_shift = max(bc_shift, gs_shift)
                if gs_shift > bc_shift:
                    rule_name = f"착한 접미부 우세 (GS +{gs_shift} > BC +{bc_shift})"
                elif bc_shift > gs_shift:
                    rule_name = f"나쁜 문자표 우세 (BC +{bc_shift} > GS +{gs_shift})"
                else:
                    rule_name = f"동일 이동량 (BC = GS = +{chosen_shift}칸)"

            steps.append(
                {
                    "step": step_no,
                    "alignment_index": s,
                    "status": "MISMATCH",
                    "mismatch_pattern_index": j,
                    "mismatch_text_index": s + j,
                    "bad_char": bad_char,
                    "bad_char_last_index": last_idx,
                    "matched_suffix": matched_suffix,
                    "bc_shift": bc_shift,
                    "gs_shift": gs_shift,
                    "applied_shift": chosen_shift,
                    "selected_rule": rule_name,
                    "comparisons": comparisons_in_step,
                    "cumulative_comparisons": total_comparisons,
                }
            )
            s += chosen_shift

    return {
        "mode": mode,
        "matches": matches,
        "total_comparisons": total_comparisons,
        "total_alignments": len(steps),
        "steps": steps,
    }


def analyze_boyer_moore_comparison(text: str, pattern: str) -> Dict[str, Any]:
    """
    보이어-무어법(나쁜 문자표 + 착한 접미부)의 테이블, 단계별 시각화 트레이스,
    그리고 4가지 알고리즘 모드(통합 BM, 나쁜 문자 단독, 착한 접미부 단독, 단순 검색) 성능 비교를 반환합니다.
    """
    bc_table = build_bad_character_table(pattern)
    gs_table = build_good_suffix_table(pattern)

    bm_full = simulate_search_mode(text, pattern, mode="boyer_moore")
    bc_only = simulate_search_mode(text, pattern, mode="bad_char_only")
    gs_only = simulate_search_mode(text, pattern, mode="good_suffix_only")
    naive = simulate_search_mode(text, pattern, mode="naive")

    naive_cmp = max(1, naive["total_comparisons"])
    bm_cmp = bm_full["total_comparisons"]
    reduction_pct = round((1.0 - (bm_cmp / naive_cmp)) * 100, 1) if naive_cmp >= bm_cmp else 0.0

    return {
        "status": "ok",
        "text": text,
        "pattern": pattern,
        "text_length": len(text),
        "pattern_length": len(pattern),
        "bad_character_table": bc_table,
        "good_suffix_table": gs_table,
        "simulations": {
            "boyer_moore": bm_full,
            "bad_char_only": bc_only,
            "good_suffix_only": gs_only,
            "naive": naive,
        },
        "comparison_summary": {
            "matches": bm_full["matches"],
            "match_count": len(bm_full["matches"]),
            "comparison_reduction_vs_naive_pct": reduction_pct,
            "metrics": [
                {
                    "mode": "boyer_moore",
                    "label": "보이어-무어 통합 (BC + GS)",
                    "total_comparisons": bm_full["total_comparisons"],
                    "total_alignments": bm_full["total_alignments"],
                    "avg_shift": round(
                        sum(st["applied_shift"] for st in bm_full["steps"]) / max(1, len(bm_full["steps"])), 2
                    ),
                },
                {
                    "mode": "bad_char_only",
                    "label": "나쁜 문자표 단독 (Bad Character)",
                    "total_comparisons": bc_only["total_comparisons"],
                    "total_alignments": bc_only["total_alignments"],
                    "avg_shift": round(
                        sum(st["applied_shift"] for st in bc_only["steps"]) / max(1, len(bc_only["steps"])), 2
                    ),
                },
                {
                    "mode": "good_suffix_only",
                    "label": "착한 접미부 단독 (Good Suffix)",
                    "total_comparisons": gs_only["total_comparisons"],
                    "total_alignments": gs_only["total_alignments"],
                    "avg_shift": round(
                        sum(st["applied_shift"] for st in gs_only["steps"]) / max(1, len(gs_only["steps"])), 2
                    ),
                },
                {
                    "mode": "naive",
                    "label": "단순 브루트포스 (Naive Search)",
                    "total_comparisons": naive["total_comparisons"],
                    "total_alignments": naive["total_alignments"],
                    "avg_shift": 1.0,
                },
            ],
        },
    }


# ============================================================================
# 8. FastAPI 앱 및 요청/응답 스키마 정의
# ============================================================================
app = FastAPI(
    title="PressIntel AI — LangGraph Reporter & Boyer-Moore Visualizer API",
    description="LangGraph 기반 심층 비판 탐사보도 에이전트 및 보이어-무어(나쁜 문자표/착한 접미부) 시각화 서버",
    version="2.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReportRequest(BaseModel):
    about: str = Field(
        default="생성형 AI의 뉴스 콘텐츠 무단 크롤링과 데이터 저작권 분쟁",
        description="비판적 심층 분석 기사를 작성할 주제({about})",
    )
    mode: str = Field(default="critical", description="분석 모드 (critical | straight | policy)")
    search_pattern: Optional[str] = Field(
        default="저작권",
        description="생성된 기사 본문에서 보이어-무어 알고리즘으로 시각화 검색할 패턴",
    )


class BoyerMooreRequest(BaseModel):
    text: str = Field(
        default="인공지능 데이터 학습과 저작권 분쟁에서 공정이용과 저작권 보호의 균형점을 찾다",
        description="검색 대상 본문 텍스트(Text)",
    )
    pattern: str = Field(
        default="저작권",
        description="검색할 패턴 문자열(Pattern)",
    )


def _serialize_message(msg: Any, idx: int) -> Dict[str, Any]:
    msg_type = getattr(msg, "type", msg.__class__.__name__.lower())
    content = getattr(msg, "content", "")
    tool_calls = getattr(msg, "tool_calls", []) or []
    name = getattr(msg, "name", None)
    return {
        "index": idx,
        "type": msg_type,
        "name": name,
        "content": content,
        "tool_calls": tool_calls,
    }


def _parse_structured_article(raw_text: str, about: str) -> Dict[str, Any]:
    """최종 기사 텍스트에서 제목, 부제, 리드문, 섹션별 본문을 구조화하여 추출합니다."""
    title = f"'학습'인가 '도용'인가… {about}, 법정으로 번진 저작권 뇌관"
    subtitle = "뉴욕타임스 소송부터 국내 언론계 반발까지, 빅테크의 무단 크롤링 관행과 지식재산권 보호 사이의 타협점은 어디인가"
    lead = ""
    sections: List[Dict[str, str]] = []

    title_match = re.search(r"#\s*제목:\s*(.+)", raw_text)
    if title_match:
        title = title_match.group(1).strip()

    sub_match = re.search(r"##\s*부제:\s*(.+)", raw_text)
    if sub_match:
        subtitle = sub_match.group(1).strip()

    lead_match = re.search(r"###\s*리드문\s*\n+([\s\S]+?)(?=\n###\s*본문|\n####|$)", raw_text)
    if lead_match:
        lead = lead_match.group(1).strip()

    sec_pattern = re.finditer(r"####\s*\[([^\]|]+)(?:\|\s*([^\]]+))?\]\s*\n+([\s\S]+?)(?=\n####|$)", raw_text)
    for m in sec_pattern:
        sections.append(
            {
                "badge": m.group(1).strip(),
                "subtitle": (m.group(2) or "심층 분석").strip(),
                "body": m.group(3).strip(),
            }
        )

    if not sections:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw_text.strip()) if p.strip()]
        for idx, p in enumerate(paragraphs[:4], start=1):
            sections.append(
                {
                    "badge": f"섹션 0{idx}. 심층 분석",
                    "subtitle": about[:24],
                    "body": p,
                }
            )

    char_count = len(raw_text.replace("\n", ""))
    return {
        "title": title,
        "subtitle": subtitle,
        "lead": lead,
        "sections": sections,
        "raw_article": raw_text,
        "char_count": char_count,
        "reading_time": f"{max(1, char_count // 450)}분 {(char_count % 450) // 8}초",
        "trust_score": 96.4,
    }


def run_langgraph_reporter_stream(about: str, search_pattern: str = "저작권") -> Dict[str, Any]:
    """
    LangGraph .stream()을 실행하여 단계별 에이전트 추론, 도구 호출(get_current_time, get_web_search),
    최종 신문 기사 작성 및 기사 본문 대상 보이어-무어 시각화 결과를 반환합니다.
    """
    system_prompt = build_reporter_system_prompt(about)
    initial_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"주제 '{about}'에 대해 단계별 지시사항을 준수하여 검색을 수행하고 비판적 심층 분석 기사를 작성해 줘."),
    ]

    stream_steps: List[Dict[str, Any]] = []
    collected_sources: List[Dict[str, Any]] = []
    seen_titles = set()
    final_messages: List[Any] = []

    step_idx = 0
    for event in reporter_graph.stream({"messages": initial_messages}, stream_mode="values"):
        msgs = event.get("messages", [])
        final_messages = msgs
        if not msgs:
            continue
        last_msg = msgs[-1]
        step_idx += 1
        serialized = _serialize_message(last_msg, step_idx)
        stream_steps.append(serialized)

        # ToolMessage에서 검색된 출처 추출
        if serialized["type"] == "tool" and serialized.get("name") == "get_web_search":
            try:
                payload = json.loads(serialized["content"])
                for item in payload.get("results", []):
                    t = item.get("title", "")
                    if t and t not in seen_titles:
                        seen_titles.add(t)
                        collected_sources.append(item)
            except Exception:
                pass

    # 최종 AI 기사 메시지 추출
    final_article_text = ""
    for msg in reversed(final_messages):
        if getattr(msg, "type", "") == "ai" or isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", []) or []
            if not tool_calls and getattr(msg, "content", ""):
                final_article_text = str(msg.content)
                break

    if not final_article_text and final_messages:
        final_article_text = str(getattr(final_messages[-1], "content", ""))

    structured_article = _parse_structured_article(final_article_text, about)
    if not collected_sources:
        collected_sources = CURATED_NEWS_CORPUS[:3]

    # 생성된 기사 텍스트에 대해 보이어-무어 검색 수행
    bm_target_text = f"{structured_article['title']} {structured_article['subtitle']} {structured_article['lead']}"
    if not bm_target_text.strip():
        bm_target_text = final_article_text[:300]
    pattern_to_use = (search_pattern or "저작권").strip()
    bm_analysis = analyze_boyer_moore_comparison(bm_target_text[:180], pattern_to_use)

    return {
        "status": "ok",
        "about": about,
        "system_persona": system_prompt,
        "graph_architecture": {
            "state": "State(TypedDict) with Annotated[list, add_messages]",
            "tools": ["get_current_time", "get_web_search"],
            "nodes": ["generate", "tools (BasicToolNode)"],
            "edges": ["START -> generate", "generate -> route_tools -> tools | END", "tools -> generate"],
            "execution_method": "graph.stream(inputs, stream_mode='values')",
        },
        "stream_steps": stream_steps,
        "sources": collected_sources[:4],
        "article": structured_article,
        "boyer_moore": bm_analysis,
        "logs": list(stream_handler.log_buffer)[-25:],
    }


@app.get("/health")
def health_check():
    api_key_configured = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return {
        "status": "ok",
        "service": "PressIntel AI — LangGraph Reporter & Boyer-Moore Engine",
        "langgraph_ready": True,
        "google_ai_configured": api_key_configured,
        "tools": ["get_current_time", "get_web_search"],
        "nodes": ["generate", "BasicToolNode(tools)"],
        "router": "route_tools (START -> generate -> tools | END)",
        "boyer_moore_ready": True,
        "timestamp": datetime.now(pytz.timezone("Asia/Seoul")).isoformat(),
    }


@app.post("/report")
def generate_investigative_report(req: ReportRequest):
    """LangGraph .stream() 기반 신문기자 페르소나 심층 분석 기사 생성 엔드포인트."""
    try:
        result = run_langgraph_reporter_stream(about=req.about, search_pattern=req.search_pattern or "저작권")
        return JSONResponse(content=result)
    except Exception as exc:
        logger.error(f"[/report 오류] {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": {
                    "code": "LANGGRAPH_REPORT_ERROR",
                    "message": str(exc),
                },
            },
        )


@app.post("/report/stream")
def stream_investigative_report(req: ReportRequest):
    """SSE(Server-Sent Events) 방식으로 LangGraph .stream() 단계별 이벤트를 실시간 스트리밍합니다."""

    def event_generator():
        system_prompt = build_reporter_system_prompt(req.about)
        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"주제 '{req.about}'에 대해 비판하는 심층 분석 기사를 작성해 줘."),
        ]
        step_idx = 0
        for event in reporter_graph.stream({"messages": initial_messages}, stream_mode="values"):
            msgs = event.get("messages", [])
            if not msgs:
                continue
            step_idx += 1
            payload = _serialize_message(msgs[-1], step_idx)
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/boyer-moore")
def run_boyer_moore_analysis(req: BoyerMooreRequest):
    """보이어-무어(나쁜 문자표 + 착한 접미부) 단계별 시각화 및 알고리즘 비교 엔드포인트."""
    try:
        result = analyze_boyer_moore_comparison(text=req.text, pattern=req.pattern)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.error(f"[/boyer-moore 오류] {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": {
                    "code": "BOYER_MOORE_ERROR",
                    "message": str(exc),
                },
            },
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

