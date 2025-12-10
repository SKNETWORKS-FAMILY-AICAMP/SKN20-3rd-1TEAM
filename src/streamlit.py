import streamlit as st
import os
from dotenv import load_dotenv

# 내부 모듈 임포트
# 내부 모듈 임포트
import sys
# 프로젝트 루트 경로 설정 (src 디렉토리의 상위)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.advanced_rag import initialize_rag_pipeline

# 1. 페이지 설정 (가장 먼저 호출되어야 함)
st.set_page_config(
    page_title="청년이음 선배봇",
    page_icon="🌟",
    layout="centered"
)

# 2. 커스텀 CSS 적용 (Chainlit 디자인 이식)
# - 배경: #F8FAFB + 그리드 패턴
# - 유저 말풍선: #4DE8DD (민트/틸) + #004D40 (텍스트)
# - 봇 말풍선: White + #E0F2F1 테두리
# - 폰트: Pretendard, Noto Sans KR
st.markdown("""
<style>
    /* 전체 배경 및 폰트 설정 */
    .stApp {
        background-color: #F8FAFB;
        background-image: 
            linear-gradient(#CFD8DC 1px, transparent 1px),
            linear-gradient(90deg, #CFD8DC 1px, transparent 1px);
        background-size: 30px 30px;
        font-family: 'Pretendard', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
    }

    /* 헤더 숨김 (깔끔하게) */
    header {visibility: hidden;}
    
    /* 채팅 컨테이너 스타일링 override */
    .stChatMessage {
        border-radius: 20px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* 유저 메시지 (Role: user) -> 짝수 번째 (2, 4, 6...) */
    div[data-testid="stChatMessage"]:nth-child(2n) {
        background-color: #4DE8DD !important;
        border: none;
        color: #004D40 !important;
    }
    
    /* 유저 메시지 텍스트 색상 강제 적용 */
    div[data-testid="stChatMessage"]:nth-child(2n) * {
        color: #004D40 !important;
        font-weight: 600;
    }

    /* 봇 메시지 (Role: assistant) -> 홀수 번째 (1, 3, 5...) - 환영 메시지가 1번 */
    div[data-testid="stChatMessage"]:nth-child(2n+1) {
        background-color: #ffffff !important;
        border: 1px solid #E0F2F1;
    }

    /* 채팅 입력창 스타일링 */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    
    /* 상단 타이틀 영역 */
    .title-area {
        background-color: rgba(255,255,255,0.9);
        padding: 1.5rem;
        border-radius: 16px;
        border: 2px solid #4DE8DD;
        box-shadow: 0 4px 15px rgba(77, 232, 221, 0.2);
        margin-bottom: 2rem;
        text-align: center;
    }
    .title-text {
        color: #004D40;
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .subtitle-text {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

</style>
""", unsafe_allow_html=True)

# 3. 타이틀 표시
st.markdown("""
<div class="title-area">
    <h1 class="title-text">🌟 청년이음 선배봇</h1>
    <p class="subtitle-text">청년 정책의 모든 것, 든든한 선배에게 물어보세요!</p>
</div>
""", unsafe_allow_html=True)

# 4. RAG 파이프라인 초기화 (세션별 독립 인스턴스 생성)
# 주의: @st.cache_resource를 사용하면 rag.memory(대화 기록)가 모든 유저에게 공유되는 치명적 문제가 발생함.
# 따라서 각 세션(브라우저 탭)마다 별도의 RAG 인스턴스를 생성해야 함.

if "rag" not in st.session_state:
    try:
        # 새로운 세션이 시작될 때마다 깨끗한 RAG 인스턴스 생성
        # 새로운 세션이 시작될 때마다 깨끗한 RAG 인스턴스 생성
        vectordb_path = os.path.join(project_root, "data", "vectordb")
        st.session_state.rag = initialize_rag_pipeline(vectordb_path=vectordb_path)
    except Exception as e:
        st.error(f"RAG 시스템 초기화 실패: {e}")
        st.session_state.rag = None

rag = st.session_state.rag

# 5. 세션 상태 초기화 (채팅 기록)
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 환영 메시지 추가
    welcome_msg = (
        "안녕! 나는 청년들의 든든한 정책 선배, 청년이음 선배봇🌟이야.\n\n"
        "주거, 월세, 일자리, 복지 정책 등 궁금한 점이 있으면 언제든지 나에게 물어봐!😺"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# 6. 채팅 기록 표시
# Streamlit은 매 실행마다 코드를 처음부터 다시 실행하므로 기록을 순회하며 그려줌
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. 사용자 입력 처리
if prompt := st.chat_input("질문을 입력해주세요... (예: 서울시 청년 월세 지원 정책 알려줘)"):
    # 유저 메시지 즉시 표시
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 기록에 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 봇 응답 생성
    if rag:
        with st.chat_message("assistant"):
            # 스피너 표시 (로딩 중)
            with st.spinner("정책 문서를 열심히 찾아보고 있어요... 🧐"):
                try:
                    # RAG 쿼리 실행 (동기 호출)
                    response_dict = rag.query(prompt)
                    answer = response_dict.get("answer", "죄송해요, 답변을 생성하지 못했어요. 😢")
                    summary = response_dict.get("summary", "")
                    
                    # Markdown 줄바꿈 보정 (Streamlit에서는 \n 하나로는 줄바꿈이 안 됨)
                    answer = answer.replace('\n', '  \n')
                    summary = summary.replace('\n', '  \n')
                    
                except Exception as e:
                    answer = f"오류가 발생했습니다: {str(e)}"
                    summary = ""
                    response_dict = {}

            # --- 생각의 과정 (Chain of Thought) 시각화 ---
            # Chainlit의 Step 기능을 접이식 UI로 구현
            with st.expander("🔍 선배봇의 생각 과정 보기 (Chain of Thought)"):
                if response_dict and "metadata" in response_dict:
                    meta = response_dict["metadata"]
                    
                    st.markdown("**1. 다중 쿼리 생성 (Multi-Query)**")
                    for q in meta.get("queries", []):
                        st.text(f"- {q}")
                    
                    st.markdown("**2. 지역 필터링 (Region Filter)**")
                    st.json(meta.get("region_filter"))
                    
                    st.markdown(f"**3. 검색된 문서 수**: {meta.get('num_docs_retrieved')}개")

            # --- 스트리밍 효과 구현 (스크롤 UX 개선) ---
            # 텍스트를 한 번에 뿌리면 스크롤이 바닥으로 튀지만, 스트리밍하면 시선이 따라감
            
            import time
            
            # 1. 상세 답변 스트리밍
            message_placeholder = st.empty()
            full_response = ""
            
            # 부드러운 출력을 위해 단어 단위로 쪼개거나 3글자씩 쪼개기
            # 여기서는 간단히 char 단위로 하되 속도를 빠르게 설정
            for chunk in answer.split(' '): # 공백 단위로 쪼개서 스트리밍 (줄바꿈 보존 위해 splitlines 안씀)
                full_response += chunk + " "
                # 커서 효과 추가 (▌)
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.05) 
            
            # 커서 제거 및 최종 출력
            message_placeholder.markdown(full_response)
            
            # 2. 요약이 있다면 이어서 출력
            if summary:
                st.markdown("---") # 구분선
                summary_placeholder = st.empty()
                full_summary = "**[핵심 요약]**  \n\n"
                
                for chunk in summary.split(' '):
                    full_summary += chunk + " "
                    summary_placeholder.markdown(full_summary + "▌")
                    time.sleep(0.05)
                
                summary_placeholder.markdown(full_summary)
            
            # 기록에 저장 (완성된 텍스트)
            final_content = answer
            if summary:
                 final_content += f"  \n  \n---  \n  \n**[핵심 요약]**  \n{summary}"
            
            st.session_state.messages.append({"role": "assistant", "content": final_content})
    else:
        st.error("RAG 시스템이 준비되지 않았습니다.")
