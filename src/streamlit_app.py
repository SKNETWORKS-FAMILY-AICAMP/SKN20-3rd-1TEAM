"""
청년 정책 Q&A 챗봇 - Streamlit Frontend
"""

import streamlit as st
import sys
import os

# Add notebooks directory to path for importing youth_policy_rag.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
notebooks_path = os.path.join(project_root, 'notebooks')

# notebooks 경로를 최우선으로 추가
if notebooks_path not in sys.path:
    sys.path.insert(0, notebooks_path)

# Import the RAG system
try:
    # importlib를 사용하여 명시적으로 로드
    import importlib.util
    rag_py_path = os.path.join(notebooks_path, 'youth_policy_rag.py')
    
    if not os.path.exists(rag_py_path):
        raise FileNotFoundError(f"youth_policy_rag.py를 찾을 수 없습니다: {rag_py_path}")
    
    spec = importlib.util.spec_from_file_location("youth_policy_rag", rag_py_path)
    youth_policy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(youth_policy_module)
    
    YouthPolicyRAG = youth_policy_module.YouthPolicyRAG
    
except FileNotFoundError as e:
    st.error(f"❌ 파일을 찾을 수 없습니다: {e}")
    st.error(f"📁 검색 경로: {notebooks_path}")
    st.stop()
except AttributeError as e:
    st.error(f"❌ YouthPolicyRAG 클래스를 찾을 수 없습니다: {e}")
    st.error(f"💡 youth_policy_rag.py에 YouthPolicyRAG 클래스가 정의되어 있는지 확인하세요.")
    st.stop()
except Exception as e:
    st.error(f"❌ RAG 시스템 로드 중 오류 발생: {e}")
    st.error(f"📁 경로: {rag_py_path if 'rag_py_path' in locals() else 'N/A'}")
    st.stop()


# ========================================
# 지역 데이터
# ========================================

def get_region_options():
    """시/도 옵션 반환"""
    return [
        "서울특별시",
        "부산광역시",
        "대구광역시",
        "인천광역시",
        "광주광역시",
        "대전광역시",
        "울산광역시",
        "세종특별자치시",
        "경기도",
        "강원특별자치도",
        "충청북도",
        "충청남도",
        "전북특별자치도",
        "전라남도",
        "경상북도",
        "경상남도",
        "제주특별자치도"
    ]
 

def get_district_options(region):
    """선택된 시/도에 따른 시/군/구 옵션 반환"""
    districts = {
        "서울특별시": ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", 
                      "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", 
                      "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", 
                      "종로구", "중구", "중랑구"],
        "부산광역시": ["강서구", "금정구", "기장군", "남구", "동구", "동래구", "부산진구", 
                      "북구", "사상구", "사하구", "서구", "수영구", "연제구", "영도구", 
                      "중구", "해운대구"],
        "대구광역시": ["남구", "달서구", "달성군", "동구", "북구", "서구", "수성구", "중구"],
        "인천광역시": ["강화군", "계양구", "남동구", "동구", "미추홀구", "부평구", 
                      "서구", "연수구", "옹진군", "중구"],
        "광주광역시": ["광산구", "남구", "동구", "북구", "서구"],
        "대전광역시": ["대덕구", "동구", "서구", "유성구", "중구"],
        "울산광역시": ["남구", "동구", "북구", "울주군", "중구"],
        "세종특별자치시": ["세종시"],
        "경기도": ["고양시", "과천시", "광명시", "광주시", "구리시", "군포시", "김포시", 
                  "남양주시", "동두천시", "부천시", "성남시", "수원시", "시흥시", "안산시", 
                  "안성시", "안양시", "양주시", "양평군", "여주시", "연천군", "오산시", 
                  "용인시", "의왕시", "의정부시", "이천시", "파주시", "평택시", "포천시", 
                  "하남시", "화성시", "가평군"],
        "강원특별자치도": ["강릉시", "고성군", "동해시", "삼척시", "속초시", "양구군", 
                         "양양군", "영월군", "원주시", "인제군", "정선군", "철원군", 
                         "춘천시", "태백시", "평창군", "홍천군", "화천군", "횡성군"],
        "충청북도": ["괴산군", "단양군", "보은군", "영동군", "옥천군", "음성군", 
                    "제천시", "증평군", "진천군", "청주시", "충주시"],
        "충청남도": ["계룡시", "공주시", "금산군", "논산시", "당진시", "보령시", 
                    "부여군", "서산시", "서천군", "아산시", "예산군", "천안시", 
                    "청양군", "태안군", "홍성군"],
        "전북특별자치도": ["고창군", "군산시", "김제시", "남원시", "무주군", "부안군", 
                         "순창군", "완주군", "익산시", "임실군", "장수군", "전주시", 
                         "정읍시", "진안군"],
        "전라남도": ["강진군", "고흥군", "곡성군", "광양시", "구례군", "나주시", 
                    "담양군", "목포시", "무안군", "보성군", "순천시", "신안군", 
                    "여수시", "영광군", "영암군", "완도군", "장성군", "장흥군", 
                    "진도군", "함평군", "해남군", "화순군"],
        "경상북도": ["경산시", "경주시", "고령군", "구미시", "군위군", "김천시", 
                    "문경시", "봉화군", "상주시", "성주군", "안동시", "영덕군", 
                    "영양군", "영주시", "영천시", "예천군", "울릉군", "울진군", 
                    "의성군", "청도군", "청송군", "칠곡군", "포항시"],
        "경상남도": ["거제시", "거창군", "고성군", "김해시", "남해군", "밀양시", 
                    "사천시", "산청군", "양산시", "의령군", "진주시", "창녕군", 
                    "창원시", "통영시", "하동군", "함안군", "함양군", "합천군"],
        "제주특별자치도": ["서귀포시", "제주시"]
    }
    return districts.get(region, ["전체"])


def get_education_options():
    """학력 옵션 반환"""
    return [
        "중학교 졸업",
        "고등학교 졸업",
        "대학교 재학",
        "대학교 졸업",
        "대학원 재학",
        "대학원 졸업",
        "기타"
    ]


# ========================================
# RAG 시스템 초기화
# ========================================

@st.cache_resource
def initialize_rag():
    """RAG 시스템 초기화 (캐싱)"""
    try:
        # 프로젝트 루트 기준 절대 경로
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        db_path = os.path.join(project_root, 'data', 'vectordb')
        
        rag = YouthPolicyRAG(db_path=db_path, use_multi_query=True)
        return rag
    except Exception as e:
        st.error(f"❌ RAG 시스템 초기화 실패: {e}")
        st.error(f"📁 데이터베이스 경로: {db_path}")
        return None


# ========================================
# UI 렌더링 함수
# ========================================

def render_user_info_form():
    """사용자 기본 정보 입력 폼"""
    st.subheader("🧑 기본 정보 입력")
    
    # 폼 밖에서 지역 선택 (동적 업데이트를 위해)
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "나이",
            min_value=15,
            max_value=39,
            value=st.session_state.get("age", 25),
            help="청년 정책 대상 연령: 15~39세",
            key="age_input"
        )
    
    with col2:
        education = st.selectbox(
            "학력",
            options=get_education_options(),
            index=get_education_options().index(st.session_state.get("education", "대학교 재학")),
            key="education_input"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        region = st.selectbox(
            "시/도",
            options=get_region_options(),
            index=get_region_options().index(st.session_state.get("region", "서울특별시")),
            key="region_input"
        )
    
    with col4:
        # 지역 선택에 따라 구/군 옵션 동적 변경
        district_options = get_district_options(region)
        current_district = st.session_state.get("district", district_options[0])
        if current_district not in district_options:
            current_district = district_options[0]
        
        district = st.selectbox(
            "시/군/구",
            options=district_options,
            index=district_options.index(current_district),
            key="district_input"
        )
    
    # 저장 버튼
    if st.button("✅ 정보 저장", use_container_width=True, type="primary"):
        st.session_state["age"] = age
        st.session_state["region"] = region
        st.session_state["district"] = district
        st.session_state["education"] = education
        st.session_state["user_info_saved"] = True
        st.success("✅ 기본 정보가 저장되었습니다!")
        st.rerun()


def display_saved_user_info():
    """저장된 사용자 정보 표시 (요약 버전)"""
    if st.session_state.get("user_info_saved", False):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.success(f"👤 {st.session_state.get('age')}세 · 🎓 {st.session_state.get('education')} · 📍 {st.session_state.get('region')} {st.session_state.get('district')}")
        with col2:
            if st.button("✏️ 수정", use_container_width=True):
                st.session_state["user_info_saved"] = False
                st.rerun()
        return True
    return False


def render_question_interface(rag):
    """질문 입력 및 답변 인터페이스"""
    st.subheader("❓ 청년정책 질문하기")
    
    # 채팅 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 질문 입력
    if question := st.chat_input("청년 정책에 대해 무엇이든 물어보세요!"):
        # 사용자 정보 설정
        if rag and st.session_state.get("user_info_saved", False):
            rag.set_user_info(
                age=st.session_state.get("age"),
                region=f"{st.session_state.get('region')} {st.session_state.get('district')}",
                education=st.session_state.get("education")
            )
        
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # RAG 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                if rag:
                    try:
                        answer = rag.query(question)
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"❌ 오류가 발생했습니다: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    error_msg = "❌ RAG 시스템이 초기화되지 않았습니다."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ========================================
# 메인 앱
# ========================================

def main():
    """메인 애플리케이션"""
    # 페이지 설정
    st.set_page_config(
        page_title="청년정책 Q&A 챗봇",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # 타이틀
    st.title("💬 청년정책 Q&A 챗봇")
    st.markdown("---")
    
    # RAG 시스템 초기화
    rag = initialize_rag()
    
    if not rag:
        st.error("RAG 시스템을 초기화할 수 없습니다. 관리자에게 문의하세요.")
        st.stop()
    
    # 메인 화면 상단: 기본 정보
    if not display_saved_user_info():
        render_user_info_form()
        st.info("💡 기본 정보를 입력하고 저장하면 맞춤형 정책 상담이 가능합니다!")
    else:
        # 대화 기록 초기화 버튼 (우측 상단)
        col1, col2, col3 = st.columns([6, 1, 1])
        with col3:
            if st.button("🗑️ 대화 초기화", use_container_width=True):
                st.session_state.messages = []
                if rag:
                    rag.chat_history = []
                st.success("대화 기록이 초기화되었습니다!")
                st.rerun()
    
    st.markdown("---")
    
    # 메인: 질문 인터페이스
    if st.session_state.get("user_info_saved", False):
        render_question_interface(rag)
    else:
        st.warning("⬆️ 먼저 위에서 기본 정보를 입력해주세요!")


if __name__ == "__main__":
    main()
