# 청년·1인 가구 생활복지·지원 챗봇 🤖
"수천 개의 정책 중 나에게 딱 맞는 혜택, AI가 10초 만에 찾아드립니다!"


🏆 [SKN Family AI캠프] 3차 단위 프로젝트 📅 개발 기간: 2025.12.10 ~ 2025.12.11

# 팀명 : 청 바 지 (청춘은 바로 지금)

## 팀원 소개

| 나호성 | 강민지 | 이지은 | 조준상 | 홍혜원 |
|---|---|---|---|---|
| <img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/aa1c4b5b-f271-44bc-8765-fb0717a255fb" /> | <img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/93140ba3-81e2-4079-8084-8ebab3121a26" /> | <img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/0ac142c1-01de-4130-bcdf-cf7bf026a371" /> | <img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/4f23001a-38bb-41bb-ae6f-0ee01d97eebf" /> | <img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/b133989b-a180-44cc-8fd6-d593e37aee8f" /> |
| 팀장(PM) | 프롬프트엔지니어링 | 프로젝트PD | 프론트엔드 | 슈퍼프로그래머 |


# 1. 🖥️ 프로젝트 개요 및 목표

## 프로젝트명 : 청년·1인 가구 생활복지·지원제도 탐색 Q&A 챗봇

2023년 청년 정책 실태조사에 따르면 청년 정책 수혜율 11%, 몰라서 못 받는 혜택들이 많습니다.<br>
본 프로젝트는  청년·1인 가구등 받을 수 있는 지원제도를 문서 기반으로 찾아주는 챗봇 시스템을 목표로 하였습니다.

### 🔗 문제의 원인

- **정보의 파편화** : 정책이 사이트별로 분산되어 탐색 비용이 큽니다.
- **비공식 경로 의존** : 청년 10명 중 5명(47%)가 지인·SNS 등에서 정보를 얻어 **정확성 저하 위험**이 존재합니다.
- **복잡한 공고문** : 긴 문서와 행정 용어로 이해 난이도가 높습니다.

### 💡 필요성

사용자 조건(나이·지역·소득 등)에 따라 정책을 **선별·요약·추천**하여 탐색 부담을 줄이고 수혜 기회를 넓히는 환경이 필요하며,<br>
정보 접근성을 높히고 합리적인 선택을 할 수 있도록 지원합니다.


### 👉 Solution
정책 정보를 통합하고 사용자 조건(나이·지역·소득) 기반으로
**맞춤형 정책을 선별·추천해주는 RAG 기반 청년·1인 가구 생활복지/지원 챗봇**을 개발했습니다.


### 기대효과
- **탐색 비용 절감** : 분산된 정책 정보를 한 곳에서 조회·비교
- **맞춤 매칭 정확도 향상** : 조건 기반으로 '수혜 가능 정책' 중심 추천
- **공고문 이해 부담 완화** : 공고문 핵심 요약으로 정보 피로도 감소

---

# 2. ⚙️ 기술적 의사결정 (Technical Decision)

**"단순 검색(Keyword Search)을 넘어, Advanced RAG를 도입한 이유"**

청년 정책 데이터는 법령·공고문 중심의 비정형 텍스트라 **키워드 매칭만으로는 정확한 탐색에 한계**가 있습니다.
이에 따라 저희는 다음 목표를 중심으로 **고도화된 RAG 파이프라인**을 설계했습니다.

- **환각(Hallucination) 최소화 및 근거 강화** : 외부 정책 데이터를 검색해 근거 기반 답변을 생성
- **정확도(Accuracy) 향상** : Vector Search + BM25 하이브리드와 Re-ranking으로 검색 품질 개선
- **데이터 최신성/실현 가능성(Feasibility)**: ‘온통청년’ Open API 기반으로 3,000여 건 정책을 실시간 수집·통합

---
# 3. 🏗️ 시스템 아키텍처
<img src = "./image/System Architecture.png" alt="system_architecture" width="800"/>

# 4. ✨ 핵심 기술 및 기능 (Key Features)

### 🔍 1. 정확도 높은 검색 (Advanced RAG)
- Vector Retriever와 BM25 Retriever 결합으로 검색 정확도 개선  
- RRF를 활용하여 서로 다른 검색 결과를 재정렬해 상위 문서 품질 향상  
- 짧고 추상적인 질문을 Multi-Query를 이용하여 정책 키워드 관점으로 확장

### 👤 2. 개인화 필터링 (Metadata Filtering)
- 대화에서 나이·지역·상태 정보를 실시간 추출  
- 사용자 자격과 불일치하는 정책을 사전 제거해 **환각(Hallucination) 위험 감소**

### 🧠 3. 지능형 라우팅 (Query Routing)
사용자 의도를 정책 검색/일상 대화/정보 요청 등으로 분기해 **불필요한 API 호출 비용을 최소화**했습니다.

### 4. 📈 트러블 슈팅 (Trouble Shooting)

#### 문제 1: 지역을 바꿔서 질문해도 서울·전국 정책이 섞여서 나옴
- **원인**: 쿼리에서 사용자 지역(user_region) 정보를 받았지만, 실제 검색 필터에서 region 조건을 제대로 안 걸어서 전국 정책 + 특정 지역 정책이 한 번에 섞여 반환됨.  
- **해결**: 벡터 검색 결과에서 metadata["region"] 값을 검사해 사용자 지역과 일치하는 정책만 남기도록 필터 로직 추가함.

#### 문제 2: Retriever 결과가 없을 때 전체 파이프라인이 바로 에러로 종료됨.
- **원인**: 검색 결과 리스트가 비어 있는데도, 이후 단계에서 docs[0]처럼 인덱싱을 시도해서 IndexError 발생.  
- **해결**: 검색 결과가 0개일 경우 “관련 정책을 찾지 못했다”는 안전한 답변을 보내거나 기본 안내 문구를 반환하도록 가드 로직 추가함.

#### 문제 3: 같은 구 이름을 검색했을 때, 다른 시·도의 정책까지 섞여서 나오는 문제
- **원인**: RAG 파이프라인에서 “사용자가 말한 시/도” 와 “문서 메타데이터의 시/도”를 비교·검증하는 단계가 없었음
- **해결**: 사용자가 말한 시/도 기준으로 한 번 더 필터링하는 로직”**을 파이프라인에 추가해서 해결

#### 문제 4: self-query로 구현으로 하려 했으나   
- **원인**: 
- **해결**: class 상속 함수를 따로 정의를 하여 해결

#### 문제 5:  ~~~ 오류발생했음.
- **원인**: 데이터 전처리 과정에서 '등록기관명'등 컬럼 기준으로 잡았었음.?
- **해결**: 데이터전처리 방식을 바꿨더니?

---

# 5. 📊 성능 평가 및 테스트 결과 (Performance)

다양한 테스트를 수행해 기능·성능·안정성을 검증했습니다.  
Hybrid Search 및 Advanced RAG 적용 결과, 정책 검색 정확도가 유의미하게 향상되었습니다.

| 평가지표 | 목표치 | 달성 결과 | 비고 |
|---|---:|---:|---|
| **사용자 만족도** | 4.0/5.0 | **4.4/5.0** | 10명 대상 UAT |
| **시스템 안정성** | 99% | **100%** | 결함 0건 |

---

# 6. 🚀 기대 효과

- 분산된 정책을 질문 한 번으로 조건 맞춤 탐색  
- 반복 문의 자동화로 담당자는 심층 상담에 집중  
- 개인 조건 기반 추천으로 **1:1 맞춤형 안내 경험** 제공  

---

# 7. 💻 실행 화면 (Demo)
(여기에 app.py 실행 스크린샷 또는 GIF 2~3장)

- 메인 화면: 프로필 설정 및 채팅 UI  
- 대화 예시:  
  “서울 사는 취준생인데 월세 지원 있어?” → 관련 정책 추천 + 출처/근거 제시

---

# 8. 🛠️ 기술 스택 (Tech Stack)

| 구분 | 기술 | 상세 |
|---|---|---|
| **Language** | Python | 3.11+ |
| **LLM** | OpenAI | GPT-4o-mini |
| **Embedding** | OpenAI | text-embedding-3-small |
| **Framework** | LangChain | RAG 파이프라인 구성(라우팅, 멀티쿼리, 리랭킹 연동) |
| **Vector DB** | ChromaDB | 정책 문서 임베딩 저장/검색 |
| **Retriever** | BM25 + Vector | 하이브리드 검색 구성 |
| **Re-ranking/Fusion** | RRF | 검색 결과 재정렬로 상위 문서 품질 향상 |
| **Web UI** | Chainlit | 사용자 프로필 입력 + 정책 Q&A 챗 UI |
| **Data Source** | 온통청년 Open API | 전국 단위 청년 정책 실시간 수집·통합 |
| **ETL/Processing** | Pandas | 데이터 정제/구조화 |
| **Environment** | dotenv | API Key/환경변수 관리 |
| **Version Control** | Git/GitHub/Discord | 협업 및 배포 관리 |

---

# 9. 🏃‍♂️ 실행 방법

### 1. 환경 설정

```bash
# Repository Clone
git clone [레포지토리 주소]

# Install Dependencies
pip install -r requirements.txt

# Environment Setup (.env 파일 생성)
# OPENAI_API_KEY=sk-...
# YOUTH_POLICY_API=...
```

### 2. 데이터 구축 (최초 1회)
```bash
# 1. API 데이터 수집 (3,550개 정책)
python notebooks/fetch_api_data.py

# 2. 벡터 DB 구축 (임베딩 및 인덱싱)
python notebooks/build_vectordb.py
```
### 3. 어플리케이션 실행
```bash
streamlit run src/streamlit_app.py
```
 --

# 10. Review

| 이름 | 소감 |
|---|---|
| **나호성** | ~ ~ ~ |
| **강민지** | ~ ~ ~ |
| **김지은** | ~ ~ ~ |
| **조준상** | ~ ~ ~ |
| **홍혜원** | ~ ~ ~ |







