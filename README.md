# RAG 기반 청년 맞춤형 복지 정책 챗봇🌟
"복잡한 정책, 선배가 간편하고 쉽게 알려줄게!"


🏆 [SKN Family AI캠프] 3차 단위 프로젝트   
📅 개발 기간: 2025.12.10 ~ 2025.12.11

## 팀명 : 청 바 지 (청춘은 바로 지금)

## 팀원 소개

| 나호성 | 강민지 | 이지은 | 조준상 | 홍혜원 |
|---|---|---|---|---|
| <img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/aa1c4b5b-f271-44bc-8765-fb0717a255fb" /> | <img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/93140ba3-81e2-4079-8084-8ebab3121a26" /> | <img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/0ac142c1-01de-4130-bcdf-cf7bf026a371" /> | <img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/4f23001a-38bb-41bb-ae6f-0ee01d97eebf" /> | <img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/b133989b-a180-44cc-8fd6-d593e37aee8f" /> |
| 팀장(PM) | 프롬프트엔지니어링 | 프로젝트PD | 프론트엔드 | 슈퍼프로그래머 |


## 1. 프로젝트 개요 및 목표

#### 문제 정의 및 현황

- **정보의 파편화** : 정책이 사이트별로 분산되어 탐색 비용이 크다.
- **비공식 경로 의존** : 청년 10명 중 5명(47%)가 지인·SNS 등에서 정보를 얻어 **정확성 저하 위험**이 존재한다.
- **어려운 공고문** : 긴 문서와 행정 용어로 이해하기 쉽지 않다.

### 프로젝트명 : 청년이음 
청년이음은 '온통청년' API 데이터를 기반으로 구축된 LLM이다. 
방대한 청년 정책 정보 속에서 사용자의 상황(나이, 거주지, 소득 등)에 딱 맞는 혜택을 찾아주고, 복잡한 공고문을 이해하기 쉬운 언어로 설명한다.

### 목표 
- 환각(Hallucination) 최소화: RAG 기술을 적용하여 근거(Document)에 기반한 정확한 답변 제공
- 사용자 개인맞춤 추천: 사용자의 지역과 키워드를 기준으로 메타데이터로 필터링해 제공 
- 사용자 친화적 경험: 딱딱한 행정 용어 대신, 친근한 선배봇 페르소나 부여 

---

## 2. 기술적 의사결정

**"단순 검색을 넘어, Advanced RAG를 도입한 이유"**

청년 정책 데이터는 법령·공고문 중심의 비정형 텍스트라 **키워드 매칭만으로는 정확한 탐색에 한계**가 있다.
이에 따라 저희는 다음 목표를 중심으로 **Advanced RAG Pipeline**을 설계했다.

- **환각 최소화 및 근거 강화** : 외부 정책 데이터를 검색해 근거 기반 답변을 생성
- **정확도 향상** : Vector Rertriever + BM25 Retriever로 검색 품질 개선
- **데이터**: ‘온통청년’ Open API 기반으로 3,550+ 건 정책을 실시간 수집·통합

---
## 3. 시스템 아키텍처
<img src = "./image/System Architecture.png" alt="system_architecture" width="800"/>

## 4. 핵심 기술 및 기능

### 4.1. 정확도 높은 검색 (Advanced RAG)
- Vector Retriever와 BM25 Retriever 결합으로 검색 정확도 개선  
- RRF를 활용하여 서로 다른 검색 결과를 재정렬해 상위 문서 품질 향상  
- 짧고 추상적인 질문을 Multi-Query를 이용하여 정책 키워드 관점으로 확장

### 4.2. 개인화 필터링 (Metadata Filtering)
- 대화에서 나이·지역·상태 정보를 실시간 추출  
- 사용자 자격과 불일치하는 정책을 사전 제거해 **환각(Hallucination) 위험 감소**

### 4.3. 지능형 라우팅 (Query Routing)
사용자 의도를 정책 검색/일상 대화/정보 요청 등으로 분기해 **불필요한 API 호출 비용을 최소화**

---

## 5. 트러블 슈팅

#### 문제 1: 지역을 바꿔서 질문해도 서울·전국 정책이 섞여서 나옴
- **원인**: 쿼리에서 `user_region` 정보를 받았지만, 실제 검색 필터에서 region 조건을 제대로 안 걸어서 전국 정책 + 특정 지역 정책이 한 번에 섞여 반환됨
- **해결**: 벡터 검색 결과에서 `metadata["region"]` 값을 검사해 사용자 지역과 일치하는 정책만 남기도록 필터 로직 추가함

#### 문제 2: `Retriever` 결과가 없을 때 전체 파이프라인이 바로 에러로 종료됨
- **원인**: 검색 결과 리스트가 비어 있는데도, 이후 단계에서 docs[0]처럼 인덱싱을 시도해서 IndexError 발생. 
- **해결**: 검색 결과가 0개일 경우 “관련 정책을 찾지 못했다”는 안전한 답변을 보내거나 기본 안내 문구를 반환하도록 가드 로직 추가함

#### 문제 3: 같은 구 이름을 검색했을 때, 다른 시·도의 정책까지 섞여서 나오는 문제
- **원인**: `RAG pipline`에서 “사용자가 말한 시/도” 와 “문서 `metadata`의 시/도”를 비교·검증하는 단계가 없었음
- **해결**: 사용자가 말한 시/도 기준으로 한 번 더 필터링하는 로직을 파이프라인에 추가해서 해결

#### 문제 4: ChromaDB에서 필터링이 되지 않아 반환값이 나오지 못하는 문제
- **원인**: SelfQueryRetriever` 사용 시 ChromaDB의  논리연산자 미지원으로 쿼리 에러 발생
- **해결**: `SelfQueryRetriever`를 비활성화하고,  `RegionFilter` 클래스를 구현하여 로직을 분리하여 해결

#### 문제 5:  대구" 검색 시 벡터 유사도가 높은 "부산 월세 지원"이 상위에 뜨는 문제
- **원인**: 지역을 필터링하지 못함 
- **해결**: `Post-Filtering` 추가 도입 후 타 지역 정책을 제외하고 반환함 

---

## 6. 실행 화면

## UI 화면 기능
#### 💬 대화 흐름이 한눈에 보이는 ‘메신저형 레이아웃’
#### 🔎 입력창이 고정되어 있어 언제든 바로 질문 가능
#### 👩🏻‍💻 감정 이모티콘·톤 조절이 가능한 답변 스타일

- 메인 화면<br>
<img width="1916" height="1026" alt="Image" src="https://github.com/user-attachments/assets/e6ce584b-bb08-4551-a82d-4665a9ae297f" /><br><br>
- 검색 화면<br>
<img width="1915" height="1030" alt="Image" src="https://github.com/user-attachments/assets/b118701b-ff1b-4ddb-bae0-18e8e516f9e0" /><br><br>
- 결과 화면<br>
<img width="1917" height="1028" alt="Image" src="https://github.com/user-attachments/assets/d05c2ec7-9419-464b-9714-6518cb395860" /><br>

---

## 7. 기술 스택

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

# 8. 실행 방법

### 1. 데이터 구축 (최초 1회)
```bash
# 1. API 데이터 수집 (3,550개 정책)
python notebooks/fetch_api_data.py

# 2. 벡터 DB 구축 (임베딩 및 인덱싱)
python notebooks/build_vectordb.py
```
### 2. 어플리케이션 실행
```bash
chainlit run chainlit.py
```
 --

# 9. Review

| 이름 | 소감 |
|---|---|
| **나호성** | AI는 아직 똑똑하지 않다. |
| **강민지** | 전통적인 관점으로 보면 "정책" 이라는 행정문서는 복잡하고 정적이어서 걱정이 많았지만, 팀원들과 함께 지역일관성, 정확성, 사용자친화성을 모두 고려하며 RAG,Router,MultiQuery,RRF 등의 기술을 사용하여 실제 사람들에게 도움이 되는 방향으로 잘 녹여낸 것 같아 뿌듯하다. |
| **이지은** | 3차 프로젝트까지 왔는데 여전히 배운 내용을 실전에 적용하는 건 어렵네요... 특히 이번 프로젝트에서는 데이터 전처리가 정말 중요하다는 걸 몸소 느꼈습니다.모델 구현에 있어 어떤 전처리 데이터값을 가지느냐에 따라 성능이 천차만별로 갈릴 수 있다는 점을 이론적으로 아는 것과 실제 코드를 입력해보고 분석하는 건 차이가 엄청나더라고요. 팀원 분들로부터 많은 걸 배울 수 있어 감사했습니다. 여전히 부족한 저를 잘 도와주셨던 팀원분들 감사드리고 마지막까지 화이팅해봐요..! |
| **조준상** | 정보의 양보다 더 중요한 것은, 필요한 조각을 *정확히 찾아내는 힘'임을 RAG 프로젝트를 통해 배웠습니다. |
| **홍혜원** | 청년이음 프로젝트를 진행하며 단순 튜토리얼을 따라가는 것과 내 힘으로 문제를 정의하고 해결해보는 과정 사이에 얼마나 큰 차이가 있는지를 다시 한 번 실감하였다. 전처리만 몇 번씩 갈아엎으며 정답이 정해져 있지 않은 문제들을 통해 내가 배운 지식을 상황에 맞게 변형하고 적용하는 능력이 얼마나 중요한지 느낄 수 있었다.  |







