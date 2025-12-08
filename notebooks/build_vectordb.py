"""
ChromaDB 벡터 데이터베이스 구축
전처리된 JSON 데이터를 임베딩하여 벡터 DB에 저장
"""

import json
import os
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)


def load_preprocessed_data(filepath):
    """
    전처리된 JSON 데이터 로드
    
    Args:
        filepath: JSON 파일 경로
        
    Returns:
        list: 정책 데이터 리스트
    """
    print(f"📂 데이터 로드 중: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 총 {len(data)}개의 정책 데이터 로드 완료")
    return data


def create_policy_text(policy):
    """
    정책 데이터를 임베딩을 위한 텍스트로 변환
    중요도 순으로 배치하여 검색 품질 향상
    
    Args:
        policy: 정책 딕셔너리
        
    Returns:
        str: 결합된 텍스트
    """
    # 주요 필드들을 결합하여 검색 가능한 텍스트 생성 (중요도 순)
    text_parts = []
    
    # 1. 가장 중요: 정책명과 분야
    if policy.get('정책명'):
        text_parts.append(f"정책명: {policy['정책명']}")
    
    if policy.get('대분류'):
        text_parts.append(f"대분류: {policy['대분류']}")
    
    if policy.get('중분류'):
        text_parts.append(f"중분류: {policy['중분류']}")
    
    # 2. 정책 설명 (핵심 내용)
    if policy.get('정책설명'):
        text_parts.append(f"정책설명: {policy['정책설명']}")
    
    # 3. 지원내용 (길이 제한 없음 - 중요한 정보)
    if policy.get('지원내용'):
        text_parts.append(f"지원내용: {policy['지원내용']}")
    
    if policy.get('정책키워드'):
        text_parts.append(f"키워드: {policy['정책키워드']}")
    
    # 4. 지역 정보 (검색 정확도 향상)
    if policy.get('지역'):
        # 지역 정보가 길 수 있으므로 적절히 포함
        region = policy['지역']
        # 쉼표로 구분된 지역을 간단히 처리
        if len(region) > 500:
            # 너무 길면 앞부분만 (전국 정책일 가능성)
            region = region[:500] + "..."
        text_parts.append(f"적용지역: {region}")
    
    # 5. 자격 조건 (상세)
    if policy.get('추가자격조건'):
        # 길이 제한 확대 (500자)
        qual = policy['추가자격조건'][:500]
        text_parts.append(f"자격조건: {qual}")
    
    # 6. 자격 요건 (한글 변환된 필드)
    if policy.get('취업상태'):
        job_status = policy['취업상태']
        # 자연어 표현 추가 (검색 향상)
        natural_terms = []
        if '미취업자' in job_status:
            natural_terms.append('실업자, 구직자, 백수, 취업준비생')
        if '재직자' in job_status:
            natural_terms.append('직장인, 근로자')
        if '창업자' in job_status or '예비' in job_status:
            natural_terms.append('창업준비, 사업자')
        
        full_status = f"취업상태: {job_status}"
        if natural_terms:
            full_status += f" ({', '.join(natural_terms)})"
        text_parts.append(full_status)
    
    if policy.get('학력요건'):
        edu_req = policy['학력요건']
        # 자연어 표현 추가
        natural_edu = []
        if '고졸' in edu_req or '고교' in edu_req:
            natural_edu.append('고등학교')
        if '대학' in edu_req or '대졸' in edu_req:
            natural_edu.append('대학교, 학사')
        if '석박사' in edu_req:
            natural_edu.append('대학원')
        
        full_edu = f"학력: {edu_req}"
        if natural_edu:
            full_edu += f" ({', '.join(natural_edu)})"
        text_parts.append(full_edu)
    
    if policy.get('전공요건'):
        major_req = policy['전공요건']
        # 자연어 표현 추가
        natural_major = []
        if '공학' in major_req:
            natural_major.append('이공계, 기술, IT, 공대')
        if '상경' in major_req:
            natural_major.append('경영, 경제, 회계')
        if '예체능' in major_req:
            natural_major.append('예술, 체육, 음악, 미술')
        
        full_major = f"전공: {major_req}"
        if natural_major:
            full_major += f" ({', '.join(natural_major)})"
        text_parts.append(full_major)
    
    if policy.get('특화분야'):
        special = policy['특화분야']
        # 자연어 표현 추가
        natural_special = []
        if '여성' in special:
            natural_special.append('여성청년, 경력단절여성')
        if '장애인' in special:
            natural_special.append('장애청년')
        if '한부모' in special:
            natural_special.append('싱글맘, 싱글대디, 미혼모, 미혼부')
        if '기초생활수급자' in special:
            natural_special.append('저소득층, 차상위계층')
        if '중소기업' in special:
            natural_special.append('중견기업, 스타트업')
        
        full_special = f"특화분야: {special}"
        if natural_special:
            full_special += f" ({', '.join(natural_special)})"
        text_parts.append(full_special)
    
    if policy.get('정책제공방법'):
        text_parts.append(f"제공방법: {policy['정책제공방법']}")
    
    if policy.get('소득조건'):
        income = policy['소득조건']
        # 자연어 표현 추가
        natural_income = []
        if '무관' in income:
            natural_income.append('소득제한없음, 누구나')
        if '연소득' in income:
            natural_income.append('소득기준, 소득제한')
        
        full_income = f"소득조건: {income}"
        if natural_income:
            full_income += f" ({', '.join(natural_income)})"
        text_parts.append(full_income)
    
    if policy.get('혼인상태'):
        text_parts.append(f"혼인상태: {policy['혼인상태']}")
    
    # 7. 연령 제한
    min_age = policy.get('지원최소연령', '0')
    max_age = policy.get('지원최대연령', '0')
    if min_age != '0' or max_age != '0':
        age_info = f"대상연령: {min_age}세 ~ {max_age}세"
        text_parts.append(age_info)
    
    # 8. 지원금액
    min_amount = policy.get('최소지원금액', '0')
    max_amount = policy.get('최대지원금액', '0')
    if min_amount != '0' or max_amount != '0':
        amount_info = f"지원금액: {min_amount}원 ~ {max_amount}원"
        text_parts.append(amount_info)
    
    if policy.get('기타지원조건'):
        other_cond = policy['기타지원조건'][:200]
        text_parts.append(f"기타조건: {other_cond}")
    
    # 9. 신청 및 선정 방법
    if policy.get('신청기간구분'):
        text_parts.append(f"신청기간: {policy['신청기간구분']}")
    
    if policy.get('사업기간구분'):
        text_parts.append(f"사업기간: {policy['사업기간구분']}")
    
    if policy.get('신청방법'):
        method = policy['신청방법'][:200]  # 너무 길면 제한
        text_parts.append(f"신청방법: {method}")
    
    if policy.get('선정방법'):
        selection = policy['선정방법'][:200]
        text_parts.append(f"선정방법: {selection}")
    
    if policy.get('제출서류'):
        docs = policy['제출서류'][:200]
        text_parts.append(f"제출서류: {docs}")
    
    # 10. 참여제외대상 (중요 정보)
    if policy.get('참여제외대상'):
        exclusion = policy['참여제외대상'][:300]
        text_parts.append(f"제외대상: {exclusion}")
    
    # 텍스트가 비어있으면 최소한의 정보라도 포함
    if not text_parts:
        text_parts.append(f"청년정책")
    
    # 텍스트 결합 및 정제
    full_text = "\n".join(text_parts)
    
    # 중복 공백 제거 및 정리
    full_text = " ".join(full_text.split())
    
    return full_text


def get_embeddings_batch(texts, model="text-embedding-3-small"):
    """
    OpenAI API를 사용하여 여러 텍스트의 임베딩을 배치로 생성 (속도 향상)
    
    Args:
        texts: 임베딩할 텍스트 리스트
        model: 사용할 임베딩 모델
        
    Returns:
        list: 임베딩 벡터 리스트
    """
    # 텍스트 정제
    cleaned_texts = []
    for text in texts:
        text = text.replace("\n", " ").strip()
        if not text or len(text) < 3:
            text = "정책 정보"
        if len(text) > 8000:
            text = text[:8000]
        cleaned_texts.append(text)
    
    # API 호출 재시도 로직
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=cleaned_texts, model=model)
            return [item.embedding for item in response.data]
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                wait_time = (attempt + 1) * 2
                print(f"  ⚠️  배치 API 오류, {wait_time}초 후 재시도... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e


def get_embedding(text, model="text-embedding-3-small"):
    """
    OpenAI API를 사용하여 텍스트 임베딩 생성
    
    Args:
        text: 임베딩할 텍스트
        model: 사용할 임베딩 모델
               - text-embedding-3-small (1536차원, 빠름, 저렴)
               - text-embedding-3-large (3072차원, 느림, 고품질, 고비용)
        
    Returns:
        list: 임베딩 벡터
    """
    # 텍스트 정제
    text = text.replace("\n", " ").strip()
    
    # 빈 텍스트 체크
    if not text or len(text) < 3:
        text = "정책 정보"
    
    # 너무 긴 텍스트는 잘라내기
    # text-embedding-3-small: 최대 8191 토큰 (~32,000자)
    # 안전을 위해 8000자로 제한
    if len(text) > 8000:
        text = text[:8000]
    
    # API 호출 재시도 로직 (Rate limit 대응)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                wait_time = (attempt + 1) * 2  # 2, 4, 6초 대기
                print(f"  ⚠️  API 오류, {wait_time}초 후 재시도... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e


def build_chromadb(policies, db_path="../data/vectordb"):
    """
    ChromaDB 벡터 데이터베이스 구축
    
    Args:
        policies: 정책 데이터 리스트
        db_path: DB 저장 경로
    """
    print("\n" + "=" * 70)
    print("🔨 ChromaDB 구축 시작")
    print("=" * 70)
    
    # DB 디렉토리 생성
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_full_path = os.path.join(project_root, "data", "vectordb")
    os.makedirs(db_full_path, exist_ok=True)
    
    print(f"📁 DB 저장 경로: {db_full_path}")
    
    # ChromaDB 클라이언트 초기화
    chroma_client = chromadb.PersistentClient(path=db_full_path)
    
    # 기존 컬렉션 삭제 (있으면)
    try:
        chroma_client.delete_collection(name="youth_policies")
        print("🗑️  기존 컬렉션 삭제")
    except:
        pass
    
    # 물리적 파일 정리 (세그먼트 폴더 삭제)
    import shutil
    for item in os.listdir(db_full_path):
        item_path = os.path.join(db_full_path, item)
        # UUID 형식의 폴더만 삭제 (chroma.sqlite3는 유지)
        if os.path.isdir(item_path) and '-' in item:
            try:
                shutil.rmtree(item_path)
                print(f"🗑️  세그먼트 폴더 삭제: {item}")
            except:
                pass
    
    # 새 컬렉션 생성
    collection = chroma_client.create_collection(
        name="youth_policies",
        metadata={"description": "온통청년 정책 데이터"}
    )
    
    print(f"\n📊 총 {len(policies)}개 정책 처리 중...")
    
    # 배치 임베딩 처리를 위한 변수
    embedding_batch_size = 20  # OpenAI API 배치 제한
    db_batch_size = 50  # DB 저장 배치
    
    all_policy_texts = []
    all_metadatas = []
    all_ids = []
    
    failed_count = 0
    
    # 1단계: 모든 정책 텍스트 생성
    print("  📝 정책 텍스트 생성 중...")
    for idx, policy in enumerate(policies, 1):
        try:
            policy_text = create_policy_text(policy)
            all_policy_texts.append(policy_text)
            
            all_metadatas.append({
                '정책명': policy.get('정책명', ''),
                '대분류': policy.get('대분류', ''),
                '중분류': policy.get('중분류', ''),
                '주관기관명': policy.get('주관기관명', ''),
                '운영기관명': policy.get('운영기관명', ''),
                '등록기관명': policy.get('등록기관명', ''),
                '상위기관명': policy.get('상위기관명', ''),
                '상위등록기관명': policy.get('상위등록기관명', ''),
                '신청URL': policy.get('신청URL', ''),
                '정책키워드': policy.get('정책키워드', ''),
                # 신청 관련
                '신청기간': policy.get('신청기간', ''),
                '신청방법': policy.get('신청방법', ''),
                '제출서류': policy.get('제출서류', ''),
                # 사업 기간
                '사업시작일': policy.get('사업시작일', ''),
                '사업종료일': policy.get('사업종료일', ''),
                # 심사·선정
                '선정방법': policy.get('선정방법', ''),
                # 자격 관련
                '추가자격조건': policy.get('추가자격조건', ''),
                '참여제외대상': policy.get('참여제외대상', ''),
                '지원최소연령': policy.get('지원최소연령', '0'),
                '지원최대연령': policy.get('지원최대연령', '0'),
                # 지원금 관련
                '최소지원금액': policy.get('최소지원금액', '0'),
                '최대지원금액': policy.get('최대지원금액', '0'),
                '기타지원조건': policy.get('기타지원조건', ''),
                # 한글 변환된 필드들
                '재공기관그룹': policy.get('재공기관그룹', ''),
                '정책제공방법': policy.get('정책제공방법', ''),
                '정책승인상태': policy.get('정책승인상태', ''),
                '신청기간구분': policy.get('신청기간구분', ''),
                '사업기간구분': policy.get('사업기간구분', ''),
                '혼인상태': policy.get('혼인상태', ''),
                '소득조건': policy.get('소득조건', ''),
                '전공요건': policy.get('전공요건', ''),
                '취업상태': policy.get('취업상태', ''),
                '학력요건': policy.get('학력요건', ''),
                '특화분야': policy.get('특화분야', ''),
                '지역': policy.get('지역', ''),
            })
            all_ids.append(f"policy_{idx}")
            
            if idx % 100 == 0:
                print(f"    {idx}/{len(policies)} ({idx/len(policies)*100:.1f}%)")
                
        except Exception as e:
            print(f"  ⚠️  정책 {idx} 텍스트 생성 오류: {e}")
            failed_count += 1
            all_policy_texts.append("청년정책")
            all_metadatas.append({})
            all_ids.append(f"policy_{idx}")
    
    # 2단계: 배치 임베딩 생성
    print(f"\n  🔮 임베딩 생성 중 (배치 크기: {embedding_batch_size})...")
    all_embeddings = []
    
    for i in range(0, len(all_policy_texts), embedding_batch_size):
        batch_texts = all_policy_texts[i:i+embedding_batch_size]
        try:
            batch_embeddings = get_embeddings_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            print(f"    임베딩: {min(i+embedding_batch_size, len(all_policy_texts))}/{len(all_policy_texts)} ({min(i+embedding_batch_size, len(all_policy_texts))/len(all_policy_texts)*100:.1f}%)")
        except Exception as e:
            print(f"  ⚠️  배치 {i//embedding_batch_size + 1} 임베딩 오류: {e}")
            # 폴백: 개별 임베딩
            for text in batch_texts:
                try:
                    emb = get_embedding(text)
                    all_embeddings.append(emb)
                except:
                    all_embeddings.append([0] * 1536)  # 빈 벡터
                    failed_count += 1
    
    # 3단계: DB에 배치 저장
    print(f"\n  💾 ChromaDB 저장 중 (배치 크기: {db_batch_size})...")
    for i in range(0, len(all_policy_texts), db_batch_size):
        batch_docs = all_policy_texts[i:i+db_batch_size]
        batch_metas = all_metadatas[i:i+db_batch_size]
        batch_ids = all_ids[i:i+db_batch_size]
        batch_embs = all_embeddings[i:i+db_batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids,
            embeddings=batch_embs
        )
        print(f"    저장: {min(i+db_batch_size, len(all_policy_texts))}/{len(all_policy_texts)} ({min(i+db_batch_size, len(all_policy_texts))/len(all_policy_texts)*100:.1f}%)")
    
    # 텍스트 길이 통계
    all_docs = collection.get()
    if all_docs['documents']:
        avg_text_length = sum([len(d) for d in all_docs['documents']]) / len(all_docs['documents'])
        max_text_length = max([len(d) for d in all_docs['documents']])
        min_text_length = min([len(d) for d in all_docs['documents']])
    else:
        avg_text_length = max_text_length = min_text_length = 0
    
    print("\n" + "=" * 70)
    print("✅ ChromaDB 구축 완료!")
    print("=" * 70)
    print(f"📍 저장 위치: {db_full_path}")
    print(f"📊 총 저장된 정책 수: {collection.count()}")
    if failed_count > 0:
        print(f"⚠️  처리 실패: {failed_count}건")
    print(f"✅ 성공률: {(len(policies) - failed_count) / len(policies) * 100:.1f}%")
    print(f"📝 텍스트 길이 - 평균: {avg_text_length:.0f}자, 최대: {max_text_length}자, 최소: {min_text_length}자")
    
    return collection


# def test_search(collection, query="취업 지원 정책", top_k=3):
#     """
#     벡터 DB 검색 테스트
    
#     Args:
#         collection: ChromaDB 컬렉션
#         query: 검색 쿼리
#         top_k: 반환할 결과 수
#     """
#     print("\n" + "=" * 70)
#     print("🔍 검색 테스트")
#     print("=" * 70)
#     print(f"질문: {query}\n")
    
#     # 쿼리 임베딩 생성
#     query_embedding = get_embedding(query)
    
#     # 유사 문서 검색
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k
#     )
    
#     print(f"상위 {top_k}개 검색 결과:\n")
    
#     for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
#         print(f"[{i}] {metadata.get('정책명', 'N/A')}")
#         print(f"    분야: {metadata.get('중분류', 'N/A')}")
#         print(f"    담당: {metadata.get('주관기관명', 'N/A')}")
#         print(f"    연령: {metadata.get('지원최소연령', '0')}세 ~ {metadata.get('지원최대연령', '0')}세")
#         print(f"    취업: {metadata.get('취업상태', 'N/A')}")
#         print(f"    내용: {doc[:150]}...")
#         print()


def main():
    print("=" * 70)
    print("ChromaDB 벡터 데이터베이스 구축")
    print("=" * 70)
    
    # API 키 확인
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY를 추가해주세요.")
        return
    
    print(f"✅ OpenAI API 키 설정 완료")
    
    # 전처리된 데이터 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "processed", "youth_policies_filtered_kr_revised.json")
    
    if not os.path.exists(data_path):
        print(f"❌ 전처리된 데이터를 찾을 수 없습니다: {data_path}")
        return
    
    policies = load_preprocessed_data(data_path)
    
    # 샘플로 일부만 처리 (테스트용)
    # policies = policies[:50]  # 처음 50개만 테스트
    # 전체 데이터 사용
    print(f"⚠️  전체 {len(policies)}개 정책 처리 - 시간이 걸릴 수 있습니다.")
    
    # ChromaDB 구축
    collection = build_chromadb(policies)
    
    # 검색 테스트
#     test_search(collection, "취업 지원 프로그램이 있나요?")
#     test_search(collection, "창업 관련 정책을 알려주세요")
    
    print("\n✅ 모든 작업 완료!")


if __name__ == "__main__":
    main()
