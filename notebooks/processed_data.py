import json
from pathlib import Path
import os
import pandas as pd

# 파일 경로 설정 (상대 경로)
current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
project_root = current_dir.parent
IN_PATH = project_root / "data" / "raw" / "youth_policies_api.json"
OUT_PATH = project_root / "data" / "processed" / "youth_policies_filtered_kr_revised.json"
ZIP_CODE_PATH = project_root / "data" / "processed" / "법정동코드 수정.txt"


# 법정동코드 매핑 딕셔너리 로드
def load_zip_code_mapping():
    """법정동코드 -> 한글 지역명 매핑 딕셔너리 생성"""
    df = pd.read_csv(ZIP_CODE_PATH, sep='\t', dtype=str, encoding='cp949')
    # 딕셔너리로 변환 {코드: 지역명}
    return dict(zip(df['법정동코드'], df['법정동명']))


# 코드 -> 한글 매핑 딕셔너리들
PVSN_INST_GROUP = {
    "0054001": "중앙부처",
    "0054002": "지자체"
}

PLCY_PVSN_METHOD = {
    "0042001": "인프라 구축",
    "0042002": "프로그램",
    "0042003": "직접대출",
    "0042004": "공공기관",
    "0042005": "계약(위탁운영)",
    "0042006": "보조금",
    "0042007": "대출보증",
    "0042008": "공적보험",
    "0042009": "조세지출",
    "0042010": "바우처",
    "0042011": "정보제공",
    "0042012": "경제적 규제",
    "0042013": "기타"
}

PLCY_APPROVAL_STATUS = {
    "0044001": "신청",
    "0044002": "승인",
    "0044003": "반려",
    "0044004": "임시저장"
}

APPLY_PERIOD_TYPE = {
    "0057001": "특정기간",
    "0057002": "상시",
    "0057003": "마감"
}

BIZ_PERIOD_TYPE = {
    "0056001": "특정기간",
    "0056002": "상시",
    "0056003": "마감"
}

MARRIAGE_STATUS = {
    "0055001": "기혼",
    "0055002": "미혼",
    "0055003": "제한없음"
}

INCOME_CONDITION = {
    "0043001": "무관",
    "0043002": "연소득",
    "0043003": "기타"
}

MAJOR_REQUIREMENT = {
    "0011001": "인문계열",
    "0011002": "사회계열",
    "0011003": "상경계열",
    "0011004": "이학계열",
    "0011005": "공학계열",
    "0011006": "예체능계열",
    "0011007": "농산업계열",
    "0011008": "기타",
    "0011009": "제한없음"
}

JOB_REQUIREMENT = {
    "0013001": "재직자",
    "0013002": "자영업자",
    "0013003": "미취업자",
    "0013004": "프리랜서",
    "0013005": "일용근로자",
    "0013006": "(예비)창업자",
    "0013007": "단기근로자",
    "0013008": "영농종사자",
    "0013009": "기타",
    "0013010": "제한없음"
}

SCHOOL_REQUIREMENT = {
    "0049001": "고졸 미만",
    "0049002": "고교 재학",
    "0049003": "고졸 예정",
    "0049004": "고교 졸업",
    "0049005": "대학 재학",
    "0049006": "대졸 예정",
    "0049007": "대학 졸업",
    "0049008": "석·박사",
    "0049009": "기타",
    "0049010": "제한없음"
}

SPECIAL_REQUIREMENT = {
    "0014001": "중소기업",
    "0014002": "여성",
    "0014003": "기초생활수급자",
    "0014004": "한부모가정",
    "0014005": "장애인",
    "0014006": "농업인",
    "0014007": "군인",
    "0014008": "지역인재",
    "0014009": "기타",
    "0014010": "제한없음"
}


# 원본 필드명 -> 전처리 결과(한글) 필드명 매핑
FIELD_MAP = {
    # 기본 정보
    "plcyNm": "정책명",
    "plcyKywdNm": "정책키워드",
    "plcyExplnCn": "정책설명",
    "lclsfNm": "대분류",
    "mclsfNm": "중분류",

    # 지원 내용·조건
    "plcySprtCn": "지원내용",
    "earnMinAmt": "최소지원금액",
    "earnMaxAmt": "최대지원금액",
    "earnEtcCn": "기타지원조건",
    "sprtTrgtMinAge": "지원최소연령",
    "sprtTrgtMaxAge": "지원최대연령",

    # 기관 정보
    "sprvsnInstCdNm": "주관기관명",
    "operInstCdNm": "운영기관명",
    "rgtrInstCdNm": "등록기관명",
    "rgtrUpInstCdNm": "상위기관명",
    "rgtrHghrkInstCdNm": "상위등록기관명",

    # 신청·사업 기간/방법
    "aplyYmd": "신청기간",
    "plcyAplyMthdCn": "신청방법",
    "sbmsnDcmntCn": "제출서류",
    "bizPrdBgngYmd": "사업시작일",
    "bizPrdEndYmd": "사업종료일",

    # 심사·선정
    "srngMthdCn": "선정방법",

    # URL
    "aplyUrlAddr": "신청URL",
    "refUrlAddr1": "참고URL1",
    "refUrlAddr2": "참고URL2",

    # 기타 텍스트 조건
    "addAplyQlfcCndCn": "추가자격조건",
    "ptcpPrpTrgtCn": "참여제외대상",
}


def decode_multiple_codes(code_string: str, code_map: dict) -> str:
    """
    쉼표로 구분된 여러 코드를 한글로 변환
    예: "0013001,0013002,0013004" -> "재직자, 자영업자, 프리랜서"
    """
    if not code_string:
        return ""
    
    # 쉼표로 분리
    codes = [c.strip() for c in code_string.split(",")]
    
    # 각 코드를 한글로 변환
    korean_values = []
    for code in codes:
        if code in code_map:
            korean_values.append(code_map[code])
        else:
            korean_values.append(code)  # 매핑 없으면 원본 유지
    
    # 쉼표로 연결하여 반환
    return ", ".join(korean_values)


def decode_zip_codes(zip_code_string: str, zip_code_map: dict) -> str:
    """
    쉼표로 구분된 법정동코드를 한글 지역명으로 변환
    예: "11110,11140,11170" -> "서울특별시 종로구, 서울특별시 중구, 서울특별시 용산구"
    """
    if not zip_code_string:
        return ""
    
    # 쉼표로 분리
    codes = [c.strip() for c in zip_code_string.split(",")]
    
    # 각 코드를 한글로 변환
    korean_regions = []
    for code in codes:
        if code in zip_code_map:
            korean_regions.append(zip_code_map[code])
        else:
            korean_regions.append(code)  # 매핑 없으면 원본 유지
    
    # 쉼표로 연결하여 반환
    return ", ".join(korean_regions)


def decode_code_to_korean(src: dict, zip_code_map: dict) -> dict:
    """
    코드 값을 한글로 변환 (코드 필드는 저장하지 않고 한글 필드만 반환)
    여러 코드가 쉼표로 구분되어 있는 경우도 처리
    """
    decoded = {}
    
    # 재공기관그룹
    if src.get("pvsnInstGroupCd"):
        decoded["재공기관그룹"] = decode_multiple_codes(src["pvsnInstGroupCd"], PVSN_INST_GROUP)
    
    # 정책제공방법
    if src.get("plcyPvsnMthdCd"):
        decoded["정책제공방법"] = decode_multiple_codes(src["plcyPvsnMthdCd"], PLCY_PVSN_METHOD)
    
    # 정책승인상태
    if src.get("plcyAprvSttsCd"):
        decoded["정책승인상태"] = decode_multiple_codes(src["plcyAprvSttsCd"], PLCY_APPROVAL_STATUS)
    
    # 신청기간구분
    if src.get("aplyPrdSeCd"):
        decoded["신청기간구분"] = decode_multiple_codes(src["aplyPrdSeCd"], APPLY_PERIOD_TYPE)
    
    # 사업기간구분
    if src.get("bizPrdSeCd"):
        decoded["사업기간구분"] = decode_multiple_codes(src["bizPrdSeCd"], BIZ_PERIOD_TYPE)
    
    # 혼인상태
    if src.get("mrgSttsCd"):
        decoded["혼인상태"] = decode_multiple_codes(src["mrgSttsCd"], MARRIAGE_STATUS)
    
    # 소득조건
    if src.get("earnCndSeCd"):
        decoded["소득조건"] = decode_multiple_codes(src["earnCndSeCd"], INCOME_CONDITION)
    
    # 전공요건
    if src.get("plcyMajorCd"):
        decoded["전공요건"] = decode_multiple_codes(src["plcyMajorCd"], MAJOR_REQUIREMENT)
    
    # 취업요건
    if src.get("jobCd"):
        decoded["취업상태"] = decode_multiple_codes(src["jobCd"], JOB_REQUIREMENT)
    
    # 학력요건
    if src.get("schoolCd"):
        decoded["학력요건"] = decode_multiple_codes(src["schoolCd"], SCHOOL_REQUIREMENT)
    
    # 특화요건
    if src.get("sbizCd"):
        decoded["특화분야"] = decode_multiple_codes(src["sbizCd"], SPECIAL_REQUIREMENT)
    
    # 법정동코드 (zipCd) -> 지역명
    if src.get("zipCd"):
        decoded["지역"] = decode_zip_codes(src["zipCd"], zip_code_map)
    
    return decoded


def transform_record(src: dict, zip_code_map: dict) -> dict:
    """
    원본 한 건(src)을 전처리 형식(dict)으로 변환.
    - FIELD_MAP에 정의된 항목만 남기고,
    - 값이 비어 있으면 생략
    - 코드 필드는 한글로 변환 후 코드는 삭제
    """
    dst = {}

    for src_key, dst_key in FIELD_MAP.items():
        value = src.get(src_key)
        if value is None:
            continue

        if isinstance(value, str) and value.strip() == "":
            continue

        dst[dst_key] = value
    
    # 코드를 한글로 변환한 필드 추가 (코드 필드는 포함하지 않음)
    decoded_fields = decode_code_to_korean(src, zip_code_map)
    dst.update(decoded_fields)

    return dst

# 지역범위(전국/지역) 추가 부분 
import re

# 시도 목록
SIDO_LIST = [
    "서울특별시","부산광역시","대구광역시","인천광역시","광주광역시",
    "대전광역시","울산광역시","세종특별자치시","경기도","강원특별자치도",
    "충청북도","충청남도","전북특별자치도","전라남도","경상북도",
    "경상남도","제주특별자치도"
]

# 시도 추출용 패턴
SIDO_PATTERN = "(" + "|".join(SIDO_LIST) + ")"


def detect_region_level(region_field: str) -> str:
    """문자열에서 시도 등장 개수를 세어 nationwide 정책 여부 판단"""
    if not region_field or not isinstance(region_field, str):
        return "지역"

    # 시도 추출
    found_sido = re.findall(SIDO_PATTERN, region_field)

    unique_sido_count = len(set(found_sido))

    # 전국 판별 기준: 17개 시도 모두 등장해야 전국
    if unique_sido_count == len(SIDO_LIST):  # == 17
        return "전국"

    return "지역"


def assign_region_level(policy: dict) -> str:
    """정책 딕셔너리에서 '지역' 필드를 기반으로 region_level 산정"""
    region_text = policy.get("지역", "")

    # ① 지역 정보가 비었거나 '전국' 형태 표시인 경우
    if not region_text or region_text.strip() in ["전국", "전국권", "전체"]:
        return "전국"

    # ② 문자열 기반 전국 판정 로직 수행
    return detect_region_level(region_text)


def main():
    # 0) 법정동코드 매핑 로드
    zip_code_map = load_zip_code_mapping()
    
    # 1) 원본 JSON 로드
    with IN_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # 2) 정책 리스트 꺼내기 (result > youthPolicyList)
    policies = raw["result"]["youthPolicyList"]

    # 3) 각 정책을 전처리 포맷으로 변환 (한글 "지역" 필드 생성)
    transformed = []
    for p in policies:
        rec = transform_record(p, zip_code_map)
        transformed.append(rec)

    # 4) 전처리된 데이터에 "지역범위" 필드 추가
    for rec in transformed:
        region_level = assign_region_level(rec)
        rec["지역범위"] = region_level

    # 5) 리스트 형태로 저장
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()