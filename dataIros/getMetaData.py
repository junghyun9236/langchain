# -*- coding: utf-8 -*-
"""
data.iros.go.kr Open API(openOapiAppl.do)에 검색어 "" 로 요청한 결과 중
contents.txt에 적힌 항목과 부합하는 데이터만 추출하는 스크립트
"""
import os
import re
import json
import requests
from urllib.parse import quote

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTENTS_FILE = os.path.join(BASE_DIR, "contents.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "matched_metadata.json")

OPEN_API_URL = "https://data.iros.go.kr/rp/oa/selectOapiApplList.do"


def load_contents_titles(filepath: str) -> list[str]:
    """contents.txt에서 검색 대상 제목 목록 로드 (빈 줄 제외, strip)."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def fetch_open_api(search_word: str = "", page_index: str = "1", page_per_count: str = "500") -> list:
    """openOapiAppl.do 에 POST 요청 후 dataList 반환. search_word는 encodeURI 방식으로 인코딩."""
    # encodeURI 방식: UTF-8 퍼센트 인코딩 (JavaScript encodeURI에 해당)
    search_word_encoded = quote(search_word, encoding="utf-8") if search_word else ""
    try:
        # search_word만 인코딩된 값 사용, 나머지는 그대로 (이중 인코딩 방지)
        body = f"pageIndex={page_index}&pagePerCount={page_per_count}&search_word={search_word_encoded}"
        resp = requests.post(
            OPEN_API_URL,
            data=body.encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("dataList", []) if isinstance(data, dict) else []
    except requests.RequestException as e:
        raise RuntimeError(f"API 요청 실패: {e}") from e
    except (json.JSONDecodeError, TypeError) as e:
        raise RuntimeError(f"응답 파싱 실패: {e}") from e


def normalize_for_match(text: str) -> str:
    """매칭용 정규화: 공백·괄호 통일."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[（）]", lambda m: "(" if m.group(0) in "（" else ")", text)
    return text


def title_matches(content_title: str, item: dict) -> bool:
    """
    contents.txt 한 줄(content_title)이 API 항목(item)과 부합하는지 판단.
    item의 제목 필드: rdata_name, title, data_name 등 가능성 있음.
    """
    content_norm = normalize_for_match(content_title)
    if not content_norm:
        return False

    # 응답에서 제목으로 쓸 수 있는 필드 후보
    name_candidates = (
        item.get("rdata_name")
        or item.get("title")
        or item.get("data_name")
        or item.get("name")
        or item.get("rdata_nm")
        or ""
    )
    if isinstance(name_candidates, list):
        name_candidates = name_candidates[0] if name_candidates else ""
    name_norm = normalize_for_match(str(name_candidates))

    # 설명 필드도 부분 매칭에 사용 (선택)
    desc = (
        item.get("rdata_dsc")
        or item.get("description")
        or item.get("desc")
        or ""
    )
    desc_norm = normalize_for_match(str(desc))

    # 완전 일치 또는 포함 관계
    if content_norm == name_norm:
        return True
    if content_norm in name_norm or name_norm in content_norm:
        return True
    if content_norm in desc_norm:
        return True
    return False


def extract_matching_data(api_items: list, content_titles: list[str]) -> list:
    """API 전체 결과 중 contents.txt 제목과 부합하는 항목만 추출."""
    matched = []
    for item in api_items:
        for title in content_titles:
            if title_matches(title, item):
                # 중복 제거: 동일 item이 여러 title에 매칭될 수 있음
                if item not in matched:
                    matched.append(item)
                break
    return matched


def main():
    content_titles = load_contents_titles(CONTENTS_FILE)
    if not content_titles:
        print("contents.txt가 비어 있거나 없습니다.")
        return

    print(f"contents.txt 기준 제목 {len(content_titles)}개 로드.")

    all_items = []
    page = 1
    while True:
        chunk = fetch_open_api(search_word="소유권이전등기", page_index=str(page))
        if not chunk:
            break
        all_items.extend(chunk)
        if len(chunk) < int("500"):
            break
        page += 1

    print(f"API 응답 총 {len(all_items)}건 수신.")

    matched = extract_matching_data(all_items, content_titles)
    print(f"contents.txt와 부합하는 데이터: {len(matched)}건.")

    # 결과 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(matched, f, ensure_ascii=False, indent=2)

    print(f"결과 저장: {OUTPUT_FILE}")

    # 콘솔에 매칭된 제목만 간단 출력 (필드명은 API 응답에 맞게 조정 가능)
    for i, item in enumerate(matched, 1):
        name = (
            item.get("rdata_name")
            or item.get("title")
            or item.get("data_name")
            or item.get("name")
            or item.get("rdata_nm")
            or "(제목없음)"
        )
        print(f"  {i}. {name}")


if __name__ == "__main__":
    main()
