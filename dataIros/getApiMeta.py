# -*- coding: utf-8 -*-
"""
matched_metadata.json의 각 rdata에 대해 selectOapiAppl.do를 요청하고,
응답 HTML에서 요청인자값·출력값을 파싱해 저장하는 스크립트.

필요 패키지: pip install requests beautifulsoup4
"""
import os
import re
import json
import time
import requests
from urllib.parse import quote

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("BeautifulSoup4 필요: pip install beautifulsoup4")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# matched_metadata.json (오타 amtched도 시도)
MATCHED_JSON = os.path.join(BASE_DIR, "matched_metadata.json")
MATCHED_JSON_ALT = os.path.join(BASE_DIR, "amtched_metadata.json")
OUTPUT_JSON = os.path.join(BASE_DIR, "api_meta.json")

SELECT_OAPI_URL = "https://data.iros.go.kr/rp/oa/selectOapiAppl.do"


def load_matched_metadata() -> list:
    """matched_metadata.json 또는 amtched_metadata.json 로드."""
    for path in (MATCHED_JSON, MATCHED_JSON_ALT):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
    raise FileNotFoundError(f"'{MATCHED_JSON}' 또는 '{MATCHED_JSON_ALT}' 없음.")


def fetch_detail_html(rdata_seq: str) -> str:
    """selectOapiAppl.do 에 rdata_seq로 POST 후 HTML 본문 반환."""
    payload = {"rdata_seq": rdata_seq}
    body = "&".join(f"{k}={quote(str(v), encoding='utf-8')}" for k, v in payload.items())
    resp = requests.post(
        SELECT_OAPI_URL,
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.text


def parse_request_params(soup) -> list:
    """HTML에서 요청인자(요청인자값) 테이블/리스트 파싱."""
    result = []
    # '요청인자' 또는 '요청인자값' 텍스트를 포함한 요소 찾기
    for label in soup.find_all(string=re.compile(r"요청\s*인자\s*(값)?", re.I)):
        parent = label.parent
        if not parent:
            continue
        # 같은 섹션 내 테이블 찾기 (다음 table 또는 부모 근처)
        table = None
        for p in [parent] + list(parent.find_parents())[:5]:
            table = p.find_next("table") if p else None
            if table:
                break
        if not table:
            # dl/dt/dd 구조일 수 있음
            dl = parent.find_next("dl") or parent.find_next("ul")
            if dl:
                for dt in dl.find_all(["dt", "li"]):
                    name_el = dt.find(["strong", "b"]) or dt
                    desc_el = dt.find_next_sibling(["dd", "li"]) or dt
                    name = (name_el.get_text(strip=True) or "").strip(" :")
                    desc = (desc_el.get_text(strip=True) if desc_el else "").strip()
                    if name:
                        result.append({"name": name, "description": desc, "required": None})
                if result:
                    return result
            continue
        # 테이블 행 파싱 (헤더 한 줄 + 데이터)
        rows = table.find_all("tr")
        if not rows:
            continue
        headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
        for tr in rows[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if not cells:
                continue
            row_dict = {}
            for i, h in enumerate(headers):
                if i < len(cells) and h:
                    key = h.replace(" ", "_").strip("_")
                    row_dict[key] = cells[i]
            if row_dict:
                result.append(row_dict)
        if result:
            return result
    return result


def parse_output_values(soup) -> list:
    """HTML에서 출력(출력값) 테이블/리스트 파싱."""
    result = []
    for label in soup.find_all(string=re.compile(r"출력\s*(값)?", re.I)):
        parent = label.parent
        if not parent:
            continue
        table = None
        for p in [parent] + list(parent.find_parents())[:5]:
            table = p.find_next("table") if p else None
            if table:
                break
        if not table:
            dl = parent.find_next("dl") or parent.find_next("ul")
            if dl:
                for dt in dl.find_all(["dt", "li"]):
                    name_el = dt.find(["strong", "b"]) or dt
                    desc_el = dt.find_next_sibling(["dd", "li"]) or dt
                    name = (name_el.get_text(strip=True) or "").strip(" :")
                    desc = (desc_el.get_text(strip=True) if desc_el else "").strip()
                    if name:
                        result.append({"name": name, "description": desc})
                if result:
                    return result
            continue
        rows = table.find_all("tr")
        if not rows:
            continue
        headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
        for tr in rows[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if not cells:
                continue
            row_dict = {}
            for i, h in enumerate(headers):
                if i < len(cells) and h:
                    key = h.replace(" ", "_").strip("_")
                    row_dict[key] = cells[i]
            if row_dict:
                result.append(row_dict)
        if result:
            return result
    return result


def extract_api_meta(html: str) -> dict:
    """HTML에서 요청인자값·출력값 추출."""
    soup = BeautifulSoup(html, "html.parser")
    return {
        "request_params": parse_request_params(soup),
        "output_values": parse_output_values(soup),
    }


def main():
    matched = load_matched_metadata()
    print(f"matched_metadata 항목 수: {len(matched)}")

    results = []
    for i, item in enumerate(matched):
        rdata_seq = item.get("rdata_seq")
        rdata_name = item.get("rdata_name") or item.get("rdata_nm") or ""
        if not rdata_seq:
            print(f"  [{i+1}] rdata_seq 없음, 건너뜀: {rdata_name[:40]}...")
            continue
        try:
            html = fetch_detail_html(str(rdata_seq).strip())
            meta = extract_api_meta(html)
            results.append({
                "rdata_seq": rdata_seq,
                "rdata_name": rdata_name,
                "request_params": meta["request_params"],
                "output_values": meta["output_values"],
            })
            print(f"  [{i+1}] {rdata_seq} 요청인자 {len(meta['request_params'])}개, 출력 {len(meta['output_values'])}개")
            time.sleep(0.3)
        except Exception as e:
            print(f"  [{i+1}] {rdata_seq} 오류: {e}")
            results.append({
                "rdata_seq": rdata_seq,
                "rdata_name": rdata_name,
                "request_params": [],
                "output_values": [],
                "error": str(e),
            })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"저장: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
