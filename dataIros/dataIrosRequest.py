# -*- coding: utf-8 -*-
"""
Supabase 연동 + 대법원 등기 Open API 수집 스크립트.

기능
- Supabase의 tb_rdata 테이블에서 rdata_seq, key(= Open API 서비스 키) 목록을 조회
- 각 rdata_seq 별로 data.iros.go.kr Open API를 호출
  - 엔드포인트: https://data.iros.go.kr/openapi/cr/rs/selectCrRsRgsCsOpenApi.rest
  - query string:
      id={rdata_seq}
      key={tb_rdata.key}
      reqtype=json
      startDt, endDt: 최근 2년 (yyyyMM)
- 응답 JSON을 Supabase 테이블(tb_api_result)에 저장

전제
- 환경변수에 아래 값이 설정되어 있어야 합니다.
  - SUPABASE_URL
  - SUPABASE_SERVICE_ROLE_KEY (또는 SUPABASE_ANON_KEY)
  - (선택) IROS_DEFAULT_KEY: tb_rdata.key 가 비어 있을 때 사용할 기본 키

필요 패키지:
  pip install supabase-py requests python-dotenv
"""

import os
import json
import datetime
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from supabase import create_client, Client


OPEN_API_BASE_URL = "https://data.iros.go.kr/openapi/cr/rs/selectCrRsRgsCsOpenApi.rest"


def _shift_months(base: datetime.date, months: int) -> datetime.date:
    """base에서 months만큼 월을 이동한 날짜(같은 일자, 없으면 말일). 내부용."""
    year = base.year + (base.month - 1 + months) // 12
    month = (base.month - 1 + months) % 12 + 1
    day = min(
        base.day,
        [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][
            month - 1
        ],
    )
    return datetime.date(year, month, day)


def get_date_ranges_last_2_years() -> list[tuple[str, str]]:
    """
    최근 2년을 3개월 단위로 나눈 여덟 구간을 yyyyMM(startDt, endDt)으로 반환.

    예) 오늘이 2026-03 이라면 (각 구간은 3개월):
      - range1: 2024-04 ~ 2024-06
      - range2: 2024-07 ~ 2024-09
      - range3: 2024-10 ~ 2024-12
      - range4: 2025-01 ~ 2025-03
      - range5: 2025-04 ~ 2025-06
      - range6: 2025-07 ~ 2025-09
      - range7: 2025-10 ~ 2025-12
      - range8: 2026-01 ~ 2026-03
    """
    today = datetime.date.today()
    # 가장 최근 3개월 구간의 끝: 이번 달 1일
    end8 = today.replace(day=1)
    start8 = _shift_months(end8, -2)  # 2개월 전 1일 → 총 3개월

    # 그 이전 3개월 * 7 구간
    end7 = _shift_months(start8, -1)
    start7 = _shift_months(end7, -2)

    end6 = _shift_months(start7, -1)
    start6 = _shift_months(end6, -2)

    end5 = _shift_months(start6, -1)
    start5 = _shift_months(end5, -2)

    end4 = _shift_months(start5, -1)
    start4 = _shift_months(end4, -2)

    end3 = _shift_months(start4, -1)
    start3 = _shift_months(end3, -2)

    end2 = _shift_months(start3, -1)
    start2 = _shift_months(end2, -2)

    end1 = _shift_months(start2, -1)
    start1 = _shift_months(end1, -2)

    return [
        (start1.strftime("%Y%m"), end1.strftime("%Y%m")),
        (start2.strftime("%Y%m"), end2.strftime("%Y%m")),
        (start3.strftime("%Y%m"), end3.strftime("%Y%m")),
        (start4.strftime("%Y%m"), end4.strftime("%Y%m")),
        (start5.strftime("%Y%m"), end5.strftime("%Y%m")),
        (start6.strftime("%Y%m"), end6.strftime("%Y%m")),
        (start7.strftime("%Y%m"), end7.strftime("%Y%m")),
        (start8.strftime("%Y%m"), end8.strftime("%Y%m")),
    ]


def init_supabase() -> Client:
    """환경변수에서 Supabase 클라이언트 생성."""
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        raise RuntimeError("SUPABASE_URL 또는 SUPABASE_SERVICE_ROLE_KEY/ANON_KEY 환경변수가 설정되지 않았습니다.")

    return create_client(url, key)


def fetch_rdata_rows(sb: Client) -> List[Dict[str, Any]]:
    """
    Supabase tb_rdata 테이블에서 rdata_seq, key 컬럼 조회.
    - 스키마 예시:
        tb_rdata(
          rdata_seq varchar primary key,
          key text        -- 각 API 서비스 키
        )
    """
    resp = sb.table("tb_rdata").select("rdata_seq,key").execute()
    if hasattr(resp, "data"):
        return resp.data or []
    # supabase-py v1 스타일 대응
    return resp.get("data", []) if isinstance(resp, dict) else []


def build_openapi_url(rdata_seq: str, api_key: str, start_ym: str, end_ym: str) -> str:
    """요청 URL 생성."""
    params = {
        "id": rdata_seq,
        "key": api_key,
        "reqtype": "json",
        "search_type_api" : "02",
        "search_start_date_api": start_ym,
        "search_end_date_api": end_ym,
    }
    # query string 조합
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{OPEN_API_BASE_URL}?{qs}"


def call_open_api(rdata_seq: str, api_key: str, start_ym: str, end_ym: str) -> Optional[Dict[str, Any]]:
    """등기 통계 Open API 호출 후 JSON 반환. 실패 시 None."""
    url = build_openapi_url(rdata_seq, api_key, start_ym, end_ym)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[{rdata_seq}] HTTP 오류: {e}")
        return None

    # Open API는 오류 시에도 JSON 대신 HTML/텍스트를 줄 수 있으므로 방어적으로 처리
    try:
        data = resp.json()
    except json.JSONDecodeError:
        print(f"[{rdata_seq}] JSON 파싱 실패. body 앞 200자: {resp.text[:200]!r}")
        return None

    return data


def upsert_api_result(
    sb: Client,
    rdata_seq: str,
    start_ym: str,
    end_ym: str,
    payload: Dict[str, Any],
) -> None:
    """
    응답 JSON을 Supabase tb_api_result 테이블에 upsert.

    JSON 최상위 key들을 그대로 컬럼으로 삽입하고,
    rdata_seq / start_ym / end_ym를 함께 저장한다.

    예시 스키마:
      tb_api_result(
        id           bigserial primary key,
        rdata_seq    varchar not null,
        start_ym     varchar(6) not null,
        end_ym       varchar(6) not null,
        ...          -- Open API JSON의 key에 해당하는 컬럼들
        created_at   timestamp with time zone default now()
      )
    """
    # payload 구조:
    # {
    #   "result": {
    #     "head": {...},
    #     "items": {
    #       "item": [ {...}, {...}, ... ]
    #     }
    #   }
    # }
    
    #items: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        result = payload.get("result") or {}
        container = result.get("items") or {}
        items = container.get("item") or []
        rows: List[Dict[str, Any]] = []
        for item in items:
            row = dict(item)  # resDate, adminRegn1Name, adminRegn2Name, tot 등을 그대로 컬럼으로 사용
            row["rdata_seq"] = rdata_seq
            rows.append(row)

        resp = sb.table("tb_api_result").insert(rows).execute()
    if getattr(resp, "error", None):
        print(f"[{rdata_seq}] tb_api_result 저장 오류: {resp.error}")


def main() -> None:
    sb = init_supabase()
    ranges = get_date_ranges_last_2_years()
    default_key = os.getenv("IROS_DEFAULT_KEY")

    rows = fetch_rdata_rows(sb)
    print(
        "tb_rdata에서 {cnt}건 조회. 기간: ".format(cnt=len(rows))
        + ", ".join(f"{s}~{e}" for s, e in ranges)
    )

    for i, row in enumerate(rows, start=1):
        rdata_seq = str(row.get("rdata_seq") or "").strip()
        api_key = (row.get("key") or default_key or "").strip()

        if not rdata_seq:
            print(f"[{i}] rdata_seq 없음. 건너뜀.")
            continue
        if not api_key:
            print(f"[{i}] rdata_seq={rdata_seq} API key 없음. 건너뜀.")
            continue

        for idx, (start_ym, end_ym) in enumerate(ranges, start=1):
            print(f"[{i}-{idx}] rdata_seq={rdata_seq} 데이터 수집 시작... ({start_ym}~{end_ym})")
            data = call_open_api(rdata_seq, api_key, start_ym, end_ym)
            if data is None:
                print(f"[{i}-{idx}] rdata_seq={rdata_seq} ({start_ym}~{end_ym}) 호출 실패.")
                continue

            upsert_api_result(sb, rdata_seq, start_ym, end_ym, data)
            print(f"[{i}-{idx}] rdata_seq={rdata_seq} ({start_ym}~{end_ym}) 저장 완료.")


if __name__ == "__main__":
    main()

