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
- 응답 JSON을 Supabase 테이블(tb_rdata_result)에 저장

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


def get_date_range_last_2_years() -> tuple[str, str]:
    """최근 2년(포함)의 yyyyMM 구간(startDt, endDt) 계산."""
    today = datetime.date.today()
    end = today.replace(day=1)  # 이번 달 1일 기준
    start = (end.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
    # start 기준으로 다시 23개월 전으로 이동하여 총 24개월(2년) 구간 확보
    for _ in range(23):
        start = (start.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)

    start_str = start.strftime("%Y%m")
    end_str = end.strftime("%Y%m")
    return start_str, end_str


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
        "startDt": start_ym,
        "endDt": end_ym,
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


def upsert_rdata_result(
    sb: Client,
    rdata_seq: str,
    start_ym: str,
    end_ym: str,
    payload: Dict[str, Any],
) -> None:
    """
    응답 JSON을 Supabase tb_rdata_result 테이블에 upsert.

    추천 스키마:
      tb_rdata_result(
        id           bigserial primary key,
        rdata_seq    varchar not null,
        start_ym     varchar(6) not null,
        end_ym       varchar(6) not null,
        raw_json     jsonb    not null,
        created_at   timestamp with time zone default now()
      )
    """
    row = {
        "rdata_seq": rdata_seq,
        "start_ym": start_ym,
        "end_ym": end_ym,
        "raw_json": payload,
    }
    # 단순 insert (중복 허용) 또는 rdata_seq+start_ym+end_ym 기준 upsert는
    # Supabase 쿼리에서 on_conflict 설정으로 조절 가능.
    resp = sb.table("tb_rdata_result").insert(row).execute()
    if getattr(resp, "error", None):
        print(f"[{rdata_seq}] tb_rdata_result 저장 오류: {resp.error}")


def main() -> None:
    sb = init_supabase()
    start_ym, end_ym = get_date_range_last_2_years()
    default_key = os.getenv("IROS_DEFAULT_KEY")

    rows = fetch_rdata_rows(sb)
    print(f"tb_rdata에서 {len(rows)}건 조회. 기간: {start_ym} ~ {end_ym}")

    for i, row in enumerate(rows, start=1):
        rdata_seq = str(row.get("rdata_seq") or "").strip()
        api_key = (row.get("key") or default_key or "").strip()

        if not rdata_seq:
            print(f"[{i}] rdata_seq 없음. 건너뜀.")
            continue
        if not api_key:
            print(f"[{i}] rdata_seq={rdata_seq} API key 없음. 건너뜀.")
            continue

        print(f"[{i}] rdata_seq={rdata_seq} 데이터 수집 시작...")
        data = call_open_api(rdata_seq, api_key, start_ym, end_ym)
        if data is None:
            print(f"[{i}] rdata_seq={rdata_seq} 호출 실패.")
            continue

        upsert_rdata_result(sb, rdata_seq, start_ym, end_ym, data)
        print(f"[{i}] rdata_seq={rdata_seq} 저장 완료.")


if __name__ == "__main__":
    main()

