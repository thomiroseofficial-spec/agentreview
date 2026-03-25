"""
agentreview — 네이버쇼핑 리뷰 수집 → HuggingFace 데이터셋 MVP

Usage:
    # 기본 실행 (전자제품 카테고리, 50개 상품)
    python scrape.py

    # 카테고리 & 상품 수 지정
    python scrape.py --query "무선이어폰" --max-products 100

    # HuggingFace 업로드 포함
    python scrape.py --query "무선이어폰" --upload --repo-id "username/naver-reviews"

환경변수:
    NAVER_CLIENT_ID     — 네이버 검색 API Client ID
    NAVER_CLIENT_SECRET — 네이버 검색 API Client Secret
    HF_TOKEN            — HuggingFace 토큰 (업로드 시 필요)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NAVER_SEARCH_URL = "https://openapi.naver.com/v1/search/shop.json"
NAVER_REVIEW_URL = "https://search.shopping.naver.com/api/review"

DATA_DIR = Path("data")

DELAY_BETWEEN_REQUESTS = 1.5  # seconds
MAX_RETRIES = 3
BACKOFF_FACTOR = 2

V1_COLUMNS = [
    "product_id",
    "platform",
    "product_name",
    "category",
    "review_id",
    "rating",
    "text",
    "date",
    "verified_purchase",
]

HEADERS_REVIEW = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://search.shopping.naver.com/",
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _request_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
) -> httpx.Response:
    """GET request with exponential backoff on rate limit / server errors."""
    delay = DELAY_BETWEEN_REQUESTS
    for attempt in range(MAX_RETRIES):
        time.sleep(delay if attempt == 0 else delay * (BACKOFF_FACTOR**attempt))
        try:
            resp = client.get(url, params=params, headers=headers, timeout=30)
        except httpx.TimeoutException:
            print(f"  [TIMEOUT] attempt {attempt + 1}/{MAX_RETRIES}: {url}")
            continue
        except httpx.HTTPError as e:
            print(f"  [ERROR] attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            print(
                f"  [RETRY] status={resp.status_code}, "
                f"attempt {attempt + 1}/{MAX_RETRIES}"
            )
            continue
        return resp

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {url}")


# ---------------------------------------------------------------------------
# Step 1: 네이버 검색 API로 상품 목록 수집
# ---------------------------------------------------------------------------


def fetch_products(
    client: httpx.Client,
    query: str,
    max_products: int,
    client_id: str,
    client_secret: str,
) -> list[dict]:
    """네이버 검색 API로 상품 목록을 가져옵니다."""
    products = []
    display = min(100, max_products)  # API 최대 100건
    start = 1

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }

    while len(products) < max_products and start <= 1000:
        params = {
            "query": query,
            "display": display,
            "start": start,
            "sort": "review",  # 리뷰 많은 순
        }

        resp = _request_with_retry(
            client, NAVER_SEARCH_URL, params=params, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        if not items:
            break

        for item in items:
            products.append(
                {
                    "product_id": item.get("productId", ""),
                    "product_name": _clean_html(item.get("title", "")),
                    "category": item.get("category1", "")
                    + ("/" + item["category2"] if item.get("category2") else "")
                    + ("/" + item["category3"] if item.get("category3") else ""),
                    "link": item.get("link", ""),
                    "mall_name": item.get("mallName", ""),
                }
            )
            if len(products) >= max_products:
                break

        start += display

    print(f"[PRODUCTS] {len(products)}개 상품 수집 완료 (query: {query})")
    return products


def _clean_html(text: str) -> str:
    """Remove HTML tags from Naver API response."""
    import re

    return re.sub(r"<[^>]+>", "", text)


# ---------------------------------------------------------------------------
# Step 2: 상품별 리뷰 수집 (XHR 엔드포인트)
# ---------------------------------------------------------------------------


def fetch_reviews_for_product(
    client: httpx.Client,
    product_id: str,
    max_pages: int = 5,
) -> list[dict]:
    """네이버쇼핑 내부 리뷰 API에서 리뷰를 수집합니다."""
    reviews = []

    for page in range(1, max_pages + 1):
        params = {
            "nvMid": product_id,
            "reviewType": "ALL",
            "page": page,
            "pageSize": 20,
            "sortType": "RECENT",
        }

        try:
            resp = _request_with_retry(
                client,
                NAVER_REVIEW_URL,
                params=params,
                headers=HEADERS_REVIEW,
            )
        except RuntimeError:
            print(f"  [SKIP] product {product_id} page {page} — retries exhausted")
            break

        if resp.status_code == 403:
            print(f"  [BLOCKED] product {product_id} — 403 Forbidden")
            break

        if resp.status_code != 200:
            print(
                f"  [SKIP] product {product_id} page {page} "
                f"— status {resp.status_code}"
            )
            break

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            print(f"  [PARSE ERROR] product {product_id} page {page}")
            break

        # 스키마 검증: 응답 구조가 예상과 다르면 중단
        review_items = _extract_review_items(data)
        if review_items is None:
            print(
                f"  [SCHEMA CHANGED] product {product_id} — "
                f"unexpected response structure. Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
            )
            break

        if not review_items:
            break  # 더 이상 리뷰 없음

        for item in review_items:
            review = _normalize_review(item, product_id)
            if review:
                reviews.append(review)

        # 마지막 페이지 감지
        total_pages = _get_total_pages(data)
        if total_pages and page >= total_pages:
            break

    return reviews


def _extract_review_items(data: dict) -> list[dict] | None:
    """리뷰 API 응답에서 리뷰 목록을 추출합니다. 구조 변경 시 None 반환."""
    if not isinstance(data, dict):
        return None

    # 여러 가능한 응답 구조를 시도
    # Pattern 1: {"reviews": [...]}
    if "reviews" in data:
        items = data["reviews"]
        if isinstance(items, list):
            return items

    # Pattern 2: {"contents": [...]}
    if "contents" in data:
        items = data["contents"]
        if isinstance(items, list):
            return items

    # Pattern 3: {"data": {"reviews": [...]}}
    if "data" in data and isinstance(data["data"], dict):
        nested = data["data"]
        if "reviews" in nested and isinstance(nested["reviews"], list):
            return nested["reviews"]
        if "contents" in nested and isinstance(nested["contents"], list):
            return nested["contents"]

    # Pattern 4: top-level list wrapped in a dict with a single key
    for key, val in data.items():
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            # Looks like a review list
            if any(
                k in val[0]
                for k in ("content", "reviewContent", "body", "text", "starScore")
            ):
                return val

    # 알려진 패턴 없음 — 구조 변경으로 간주
    return None


def _get_total_pages(data: dict) -> int | None:
    """응답에서 전체 페이지 수를 추출합니다."""
    for key in ("totalPages", "total_pages", "lastPage"):
        if key in data:
            try:
                return int(data[key])
            except (ValueError, TypeError):
                pass
    if "data" in data and isinstance(data["data"], dict):
        return _get_total_pages(data["data"])
    return None


# ---------------------------------------------------------------------------
# Step 3: 리뷰 정규화 (V1 스키마)
# ---------------------------------------------------------------------------


def _normalize_review(raw: dict, product_id: str) -> dict | None:
    """원본 리뷰 데이터를 V1 스키마로 정규화합니다."""
    if not isinstance(raw, dict):
        return None

    # 리뷰 텍스트 추출 (여러 필드명 시도)
    text = None
    for key in ("content", "reviewContent", "body", "text", "commentContent"):
        if key in raw and raw[key]:
            text = str(raw[key]).strip()
            break

    if not text:
        return None  # 텍스트 없는 리뷰는 스킵

    # 별점 추출
    rating = None
    for key in ("starScore", "rating", "score", "star", "grade"):
        if key in raw:
            try:
                rating = int(raw[key])
                break
            except (ValueError, TypeError):
                pass

    # 리뷰 ID 추출
    review_id = None
    for key in ("reviewId", "id", "reviewNo", "no"):
        if key in raw:
            review_id = str(raw[key])
            break
    if not review_id:
        review_id = f"{product_id}_{hash(text) % 10**8}"

    # 날짜 추출
    date_str = None
    for key in (
        "createDate",
        "created",
        "createdAt",
        "registerDate",
        "writeDate",
        "date",
    ):
        if key in raw and raw[key]:
            date_str = _parse_date(str(raw[key]))
            break

    # 구매 인증 여부
    verified = False
    for key in ("purchaseVerified", "verified", "isBuyer", "buyerReview"):
        if key in raw:
            verified = bool(raw[key])
            break

    return {
        "product_id": product_id,
        "review_id": review_id,
        "rating": rating,
        "text": text,
        "date": date_str,
        "verified_purchase": verified,
    }


def _parse_date(date_str: str) -> str | None:
    """다양한 날짜 형식을 YYYY-MM-DD로 변환합니다."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y.%m.%d",
        "%Y.%m.%d.",
    ):
        try:
            dt = datetime.strptime(date_str[:len(fmt.replace('%', ''))+ 5], fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # ISO format fallback
    if "T" in date_str:
        try:
            return date_str[:10]
        except Exception:
            pass

    return date_str[:10] if len(date_str) >= 10 else None


# ---------------------------------------------------------------------------
# Step 4: 내보내기 (JSON + Parquet)
# ---------------------------------------------------------------------------


def export_data(reviews: list[dict], query: str) -> tuple[Path, Path]:
    """리뷰 데이터를 JSON + Parquet 파일로 내보냅니다."""
    DATA_DIR.mkdir(exist_ok=True)

    df = pd.DataFrame(reviews, columns=V1_COLUMNS)

    # review_id 기반 중복 제거
    before = len(df)
    df = df.drop_duplicates(subset=["review_id"], keep="first")
    after = len(df)
    if before != after:
        print(f"[DEDUP] {before - after}개 중복 리뷰 제거")

    timestamp = datetime.now().strftime("%Y%m%d")
    safe_query = "".join(c if c.isalnum() else "_" for c in query)

    json_path = DATA_DIR / f"reviews_{safe_query}_{timestamp}.json"
    parquet_path = DATA_DIR / f"reviews_{safe_query}_{timestamp}.parquet"

    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    print(f"[EXPORT] {len(df)}개 리뷰 저장")
    print(f"  JSON:    {json_path}")
    print(f"  Parquet: {parquet_path}")

    return json_path, parquet_path


# ---------------------------------------------------------------------------
# Step 5: HuggingFace 업로드
# ---------------------------------------------------------------------------


def upload_to_hub(parquet_path: Path, repo_id: str) -> None:
    """HuggingFace Hub에 데이터셋을 업로드합니다."""
    from datasets import Dataset

    df = pd.read_parquet(parquet_path)
    ds = Dataset.from_pandas(df)
    ds.push_to_hub(repo_id, private=False)
    print(f"[UPLOAD] HuggingFace에 업로드 완료: https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="네이버쇼핑 리뷰 수집 → HuggingFace 데이터셋"
    )
    parser.add_argument(
        "--query", default="무선이어폰", help="검색 키워드 (default: 무선이어폰)"
    )
    parser.add_argument(
        "--max-products", type=int, default=50, help="수집할 상품 수 (default: 50)"
    )
    parser.add_argument(
        "--max-review-pages",
        type=int,
        default=5,
        help="상품당 최대 리뷰 페이지 수 (default: 5, 페이지당 20개)",
    )
    parser.add_argument("--upload", action="store_true", help="HuggingFace에 업로드")
    parser.add_argument("--repo-id", default=None, help="HuggingFace repo ID")
    args = parser.parse_args()

    # 환경변수 확인
    client_id = os.environ.get("NAVER_CLIENT_ID")
    client_secret = os.environ.get("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        print(
            "ERROR: NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET 환경변수를 설정하세요.\n"
            "  https://developers.naver.com/apps/ 에서 애플리케이션을 등록하세요.\n"
            "  검색 > 쇼핑 API를 사용설정해야 합니다."
        )
        sys.exit(1)

    if args.upload and not args.repo_id:
        print("ERROR: --upload 시 --repo-id가 필요합니다.")
        sys.exit(1)

    if args.upload and not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN 환경변수를 설정하세요.")
        sys.exit(1)

    print(f"=== agentreview MVP ===")
    print(f"Query: {args.query}")
    print(f"Max products: {args.max_products}")
    print(f"Max review pages per product: {args.max_review_pages}")
    print()

    with httpx.Client() as client:
        # Step 1: 상품 목록 수집
        products = fetch_products(
            client, args.query, args.max_products, client_id, client_secret
        )

        if not products:
            print("ERROR: 상품을 찾을 수 없습니다.")
            sys.exit(1)

        # Step 2: 리뷰 수집
        all_reviews = []
        for i, product in enumerate(products):
            pid = product["product_id"]
            pname = product["product_name"][:30]
            print(
                f"[{i + 1}/{len(products)}] {pname}... (id: {pid})"
            )

            reviews = fetch_reviews_for_product(
                client, pid, max_pages=args.max_review_pages
            )

            # 상품 정보 병합
            for r in reviews:
                r["platform"] = "naver"
                r["product_name"] = product["product_name"]
                r["category"] = product["category"]

            all_reviews.extend(reviews)
            print(f"  → {len(reviews)}개 리뷰 수집")

        # Zero-result 감지
        if not all_reviews:
            print(
                "\nERROR: 리뷰를 하나도 수집하지 못했습니다.\n"
                "가능한 원인:\n"
                "  1. 네이버 리뷰 API 엔드포인트가 변경되었을 수 있습니다\n"
                "  2. IP가 차단되었을 수 있습니다\n"
                "  3. 검색 결과 상품에 리뷰가 없을 수 있습니다"
            )
            sys.exit(1)

        print(f"\n[TOTAL] {len(all_reviews)}개 리뷰 수집 완료")

        # Step 3-4: 내보내기
        json_path, parquet_path = export_data(all_reviews, args.query)

        # Step 5: 업로드 (선택)
        if args.upload:
            upload_to_hub(parquet_path, args.repo_id)

    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
