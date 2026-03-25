"""Tests for scrape.py — normalizer, date parser, schema validation, export, HTTP."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from scrape import (
    _clean_html,
    _extract_review_items,
    _get_total_pages,
    _normalize_review,
    _parse_date,
    _request_with_retry,
    export_data,
    fetch_products,
    fetch_reviews_for_product,
    NAVER_REVIEW_URL,
    NAVER_SEARCH_URL,
)


# ---------------------------------------------------------------------------
# _clean_html
# ---------------------------------------------------------------------------


class TestCleanHtml:
    def test_removes_bold_tags(self):
        assert _clean_html("<b>무선이어폰</b> 추천") == "무선이어폰 추천"

    def test_removes_nested_tags(self):
        assert _clean_html("<b><i>test</i></b>") == "test"

    def test_no_tags(self):
        assert _clean_html("plain text") == "plain text"

    def test_empty_string(self):
        assert _clean_html("") == ""


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------


class TestParseDate:
    def test_iso_format(self):
        assert _parse_date("2024-01-15T10:30:00") == "2024-01-15"

    def test_iso_with_milliseconds(self):
        assert _parse_date("2024-01-15T10:30:00.123") == "2024-01-15"

    def test_iso_with_z(self):
        assert _parse_date("2024-01-15T10:30:00Z") == "2024-01-15"

    def test_date_only(self):
        assert _parse_date("2024-01-15") == "2024-01-15"

    def test_dot_format(self):
        assert _parse_date("2024.01.15") == "2024-01-15"

    def test_dot_format_trailing(self):
        assert _parse_date("2024.01.15.") == "2024-01-15"

    def test_datetime_space(self):
        assert _parse_date("2024-01-15 10:30:00") == "2024-01-15"

    def test_iso_with_timezone(self):
        result = _parse_date("2024-01-15T10:30:00+09:00")
        assert result == "2024-01-15"

    def test_short_string(self):
        assert _parse_date("2024") is None

    def test_garbage(self):
        result = _parse_date("not a date at all")
        # Should not crash
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# _extract_review_items
# ---------------------------------------------------------------------------


class TestExtractReviewItems:
    def test_pattern1_reviews_key(self):
        data = {"reviews": [{"content": "좋아요", "starScore": 5}]}
        assert _extract_review_items(data) == [{"content": "좋아요", "starScore": 5}]

    def test_pattern2_contents_key(self):
        data = {"contents": [{"content": "좋아요", "starScore": 5}]}
        assert _extract_review_items(data) == [{"content": "좋아요", "starScore": 5}]

    def test_pattern3_nested_data(self):
        data = {"data": {"reviews": [{"content": "좋아요"}]}}
        assert _extract_review_items(data) == [{"content": "좋아요"}]

    def test_pattern4_unknown_key_with_review_fields(self):
        data = {"items": [{"content": "좋아요", "starScore": 5}]}
        assert _extract_review_items(data) == [{"content": "좋아요", "starScore": 5}]

    def test_empty_reviews(self):
        data = {"reviews": []}
        assert _extract_review_items(data) == []

    def test_not_a_dict(self):
        assert _extract_review_items("string") is None

    def test_unknown_structure(self):
        data = {"foo": "bar", "baz": 42}
        assert _extract_review_items(data) is None

    def test_list_without_review_fields(self):
        data = {"items": [{"x": 1, "y": 2}]}
        assert _extract_review_items(data) is None


# ---------------------------------------------------------------------------
# _get_total_pages
# ---------------------------------------------------------------------------


class TestGetTotalPages:
    def test_total_pages_key(self):
        assert _get_total_pages({"totalPages": 5}) == 5

    def test_last_page_key(self):
        assert _get_total_pages({"lastPage": 3}) == 3

    def test_nested_data(self):
        assert _get_total_pages({"data": {"totalPages": 7}}) == 7

    def test_missing(self):
        assert _get_total_pages({"reviews": []}) is None

    def test_string_value(self):
        assert _get_total_pages({"totalPages": "5"}) == 5


# ---------------------------------------------------------------------------
# _normalize_review
# ---------------------------------------------------------------------------


class TestNormalizeReview:
    def test_full_review(self):
        raw = {
            "reviewId": "12345",
            "content": "배송 빠르고 음질 좋아요",
            "starScore": 5,
            "createDate": "2024-03-15T09:00:00",
            "purchaseVerified": True,
        }
        result = _normalize_review(raw, "prod_001")
        assert result["product_id"] == "prod_001"
        assert result["review_id"] == "12345"
        assert result["text"] == "배송 빠르고 음질 좋아요"
        assert result["rating"] == 5
        assert result["date"] == "2024-03-15"
        assert result["verified_purchase"] is True

    def test_alternative_field_names(self):
        raw = {
            "id": "999",
            "reviewContent": "좋습니다",
            "rating": 4,
            "created": "2024-06-01",
            "isBuyer": True,
        }
        result = _normalize_review(raw, "prod_002")
        assert result["review_id"] == "999"
        assert result["text"] == "좋습니다"
        assert result["rating"] == 4
        assert result["verified_purchase"] is True

    def test_missing_text_returns_none(self):
        raw = {"reviewId": "12345", "starScore": 5}
        assert _normalize_review(raw, "prod_001") is None

    def test_empty_text_returns_none(self):
        raw = {"reviewId": "12345", "content": "", "starScore": 5}
        assert _normalize_review(raw, "prod_001") is None

    def test_whitespace_text_returns_none(self):
        raw = {"reviewId": "12345", "content": "   ", "starScore": 5}
        assert _normalize_review(raw, "prod_001") is None

    def test_missing_rating(self):
        raw = {"reviewId": "12345", "content": "좋아요"}
        result = _normalize_review(raw, "prod_001")
        assert result is not None
        assert result["rating"] is None

    def test_not_a_dict(self):
        assert _normalize_review("string", "prod_001") is None
        assert _normalize_review(None, "prod_001") is None

    def test_generates_review_id_from_hash(self):
        raw = {"content": "좋아요"}
        result = _normalize_review(raw, "prod_001")
        assert result["review_id"].startswith("prod_001_")

    def test_korean_unicode(self):
        raw = {"reviewId": "1", "content": "한글 리뷰 🎧 특수문자!", "starScore": 3}
        result = _normalize_review(raw, "p1")
        assert result["text"] == "한글 리뷰 🎧 특수문자!"


# ---------------------------------------------------------------------------
# export_data
# ---------------------------------------------------------------------------


class TestExportData:
    def test_creates_json_and_parquet(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scrape.DATA_DIR", tmp_path)
        reviews = [
            {
                "product_id": "1",
                "platform": "naver",
                "product_name": "이어폰",
                "category": "디지털",
                "review_id": "r1",
                "rating": 5,
                "text": "좋아요",
                "date": "2024-01-01",
                "verified_purchase": True,
            },
            {
                "product_id": "1",
                "platform": "naver",
                "product_name": "이어폰",
                "category": "디지털",
                "review_id": "r2",
                "rating": 3,
                "text": "보통",
                "date": "2024-01-02",
                "verified_purchase": False,
            },
        ]
        json_path, parquet_path = export_data(reviews, "이어폰")
        assert json_path.exists()
        assert parquet_path.exists()

        # Verify JSON content
        with open(json_path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["text"] == "좋아요"

        # Verify Parquet content
        df = pd.read_parquet(parquet_path)
        assert len(df) == 2
        assert list(df.columns) == [
            "product_id", "platform", "product_name", "category",
            "review_id", "rating", "text", "date", "verified_purchase",
        ]

    def test_dedup_by_review_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scrape.DATA_DIR", tmp_path)
        reviews = [
            {
                "product_id": "1", "platform": "naver", "product_name": "x",
                "category": "c", "review_id": "dup", "rating": 5,
                "text": "first", "date": "2024-01-01", "verified_purchase": True,
            },
            {
                "product_id": "1", "platform": "naver", "product_name": "x",
                "category": "c", "review_id": "dup", "rating": 3,
                "text": "second", "date": "2024-01-02", "verified_purchase": False,
            },
        ]
        json_path, _ = export_data(reviews, "test")
        with open(json_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["text"] == "first"  # keeps first occurrence

    def test_empty_reviews(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scrape.DATA_DIR", tmp_path)
        json_path, parquet_path = export_data([], "empty")
        df = pd.read_parquet(parquet_path)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# _request_with_retry (with respx mocking)
# ---------------------------------------------------------------------------


class TestRequestWithRetry:
    @respx.mock
    def test_success_on_first_try(self, monkeypatch):
        monkeypatch.setattr("scrape.DELAY_BETWEEN_REQUESTS", 0)
        respx.get("https://example.com/api").respond(200, json={"ok": True})
        with httpx.Client() as client:
            resp = _request_with_retry(client, "https://example.com/api")
        assert resp.status_code == 200

    @respx.mock
    def test_retries_on_500(self, monkeypatch):
        monkeypatch.setattr("scrape.DELAY_BETWEEN_REQUESTS", 0)
        route = respx.get("https://example.com/api")
        route.side_effect = [
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json={"ok": True}),
        ]
        with httpx.Client() as client:
            resp = _request_with_retry(client, "https://example.com/api")
        assert resp.status_code == 200

    @respx.mock
    def test_retries_on_429(self, monkeypatch):
        monkeypatch.setattr("scrape.DELAY_BETWEEN_REQUESTS", 0)
        route = respx.get("https://example.com/api")
        route.side_effect = [
            httpx.Response(429),
            httpx.Response(200, json={"ok": True}),
        ]
        with httpx.Client() as client:
            resp = _request_with_retry(client, "https://example.com/api")
        assert resp.status_code == 200

    @respx.mock
    def test_raises_after_max_retries(self, monkeypatch):
        monkeypatch.setattr("scrape.DELAY_BETWEEN_REQUESTS", 0)
        monkeypatch.setattr("scrape.MAX_RETRIES", 2)
        respx.get("https://example.com/api").respond(500)
        with httpx.Client() as client:
            with pytest.raises(RuntimeError, match="Failed after"):
                _request_with_retry(client, "https://example.com/api")

    @respx.mock
    def test_passes_through_4xx(self, monkeypatch):
        monkeypatch.setattr("scrape.DELAY_BETWEEN_REQUESTS", 0)
        respx.get("https://example.com/api").respond(403)
        with httpx.Client() as client:
            resp = _request_with_retry(client, "https://example.com/api")
        assert resp.status_code == 403  # not retried


# ---------------------------------------------------------------------------
# fetch_products (with respx mocking)
# ---------------------------------------------------------------------------


class TestFetchProducts:
    @respx.mock
    def test_fetches_products(self, monkeypatch):
        monkeypatch.setattr("scrape.DELAY_BETWEEN_REQUESTS", 0)
        respx.get(NAVER_SEARCH_URL).respond(
            200,
            json={
                "items": [
                    {
                        "productId": "111",
                        "title": "<b>무선</b>이어폰",
                        "category1": "디지털",
                        "category2": "이어폰",
                        "category3": "",
                        "link": "https://example.com",
                        "mallName": "TestMall",
                    }
                ]
            },
        )
        with httpx.Client() as client:
            products = fetch_products(client, "이어폰", 1, "id", "secret")
        assert len(products) == 1
        assert products[0]["product_id"] == "111"
        assert products[0]["product_name"] == "무선이어폰"  # HTML tags removed
        assert products[0]["category"] == "디지털/이어폰"

    @respx.mock
    def test_empty_results(self, monkeypatch):
        monkeypatch.setattr("scrape.DELAY_BETWEEN_REQUESTS", 0)
        respx.get(NAVER_SEARCH_URL).respond(200, json={"items": []})
        with httpx.Client() as client:
            products = fetch_products(client, "없는상품", 5, "id", "secret")
        assert products == []


# ---------------------------------------------------------------------------
# fetch_reviews_for_product (with respx mocking)
# ---------------------------------------------------------------------------


class TestFetchReviewsForProduct:
    def test_legacy_returns_empty(self):
        """fetch_reviews_for_product is legacy (봇 감지로 비활성화) — 항상 빈 리스트 반환."""
        with httpx.Client() as client:
            reviews = fetch_reviews_for_product(client, "any_product", max_pages=1)
        assert reviews == []


# ---------------------------------------------------------------------------
# _extract_review_from_element (unit test for browser extraction logic)
# ---------------------------------------------------------------------------


class TestExtractReviewFromElement:
    def test_normalize_review_still_works(self):
        """Playwright 전환 후에도 _normalize_review는 정상 동작해야 함."""
        raw = {
            "reviewId": "r1",
            "content": "배송 빠르고 좋아요",
            "starScore": 5,
            "createDate": "2024-03-15T09:00:00",
            "purchaseVerified": True,
        }
        from scrape import _normalize_review
        result = _normalize_review(raw, "prod_001")
        assert result["text"] == "배송 빠르고 좋아요"
        assert result["rating"] == 5


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------


class TestMainValidation:
    def test_exits_without_env_vars(self, monkeypatch):
        monkeypatch.delenv("NAVER_CLIENT_ID", raising=False)
        monkeypatch.delenv("NAVER_CLIENT_SECRET", raising=False)
        monkeypatch.setattr("sys.argv", ["scrape.py"])
        with pytest.raises(SystemExit) as exc:
            from scrape import main
            main()
        assert exc.value.code == 1
