# agentreview

AI 쇼핑 에이전트를 위한 네이버쇼핑 상품 리뷰 데이터셋.

에이전트가 바로 활용할 수 있는 구조화된 리뷰 데이터를 제공합니다.

## 데이터 스키마 (V1)

| 필드 | 타입 | 설명 |
|------|------|------|
| `product_id` | string | 네이버 상품 ID |
| `platform` | string | 플랫폼 (naver) |
| `product_name` | string | 상품명 |
| `category` | string | 카테고리 (예: 디지털/가전/이어폰) |
| `review_id` | string | 리뷰 고유 ID |
| `rating` | int | 별점 (1-5) |
| `text` | string | 리뷰 본문 |
| `date` | string | 작성일 (YYYY-MM-DD) |
| `verified_purchase` | bool | 구매 인증 여부 |

## 설치

```bash
# uv 사용 (권장)
uv sync

# pip 사용
pip install -e .
```

## 사전 준비

1. [네이버 개발자 센터](https://developers.naver.com/apps/)에서 애플리케이션 등록
2. **검색 > 쇼핑** API 사용 설정
3. 환경변수 설정:

```bash
export NAVER_CLIENT_ID="your_client_id"
export NAVER_CLIENT_SECRET="your_client_secret"

# HuggingFace 업로드 시
export HF_TOKEN="your_hf_token"
```

## 사용법

```bash
# 기본 실행 (무선이어폰, 50개 상품)
python scrape.py

# 카테고리 & 상품 수 지정
python scrape.py --query "노트북" --max-products 100

# HuggingFace에 업로드
python scrape.py --query "무선이어폰" --upload --repo-id "username/naver-reviews"
```

## 에이전트에서 사용하기

### Python

```python
from datasets import load_dataset

ds = load_dataset("username/naver-reviews")

# 특정 상품의 리뷰 필터링
product_reviews = ds.filter(lambda x: x["product_id"] == "12345")

# 별점 4점 이상 리뷰만
good_reviews = ds.filter(lambda x: x["rating"] and x["rating"] >= 4)

# pandas로 변환
df = ds["train"].to_pandas()
```

### JavaScript / TypeScript

```typescript
// HuggingFace에서 직접 Parquet 다운로드
const response = await fetch(
  "https://huggingface.co/datasets/username/naver-reviews/resolve/main/data/train-00000-of-00001.parquet"
);
```

## 출력 파일

`data/` 디렉토리에 생성됩니다:
- `reviews_{query}_{date}.json` — JSON 형식
- `reviews_{query}_{date}.parquet` — Parquet 형식 (에이전트 통합에 권장)

## 법적 고지

이 도구는 교육 및 연구 목적으로 제공됩니다. 크롤링 실행 전 네이버 이용약관과 관련 법률을 반드시 확인하세요.

## License

MIT
