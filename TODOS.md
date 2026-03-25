# TODOS

## BLOCKING — 코드 작성 전 해결 필요

### 1. 법적 검토: 네이버 리뷰 크롤링 + 재배포 합법성 확인
- **What:** 한국 IP 변호사와 네이버쇼핑 리뷰 크롤링 + HuggingFace 재배포의 법적 허용 범위 확인
- **Why:** 네이버 ToS 위반 시 프로젝트 자체가 불가능. 한국법 하에서 UGC(사용자 생성 콘텐츠) 크롤링/재배포는 공격적으로 집행됨
- **Pros:** 법적 리스크 제거, 프로젝트 존폐 조기 결정
- **Cons:** 변호사 비용 발생, 시간 소요
- **Context:** /office-hours 디자인 문서에서 Constraint #1로 이미 명시. /plan-eng-review의 Outside Voice가 "리스크가 아닌 blocker"로 격상 권고. 만약 불가능한 경우 대안: (a) 네이버 공식 API/파트너십, (b) 사용자 자발적 리뷰 제출 모델, (c) 타 플랫폼 검토
- **Depends on:** 없음 — 최우선
- **Trigger:** 즉시 (코드 작성 전)

---

## DEFERRED — 수요 확인 후 진행

### 2. Phase 2: 프로덕션 파이프라인 구축
- **What:** scrape.py → 4개 모듈(crawler/normalizer/exporter/config) 분리, 24개 테스트 패스 작성, GitHub Actions 주간 크론, 증분 업데이트(review_id dedup), 데이터셋 버전 관리
- **Why:** Phase 1 MVP(단일 스크립트)로 수요를 확인한 후, 안정적 운영을 위해 모듈화 + 자동화 필요
- **Pros:** 자동 주간 업데이트, 안정적 에러 처리, 100% 테스트 커버리지
- **Cons:** 수요 없으면 전부 낭비. ~2-3시간 CC 작업
- **Context:** /plan-eng-review에서 24개 코드 패스 + 2개 E2E 테스트 설계 완료. 테스트 플랜: ~/.gstack/projects/agentreview/slave2-unknown-eng-review-test-plan-20260325-215703.md
- **Depends on:** Phase 1 MVP 배포 + 수요 확인 (다운로드 100+ 또는 외부 피드백 1+)
- **Trigger:** HuggingFace 4주 후 수요 지표 확인
