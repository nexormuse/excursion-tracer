# 실패 사례 분석 2차 (test2, agent_v2)

선정 규칙: 원인 시나리오는 Hit@3 엄격 실패, 원인 없는 시나리오는 오경보를 실패로 보고, 실패 비율이 높은 원인 유형부터 5개 유형을 골라 유형마다 번호가 가장 작은 실패 시나리오를 택했다.

| 유형 | 실패 | 전체 |
|---|---|---|
| F0a | 4 | 12 |
| F0b | 6 | 12 |
| F1 | 2 | 18 |
| F2 | 5 | 15 |
| F3 | 3 | 15 |
| F4 | 3 | 15 |
| F5 | 14 | 15 |
| F6 | 15 | 18 |

<!-- manual:start -->
## 사례별 판단 지점 요약 (기록의 사실만)

| 시나리오 | 유형 | 정답 | 에이전트 최종 1순위 | 판단이 갈린 지점 (근거 묶음·확인 결과 원문) |
|---|---|---|---|---|
| scn_7002 | F5 | 레시피 S20-R1 (시작 2026-01-09T05:01) | no_equipment_cause | 이벤트 스캔 줄 E26이 `RECIPE_CHANGE S20-R1 ... median diff -0.073, q 0.000, 3.77x normal spread`를 보여 줬다. 잠정 보고서는 설비 S37-T1(0.75) 등을 냈고, 확인 C1 time_trend는 `entity_id is required` 오류로 돌아왔다. 최종 설명은 "All candidate differences are below 1.5x the normal spread"였다 |
| scn_7004 | F6 | 챔버 S21-T2-C1 | no_equipment_cause | 정답 챔버가 근거 묶음 어느 줄에도 나오지 않았다. 제품 보정 줄 E6은 `change 0.002; ... held at the pre-alert shares -0.001`. 확인은 product_mix 1건만 요청했다 |
| scn_7005 | F0b | 설비 원인 없음 (P2 비중 변화) | chamber S01-T2-C3, 신뢰도 0.7 | 인용한 E17은 `1.15x normal spread`, E29 변화점은 `permutation p 0.265`. 1.5배 미만 후보를 신뢰도 0.7로 유지했다 |
| scn_7007 | F2 | S35-T4 × S35-R2 | tool_recipe S25-T3 × S25-R2, 신뢰도 0.8 | 근거 묶음에 E14 `S35-T4 ... 1.89x normal spread`와 E43 `most deviant cell S35-T4 x S35-R2`가 있었다. 최종 보고서는 S35-T4를 3순위 설비(레시피 없음, 0.6)로 내 F2 엄격 규칙에 미달했다. 확인 C2 time_trend는 `entity_id is required` 오류였다 |
| scn_7017 | F0a | 설비 원인 없음 | tool S21-T4, 신뢰도 0.75 | 인용한 E12는 `1.35x normal spread`, E29 변화점은 `permutation p 0.008`. 확인 C2 time_trend는 `entity_id is required` 오류, C3 confounding_check는 S21-T4와 S21-T4-C3를 비교했다 |
<!-- manual:end -->

<!-- diagnosis:start -->
## 1차 진단 D-1~D-6의 2차 결과

1차 = 에이전트 v1·기준선 v1, test1 / 2차 = 에이전트 v2·기준선 v2, test2 (서로 다른 시험 세트). 원천: results/diagnosis_v1_v2.json

| 진단 | 1차 | 2차 |
|---|---|---|
| D-1 줄여 쓴 ID가 오답 처리 | 틀린 원인 시나리오 57개 중 ID 표기 문제 15개 (정규화하면 정답 13개); F1 Hit@3 엄격 13/24 = 0.542 | ID 정규화 0건, 재요청 0/120 = 0.000, 남은 문제 0/120 = 0.000; F1 Hit@3 엄격 16/18 = 0.889 |
| D-2 Hit@1 > Hit@3 | 통일 후 Hit@1 45/96 = 0.469, Hit@3 50/96 = 0.521 | Hit@1 50/96 = 0.521, Hit@3 54/96 = 0.562 |
| D-3 오경보·신뢰도 | 에이전트 17/24 = 0.708, 기준선 24/24 = 1.000; 원인 없는 시나리오 1순위 신뢰도 0.8 이상 16/17 | 에이전트 10/24 = 0.417, 기준선 v2 12/24 = 0.500; 1순위 신뢰도 0.8 이상 2/10 |
| D-4 추가 확인 요청 | 0/120 = 0.000 | 120/120 = 1.000, 1순위 변경 9/120 = 0.075 |
| D-5 단계 전체 원인(F5)·기준선 설비×레시피(F2) | F5 에이전트 0/18 = 0.000, 기준선 0/18 = 0.000; 기준선 F2 0/18 = 0.000 | F5 에이전트 1/15 = 0.067, 기준선 v2 14/15 = 0.933; 기준선 v2 F2 11/15 = 0.733 |
| D-6 출력 길이에서 끊김 | 스키마 재요청 시나리오 16/120, 끊긴 응답 16 | 스키마 재요청 시나리오 2/120, 끊긴 응답 2 |
| 추가: 확인 요청이 오류로 돌아감 (필드 누락 등) | 확인 요청 0건 | 확인 요청 226건 중 50건 (시나리오 47개; before_after 7, confounding_check 3, drilldown 2, interaction_test 1, time_trend 37) |
<!-- diagnosis:end -->

### scn_7002 — F5 (효과 medium, stickiness high, 알림 alarm)

**정답**
- F5: step S20, S20-R1, 시작 2026-01-09T05:01:29, δ 0.07
  - S20-R1: 전체 공통성 스캔 스캔 대상 아님
  - 근거 묶음에 나온 줄: 1개
    - `E26 EV0260 RECIPE_CHANGE S20-R1 (step S20) at 2026-01-09T05:01: n before 1775, after 1850, median diff -0.073, q 0.000, 3.77x normal spread`

**에이전트 최종 보고서**
- 판정 no_equipment_cause, 추가 확인 요청 True, 요청 2회
- 원인 없음 설명: "All candidate differences are below 1.5x the normal spread (all are around 0.6x to 0.75x of the normal spread baseline). The overall yield drop is driven by product mix shifts and baseline variations rather than a specific equipment excursion."
- 교란 메모: "Product mix changes over time account for shifts in the aggregate yield metrics."
- 한계: "Candidate yield differences do not exceed normal pre-alert spreads."
- 추가 확인 C1 time_trend: error in time_trend: entity_id is required
- 추가 확인 C2 commonality_scan: commonality_scan (level all, product all): S27-T2 [tool] n 3350 low 0.618 vs 0.517 median diff -0.014 q 2.88e-41; S24-T3 [tool] n 3300 low 0.645 vs 0.499 median diff -0.013 q 1.17e-30; S34-T4 [tool] n

기록: `runs/2026-09-29-gemini-3.5-flash-lite-v2-c2ed5083/scn_7002.json`

### scn_7004 — F6 (효과 large, stickiness low, 알림 warning)

**정답**
- F6: step S21, S21-T2-C1, S21-T2, 시작 2026-01-12T06:43:41, δ 0.12
  - S21-T2-C1: 전체 공통성 스캔 55위 / 347 (q 0.0302, 중앙값 차이 -0.008)
  - S21-T2: 전체 공통성 스캔 20위 / 347 (q 0.000115, 중앙값 차이 -0.010)
  - 근거 묶음에 나온 줄: 0개

**에이전트 최종 보고서**
- 판정 no_equipment_cause, 추가 확인 요청 True, 요청 2회
- 원인 없음 설명: "The product-adjusted yield change for all products combined is 0.002, and with the product mix held at pre-alert shares it is -0.001 (E6). Yields remained stable within each individual product over time as confirmed by the check results (C1), indicating that the alert was triggered purely by product"
- 교란 메모: "None."
- 한계: "Evidence confirms that wafer yield variations are entirely attributable to product mix changes."
- 추가 확인 C1 product_mix: product_mix (by test day: P2 share, mean yield per product): 01-03 P2 share 0.389 P1 0.933 P2 0.889; 01-04 P2 share 0.542 P1 0.927 P2 0.886; 01-05 P2 share 0.478 P1 0.924 P2 0.890; 01-06 P2 share 0.33

기록: `runs/2026-09-29-gemini-3.5-flash-lite-v2-c2ed5083/scn_7004.json`

### scn_7005 — F0b (효과 -, stickiness low, 알림 alarm)

**정답**
- 설비 원인 없음

**에이전트 최종 보고서**
- 판정 cause_found, 추가 확인 요청 True, 요청 2회
- #1 chamber S01-T2-C3 (step S01), 신뢰도 0.7, 시작 2026-01-03T10:30
  - 인용 E17: "S01-T2-C3 shows low-yield rate 0.190 vs rest 0.144, median diff -0.018, q 1.04e-10, 1.15x normal spread."
    - 원문 `E17 [chamber] S01-T2-C3 (step S01): n 357, low-yield rate 0.190 vs rest 0.144, median diff -0.018, OR 1.507, p 2.39e-12, q 1.04e-10, 1.15x normal spread`
  - 인용 E29: "S01-T2-C3 change point at 2026-01-03T10:30, mean before -0.001, after -0.015."
    - 원문 `E29 S01-T2-C3: change point 2026-01-03T10:30, mean before -0.001, after -0.015, permutation p 0.265, lots 83`
- 교란 메모: "The difference is slightly above the normal spread (1.15x), but supported by a distinct change point and event timing."
- 한계: "Permutation p-value for the change point is not highly significant."
- 추가 확인 C1 commonality_scan: commonality_scan (level chamber, product all): S16-T1-C1 [chamber] n 780 low 0.199 vs 0.139 median diff -0.011 q 8.64e-12; S01-T2-C3 [chamber] n 357 low 0.190 vs 0.144 median diff -0.018 q 1.04e-10; S

기록: `runs/2026-09-29-gemini-3.5-flash-lite-v2-c2ed5083/scn_7005.json`

### scn_7007 — F2 (효과 large, stickiness high, 알림 alarm)

**정답**
- F2: step S35, S35-T4, S35-R2, 시작 2026-01-11T08:39:59, δ 0.12
  - S35-T4: 전체 공통성 스캔 1위 / 347 (q 1.47e-172, 중앙값 차이 -0.038)
  - S35-R2: 전체 공통성 스캔 스캔 대상 아님
  - 근거 묶음에 나온 줄: 5개
    - `E14 [tool] S35-T4 (step S35): n 3875, low-yield rate 0.380 vs rest 0.094, median diff -0.038, OR 10.505, p 4.23e-175, q 1.47e-172, 1.89x normal spread`
    - `E32 S35-T4: change point 2026-01-11T09:02, mean before -0.003, after -0.056, permutation p 0.002, lots 230`
    - `E37 S35-T4: none`
    - `E39 phi: S21-T3~S25-T3 -0.029, S21-T3~S21-T3-C1 0.701, S21-T3~S35-T4 0.059, S21-T3~S25-T3-C1 -0.020, S25-T3~S21-T3-C1 -0.020, S25-T3~S35-T4 0.103, S25-T3~S25-T3-C1 0.704, S21-T3-C1~S35-T4 0.041, S21-T3-C1~S25-T3-C1 -0.01`
    - `E43 S35: interaction p 0.000; most deviant cell S35-T4 x S35-R2 n 1575, mean 0.787 vs additive 0.814`

**에이전트 최종 보고서**
- 판정 cause_found, 추가 확인 요청 True, 요청 2회
- #1 tool_recipe S25-T3 × S25-R2 (step S25), 신뢰도 0.8, 시작 2026-01-11T16:03
  - 인용 E13: "Tool S25-T3 shows significant low-yield rate 0.456 vs rest 0.236, median diff -0.051 (2.52x normal spread)."
    - 원문 `E13 [tool] S25-T3 (step S25): n 250, low-yield rate 0.456 vs rest 0.236, median diff -0.051, OR 16.249, p 4.82e-37, q 4.18e-35, 2.52x normal spread`
  - 인용 E42: "Significant tool x recipe interaction at step S25 (p = 6.96e-14), most deviant cell S25-T3 x S25-R2 with mean yield 0.764."
    - 원문 `E42 S25: interaction p 6.96e-14; most deviant cell S25-T3 x S25-R2 n 75, mean 0.764 vs additive 0.793`
- #2 tool S21-T3 (step S21), 신뢰도 0.65, 시작 2026-01-10T22:04
  - 인용 E12: "Tool S21-T3 shows low-yield rate 0.594 vs rest 0.235, median diff -0.059 (2.89x normal spread)."
    - 원문 `E12 [tool] S21-T3 (step S21): n 175, low-yield rate 0.594 vs rest 0.235, median diff -0.059, OR 3.353, p 8.47e-11, q 1.01e-09, 2.89x normal spread`
  - 인용 E34: "Change point preceded by PM S21-T3 and chamber PMs on S21-T3-C1 and S21-T3-C2 within 2 days."
    - 원문 `E34 S21-T3: PM S21-T3 at 2026-01-09T12:19; CHAMBER_PM S21-T3-C2 at 2026-01-10T03:46; CHAMBER_PM S21-T3-C1 at 2026-01-11T03:22`
- #3 tool S35-T4 (step S35), 신뢰도 0.6, 시작 2026-01-11T09:02
  - 인용 E14: "Tool S35-T4 shows low-yield rate 0.380 vs rest 0.094, median diff -0.038 (1.89x normal spread)."
    - 원문 `E14 [tool] S35-T4 (step S35): n 3875, low-yield rate 0.380 vs rest 0.094, median diff -0.038, OR 10.505, p 4.23e-175, q 1.47e-172, 1.89x normal spread`
  - 인용 E32: "Change point detected at 2026-01-11T09:02 with significant permutation p = 0.002."
    - 원문 `E32 S35-T4: change point 2026-01-11T09:02, mean before -0.003, after -0.056, permutation p 0.002, lots 230`
- 교란 메모: "High overlap (phi > 0.7) between tools and specific chambers (e.g., S25-T3 and S25-T3-C1). Also, product mix shift toward P2 contributes to overall fab yield degradation."
- 한계: "Some change point permutation p-values are above 0.05, suggesting gradual drift rather than sharp step changes."
- 추가 확인 C1 interaction_test: interaction_test S25: interaction p 6.96e-14; most deviant cell S25-T3 x S25-R2 n 75 mean 0.764 vs additive 0.793
- 추가 확인 C2 time_trend: error in time_trend: entity_id is required

기록: `runs/2026-09-29-gemini-3.5-flash-lite-v2-c2ed5083/scn_7007.json`

### scn_7017 — F0a (효과 -, stickiness low, 알림 warning)

**정답**
- 설비 원인 없음

**에이전트 최종 보고서**
- 판정 cause_found, 추가 확인 요청 True, 요청 2회
- #1 tool S21-T4 (step S21), 신뢰도 0.75, 시작 2026-01-16T23:30
  - 인용 E12: "S21-T4 shows significant low-yield rate increase (0.253 vs 0.068 rest) with median diff -0.021 and 1.35x normal spread."
    - 원문 `E12 [tool] S21-T4 (step S21): n 400, low-yield rate 0.253 vs rest 0.068, median diff -0.021, OR 5.250, p 6.63e-19, q 2.30e-16, 1.35x normal spread`
  - 인용 E29: "Change point detected at 2026-01-16T23:30 with post-change mean drop to -0.024 (p = 0.008)."
    - 원문 `E29 S21-T4: change point 2026-01-16T23:30, mean before -8.30e-04, after -0.024, permutation p 0.008, lots 63`
- #2 chamber S21-T4-C3 (step S21), 신뢰도 0.7, 시작 2026-01-14T02:37
  - 인용 E17: "S21-T4-C3 shows high low-yield rate (0.304 vs 0.089 rest) with median diff -0.024 and 1.23x normal spread."
    - 원문 `E17 [chamber] S21-T4-C3 (step S21): n 102, low-yield rate 0.304 vs rest 0.089, median diff -0.024, OR 5.682, p 1.81e-07, q 3.31e-06, 1.23x normal spread`
  - 인용 E30: "Change point identified at 2026-01-14T02:37 with mean drop to -0.022 (p = 0.006)."
    - 원문 `E30 S21-T4-C3: change point 2026-01-14T02:37, mean before 0.002, after -0.022, permutation p 0.006, lots 63`
- #3 tool S18-T2 (step S18), 신뢰도 0.6, 시작 2026-01-19T10:42
  - 인용 E13: "S18-T2 shows elevated low-yield rate (0.187 vs 0.072) with median diff -0.014."
    - 원문 `E13 [tool] S18-T2 (step S18): n 550, low-yield rate 0.187 vs rest 0.072, median diff -0.014, OR 3.200, p 1.65e-14, q 1.14e-12, 0.92x normal spread`
  - 인용 E32: "Change point detected on 2026-01-19T10:42 with drop to -0.047 (p = 0.002)."
    - 원문 `E32 S18-T2: change point 2026-01-19T10:42, mean before -0.008, after -0.047, permutation p 0.002, lots 79`
- 교란 메모: "S21-T4 and its chambers S21-T4-C3 and S21-T4-C4 share substantial wafer overlap (phi ~0.47), creating collinearity between the tool and chamber level findings."
- 한계: "Normal spread comparisons indicate ratios are near the 1.0x-1.35x threshold, requiring cautious interpretation."
- 추가 확인 C1 commonality_scan: commonality_scan (level chamber, product all): S19-T3-C1 [chamber] n 436 low 0.140 vs 0.089 median diff -0.015 q 4.38e-11; S09-T2-C3 [chamber] n 199 low 0.151 vs 0.093 median diff -0.017 q 2.23e-10; S
- 추가 확인 C2 time_trend: error in time_trend: entity_id is required
- 추가 확인 C3 confounding_check: confounding_check: S21-T4 vs S21-T4-C3 phi 0.473: a_only n 298 resid -0.016 low 0.235, b_only n 0 resid n/a low n/a, both n 102 resid -0.021 low 0.304, neither n 2125 resid 0.002 low 0.068; S21-T4 vs 

기록: `runs/2026-09-29-gemini-3.5-flash-lite-v2-c2ed5083/scn_7017.json`
