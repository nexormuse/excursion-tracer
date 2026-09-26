# 실패 사례 분석 (test, agent)

선정 규칙: 원인 시나리오는 Hit@3 엄격 실패, 원인 없는 시나리오는 오경보를 실패로 보고, 실패 비율이 높은 원인 유형부터 5개 유형을 골라 유형마다 번호가 가장 작은 실패 시나리오를 택했다.

| 유형 | 실패 | 전체 |
|---|---|---|
| F0a | 11 | 12 |
| F0b | 6 | 12 |
| F1 | 11 | 24 |
| F2 | 12 | 18 |
| F3 | 3 | 18 |
| F4 | 13 | 18 |
| F5 | 18 | 18 |

<!-- manual:start -->
## 사례별 판단 지점 요약 (기록의 사실만)

| 시나리오 | 유형 | 정답 | 에이전트 1순위 | 판단이 갈린 지점 (근거 묶음 원문) |
|---|---|---|---|---|
| scn_5003 | F5 | 레시피 S09-R1 (공통 레시피 단계) | no_equipment_cause | S09-R1은 공통성 스캔 대상이 아니고 근거 묶음 어느 줄에도 나오지 않음. 근거 묶음의 이벤트 항목은 상위 후보 변화점 ±2일만 보여 주며, 상위 5개 후보의 변화점 p는 0.238~0.878(E18~E22). 보고서는 이 p값들을 근거로 원인 없음 판정. 추가 확인 요청 없음 |
| scn_5005 | F0a | 설비 원인 없음 | chamber S15-T4-C1, 신뢰도 0.85 | 인용: E12(중앙값 차이 -0.018, q 3.30e-16), E25(PM). 같은 후보의 변화점 줄 E20은 permutation p 0.549이고 보고서는 E20을 인용하지 않음 |
| scn_5001 | F4 (F2+F3) | S34-T2 × S34-R1, S17-T2 | tool S34-T2, 2순위 tool S17-T2 | 두 정답 설비를 1·2순위로 냈으나 S34-T2를 설비(tool)로만 내고 레시피를 비워 F2 엄격 규칙(설비+레시피)에 미달. 상호작용 줄 E30의 가장 벗어난 셀은 S34-T1 × S34-R2 |
| scn_5019 | F2 | S37-T1 × S37-R1 | tool S37-T1 | 상호작용 줄 E29가 "most deviant cell S37-T1 x S37-R1"을 보여 줬으나 보고서는 E7·E18만 인용하고 설비(tool)로 제출 |
| scn_5011 | F0b | 설비 원인 없음 (P2 비중 변화) | chamber S17-T2-C2, 신뢰도 0.85 | 인용: E12(중앙값 차이 -0.013), E19(변화점 permutation p 0.345). 제품 구성 줄 E31은 P2 비중 기준 0.453 → 알림 구간 0.664였고 보고서의 원인 없음 설명·한계는 비어 있음. 1회차 첫 응답은 JSON이 끊겨 재요청 |

추가 원문:
- scn_5003 `E18 S10-T1: change point 2026-01-03T15:07, mean before -0.011, after -0.020, permutation p 0.275, lots 220` … `E22 S14-T5: change point 2026-01-02T19:55, mean before -0.012, after -0.018, permutation p 0.878, lots 169`
- scn_5005 `E20 S15-T4-C1: change point 2026-01-02T23:00, mean before -0.005, after -0.022, permutation p 0.549, lots 90`
- scn_5001 `E30 S34: interaction p 1.13e-133; most deviant cell S34-T1 x S34-R2 n 1250, mean 0.843 vs additive 0.861`
- scn_5011 `E31 P2 share of measured wafers: baseline days mean 0.453, alert window mean 0.664, daily min 0.273, max 0.808`
<!-- manual:end -->

### scn_5003 — F5 (효과 large, stickiness high, 알림 alarm)

**정답**
- F5: step S09, S09-R1, 시작 2026-01-10T20:00:50, δ 0.12
  - S09-R1: 전체 공통성 스캔 스캔 대상 아님
  - 근거 묶음에 나온 줄: 0개

**에이전트 최종 보고서**
- 판정 no_equipment_cause, 추가 확인 요청 False, 요청 1회
- 원인 없음 설명: "Although multiple tools show statistical significance in commonality scans (e.g., S10-T1 and S11-T2), none of the top candidates exhibit clear or statistically significant change points associated with maintenance events (permutation p-values > 0.2). Furthermore, product mix variations remain minor "
- 교란 메모: "High correlation (phi > 0.8) between S10-T1 and S11-T2 reflects overlapping wafer routing rather than a true isolated equipment failure."
- 한계: "The excursion impacts multiple tools simultaneously with weak temporal correlation to specific tool events."

기록: `runs/2026-09-26-gemini-3.5-flash-lite-87ff5056/scn_5003.json`

### scn_5005 — F0a (효과 -, stickiness low, 알림 alarm)

**정답**
- 설비 원인 없음

**에이전트 최종 보고서**
- 판정 cause_found, 추가 확인 요청 False, 요청 1회
- #1 chamber S15-T4-C1 (step S15), 신뢰도 0.85, 시작 2026-01-02T23:00
  - 인용 E12: "S15-T4-C1 shows highest effect size with low-yield rate 0.190 vs rest 0.106 and median diff -0.018 (p 2.80e-18)"
    - 원문 `E12 [chamber] S15-T4-C1 (step S15): n 443, low-yield rate 0.190 vs rest 0.106, median diff -0.018, OR 2.176, p 2.80e-18, q 3.30e-16`
  - 인용 E29: "Stratification shows wafers processed on both S15-T4 and C1 have the highest low-yield rate of 0.190 and mean residual -0.015"
    - 원문 `E29 S15-T4 vs S15-T4-C1 (phi 0.526): a_only n 882 mean resid -0.005 low 0.147, b_only n 0 mean resid n/a low n/a, both n 443 mean resid -0.015 low 0.190, neither n 4250 mean resid 0.004 low 0.098`
  - 인용 E25: "PM event on S15-T4 occurred at 2026-01-03T07:29, closely preceding the change point"
    - 원문 `E25 S15-T4-C1: PM S15-T4 at 2026-01-03T07:29`
- #2 tool S15-T4 (step S15), 신뢰도 0.7, 시작 2026-01-03T12:59
  - 인용 E8: "S15-T4 has a high low-yield rate of 0.162 vs rest 0.098 and an odds ratio of 1.914"
    - 원문 `E8 [tool] S15-T4 (step S15): n 1325, low-yield rate 0.162 vs rest 0.098, median diff -0.014, OR 1.914, p 1.84e-22, q 3.25e-20`
  - 인용 E24: "A PM event was logged for S15-T4 at 2026-01-03T07:29, right before the change point"
    - 원문 `E24 S15-T4: PM S15-T4 at 2026-01-03T07:29`
- 교란 메모: "S15-T4 and S15-T4-C1 are highly confounded (phi 0.526), but stratification confirms that the specific chamber C1 combination drives the worst yield performance."
- 한계: "Product mix P2 share fluctuates during the window, but the strong localized effect on S15-T4-C1 isolates the equipment cause."

기록: `runs/2026-09-26-gemini-3.5-flash-lite-87ff5056/scn_5005.json`

### scn_5001 — F4 (효과 medium, stickiness low, 알림 alarm)

**정답**
- F2: step S34, S34-T2, S34-R1, 시작 2026-01-11T04:59:51, δ 0.07
  - S34-T2: 전체 공통성 스캔 1위 / 354 (q 5.79e-128, 중앙값 차이 -0.033)
  - S34-R1: 전체 공통성 스캔 스캔 대상 아님
  - 근거 묶음에 나온 줄: 4개
    - `E7 [tool] S34-T2 (step S34): n 3525, low-yield rate 0.379 vs rest 0.169, median diff -0.033, OR 2.801, p 1.64e-130, q 5.79e-128`
    - `E18 S34-T2: change point 2026-01-11T07:46, mean before 0.003, after -0.042, permutation p 0.002, lots 211`
    - `E23 S34-T2: PM S34-T2 at 2026-01-13T05:48`
    - `E28 phi: S34-T2~S17-T2 -0.063, S34-T2~S17-T2-C1 -0.043, S34-T2~S15-T1 -0.007, S34-T2~S16-T2 -0.012, S17-T2~S17-T2-C1 0.634, S17-T2~S15-T1 0.105, S17-T2~S16-T2 -0.087, S17-T2-C1~S15-T1 0.066, S17-T2-C1~S16-T2 -0.056, S15-`
- F3: step S17, S17-T2, 시작 2026-01-09T22:51:59, δ 0.07
  - S17-T2: 전체 공통성 스캔 2위 / 354 (q 7.34e-81, 중앙값 차이 -0.026)
  - 근거 묶음에 나온 줄: 6개
    - `E8 [tool] S17-T2 (step S17): n 2250, low-yield rate 0.408 vs rest 0.214, median diff -0.026, OR 2.601, p 4.15e-83, q 7.34e-81`
    - `E19 S17-T2: change point 2026-01-11T12:55, mean before 0.003, after -0.041, permutation p 0.002, lots 140`
    - `E24 S17-T2: RECIPE_CHANGE S17-R1 at 2026-01-09T22:05`
    - `E25 S17-T2-C1: PM S17-T2 at 2026-01-13T16:18`
    - `E28 phi: S34-T2~S17-T2 -0.063, S34-T2~S17-T2-C1 -0.043, S34-T2~S15-T1 -0.007, S34-T2~S16-T2 -0.012, S17-T2~S17-T2-C1 0.634, S17-T2~S15-T1 0.105, S17-T2~S16-T2 -0.087, S17-T2-C1~S15-T1 0.066, S17-T2-C1~S16-T2 -0.056, S15-`
    - `E29 S17-T2 vs S17-T2-C1 (phi 0.634): a_only n 1121 mean resid -0.014 low 0.390, b_only n 0 mean resid n/a low n/a, both n 1129 mean resid -0.020 low 0.426, neither n 4500 mean resid 0.008 low 0.214`

**에이전트 최종 보고서**
- 판정 cause_found, 추가 확인 요청 False, 요청 1회
- #1 tool S34-T2 (step S34), 신뢰도 0.85, 시작 2026-01-11T07:46
  - 인용 E7: "S34-T2 high low-yield rate (0.379 vs rest 0.169, OR 2.801, p 1.64e-130)"
    - 원문 `E7 [tool] S34-T2 (step S34): n 3525, low-yield rate 0.379 vs rest 0.169, median diff -0.033, OR 2.801, p 1.64e-130, q 5.79e-128`
  - 인용 E18: "Change point detected at 2026-01-11T07:46 with mean drop of -0.042 (permutation p 0.002)"
    - 원문 `E18 S34-T2: change point 2026-01-11T07:46, mean before 0.003, after -0.042, permutation p 0.002, lots 211`
- #2 tool S17-T2 (step S17), 신뢰도 0.8, 시작 2026-01-11T12:55
  - 인용 E8: "S17-T2 high low-yield rate (0.408 vs rest 0.214, OR 2.601, p 4.15e-83)"
    - 원문 `E8 [tool] S17-T2 (step S17): n 2250, low-yield rate 0.408 vs rest 0.214, median diff -0.026, OR 2.601, p 4.15e-83, q 7.34e-81`
  - 인용 E19: "Change point detected at 2026-01-11T12:55 with mean drop of -0.041 (permutation p 0.002)"
    - 원문 `E19 S17-T2: change point 2026-01-11T12:55, mean before 0.003, after -0.041, permutation p 0.002, lots 140`
- #3 chamber S17-T2-C1 (step S17), 신뢰도 0.7, 시작 2026-01-13T16:22
  - 인용 E12: "S17-T2-C1 high low-yield rate (0.426 vs rest 0.249, OR 2.286, p 4.70e-45)"
    - 원문 `E12 [chamber] S17-T2-C1 (step S17): n 1129, low-yield rate 0.426 vs rest 0.249, median diff -0.025, OR 2.286, p 4.70e-45, q 5.54e-43`
  - 인용 E20: "Change point detected at 2026-01-13T16:22 with mean drop of -0.045 (permutation p 0.002)"
    - 원문 `E20 S17-T2-C1: change point 2026-01-13T16:22, mean before -0.003, after -0.045, permutation p 0.002, lots 140`
- 교란 메모: "S17-T2 and chamber S17-T2-C1 share high phi (0.634), indicating that the tool-level excursion is heavily driven by chamber C1."
- 한계: "Product mix shift shows minor increase in P2 share during the alert window, but commonality scans isolate specific tools with very large effect sizes."

기록: `runs/2026-09-26-gemini-3.5-flash-lite-87ff5056/scn_5001.json`

### scn_5019 — F2 (효과 small, stickiness low, 알림 warning)

**정답**
- F2: step S37, S37-T1, S37-R1, 시작 2026-01-09T13:32:24, δ 0.04
  - S37-T1: 전체 공통성 스캔 1위 / 354 (q 9.53e-37, 중앙값 차이 -0.025)
  - S37-R1: 전체 공통성 스캔 스캔 대상 아님
  - 근거 묶음에 나온 줄: 5개
    - `E7 [tool] S37-T1 (step S37): n 525, low-yield rate 0.223 vs rest 0.122, median diff -0.025, OR 2.140, p 2.69e-39, q 9.53e-37`
    - `E18 S37-T1: change point 2026-01-10T09:40, mean before -0.004, after -0.024, permutation p 0.006, lots 108`
    - `E23 S37-T1: none`
    - `E28 phi: S37-T1~S36-T1 0.432, S37-T1~S16-T3 0.143, S37-T1~S28-T1 0.120, S37-T1~S27-T2 0.166, S36-T1~S16-T3 0.094, S36-T1~S28-T1 0.067, S36-T1~S27-T2 0.215, S16-T3~S28-T1 0.056, S16-T3~S27-T2 0.206, S28-T1~S27-T2 0.263`
    - `E29 S37: interaction p 7.69e-15; most deviant cell S37-T1 x S37-R1 n 300, mean 0.870 vs additive 0.880`

**에이전트 최종 보고서**
- 판정 cause_found, 추가 확인 요청 False, 요청 1회
- #1 tool S37-T1 (step S37), 신뢰도 0.85, 시작 2026-01-10T09:40
  - 인용 E7: "S37-T1 has high low-yield rate 0.223 vs rest 0.122, OR 2.140, p 2.69e-39, q 9.53e-37"
    - 원문 `E7 [tool] S37-T1 (step S37): n 525, low-yield rate 0.223 vs rest 0.122, median diff -0.025, OR 2.140, p 2.69e-39, q 9.53e-37`
  - 인용 E18: "Change point at 2026-01-10T09:40 with mean before -0.004, after -0.024, permutation p 0.006"
    - 원문 `E18 S37-T1: change point 2026-01-10T09:40, mean before -0.004, after -0.024, permutation p 0.006, lots 108`
- #2 tool S36-T1 (step S36), 신뢰도 0.65, 시작 2026-01-16T12:09
  - 인용 E8: "S36-T1 shows low-yield rate 0.232 vs rest 0.107, OR 2.418, p 8.67e-30"
    - 원문 `E8 [tool] S36-T1 (step S36): n 725, low-yield rate 0.232 vs rest 0.107, median diff -0.022, OR 2.418, p 8.67e-30, q 1.54e-27`
  - 인용 E19: "Change point detected on 2026-01-16T12:09 with after-mean -0.021"
    - 원문 `E19 S36-T1: change point 2026-01-16T12:09, mean before -0.006, after -0.021, permutation p 0.078, lots 135`
- 교란 메모: "Moderate overlap (phi 0.432) between S37-T1 and S36-T1, but S37-T1 has a statistically significant permutation p-value and earlier onset."
- 한계: "Product mix variations exist but do not fully account for the strong tool-specific yield divergence."

기록: `runs/2026-09-26-gemini-3.5-flash-lite-87ff5056/scn_5019.json`

### scn_5011 — F0b (효과 -, stickiness low, 알림 alarm)

**정답**
- 설비 원인 없음

**에이전트 최종 보고서**
- 판정 cause_found, 추가 확인 요청 False, 요청 2회
- #1 chamber S17-T2-C2 (step S17), 신뢰도 0.85, 시작 2026-01-06T03:08
  - 인용 E12: "S17-T2-C2 chamber low-yield rate 0.178 vs rest 0.106, median diff -0.013, OR 1.831"
    - 원문 `E12 [chamber] S17-T2-C2 (step S17): n 1071, low-yield rate 0.178 vs rest 0.106, median diff -0.013, OR 1.831, p 1.75e-20, q 3.09e-18`
  - 인용 E19: "Change point at 2026-01-06T03:08 with mean after -0.014"
    - 원문 `E19 S17-T2-C2: change point 2026-01-06T03:08, mean before -0.004, after -0.014, permutation p 0.345, lots 147`
- 교란 메모: "S17-T2 and S17-T2-C2 are highly confounded with a phi of 0.627, but stratification confirms the chamber C2 drives the larger yield degradation."

기록: `runs/2026-09-26-gemini-3.5-flash-lite-87ff5056/scn_5011.json`
