# 대표 시나리오 추적 기록

세트 test, 실행 `runs/2026-09-26-gemini-3.5-flash-lite-87ff5056`. 선정 규칙 (results/showcase_selection.json):

성공 1건(에이전트 Hit@1 엄격 적중, 기준선 Hit@3 엄격 실패인 것 중 번호가 가장 작은 것, 없으면 에이전트 Hit@1 엄격 적중 중 가장 작은 것), 실패 1건(원인 시나리오에서 에이전트 Hit@3 느슨 실패 중 가장 작은 것), F5 1건(F5 중 번호가 가장 작은 것, 앞의 두 건과 겹치면 다음 것).

| 구분 | 시나리오 | 원인 유형 | 에이전트 Hit@1 엄격 | 에이전트 Hit@3 느슨 | 기준선 Hit@3 엄격 | 파일 |
|---|---|---|---|---|---|---|
| 성공 | scn_5029 | F2 | True | True | False | traces/scn_5029.md |
| 실패 | scn_5002 | F1 | False | False | True | traces/scn_5002.md |
| F5 | scn_5003 | F5 | False | False | False | traces/scn_5003.md |

fig6_case_timeline.png는 성공 사례를 그린다.
