# 데이터 컬럼 명세서 (Data Column Descriptions)

이 문서는 모델링에 사용되는 데이터(`data_for_model.csv`)의 컬럼들을 설명합니다.

## 1. 고객 정보 (Client Data)
| 컬럼명 | 설명 |
|---|---|
| `id` | 고객 식별자 |
| `channel_sales` | 판매 채널 코드 |
| `cons_12m` | 지난 12개월 전기 소비량 |
| `cons_gas_12m` | 지난 12개월 가스 소비량 |
| `cons_last_month` | 지난달 전기 소비량 |
| `date_activ` | 계약 활성화 날짜 |
| `date_end` | 계약 등록 마감 날짜 |
| `date_modif_prod` | 마지막 제품 수정 날짜 |
| `date_renewal` | 다음 계약 갱신 날짜 |
| `forecast_cons_12m` | 향후 12개월 예측 전기 소비량 |
| `forecast_cons_year` | 향후 1년 예측 소비량 |
| `forecast_discount_energy` | 에너지 할인 예측값 |
| `forecast_meter_rent_12m` | 향후 12개월 예측 계량기 임대료 |
| `forecast_price_energy_off_peak` | 비피크 시간대 에너지 예측 가격 |
| `forecast_price_energy_peak` | 피크 시간대 에너지 예측 가격 |
| `forecast_price_pow_off_peak` | 비피크 시간대 전력 예측 가격 |
| `has_gas` | 가스 사용 여부 (T/F) |
| `imp_cons` | 현재 지불된 소비량 |
| `margin_gross_pow_ele` | 전력 총 마진 |
| `margin_net_pow_ele` | 전력 순 마진 |
| `nb_prod_act` | 활성 제품 수 |
| `net_margin` | 총 순 마진 |
| `num_years_antig` | 고객 유지 연수 (Antiquity) |
| `origin_up` | 계약 체결 경로 (캠페인 등) |
| `pow_max` | 계약 전력량 (Power) |
| `churn` | **이탈 여부 (Target Variable)** (1: 이탈, 0: 유지) |

## 2. 가격 피처 (Price Features) - 파생 변수
가격 데이터(`price_data.csv`)를 고객 ID별로 집계하여 생성한 변수들입니다.
통계량 접미사: `_mean` (평균), `_max` (최대), `_min` (최소), `_std` (표준편차)

| 컬럼명 패턴 | 설명 |
|---|---|
| `price_off_peak_var_*` | 비피크 시간대 전력량 요금(변동) 통계 |
| `price_peak_var_*` | 피크 시간대 전력량 요금(변동) 통계 |
| `price_off_peak_fix_*` | 비피크 시간대 전력 요금(고정) 통계 |
| `price_peak_fix_*` | 피크 시간대 전력 요금(고정) 통계 |

## 3. 신규 생성 피처 (New Features)
| 컬럼명 | 설명 |
|---|---|
| `off_peak_diff` | 12월 가격 - 1월 가격 차이 (비피크 변동 요금 기준) |
| `max_volatility` | 연중 최대 가격 변동폭 (`price_off_peak_var`의 Max - Min) |
| `price_shock_encoded` | 가격 충격 그룹 인코딩 <br> `0`: Price Down (가격 하락) <br> `1`: No Change (변동 없음) <br> `2`: Price Up (가격 상승) |
