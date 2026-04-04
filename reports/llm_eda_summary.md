# Higgsfield Hackathon Data - Automated EDA Report

# EDA Summary for 'TRAIN' Split

## Table: Users

**Shape:** 90000 rows, 2 columns

*No missing values.*

### Numerical & Categorical Summary
|        | user_id                                   | churn_status   |
|:-------|:------------------------------------------|:---------------|
| count  | 90000                                     | 90000          |
| unique | 90000                                     | 3              |
| top    | user_524a96be-1872-40fd-b6c5-38cee925bbe4 | not_churned    |
| freq   | 1                                         | 45000          |

### Top Value Distributions (Categorical)
**churn_status**
| churn_status   |   Count |
|:---------------|--------:|
| not_churned    |   45000 |
| invol_churn    |   22500 |
| vol_churn      |   22500 |

## Table: Generations

*Dataset missing or skipped.*

## Table: Properties

**Shape:** 90000 rows, 4 columns

### Missing Values
|              |   Missing Count |
|:-------------|----------------:|
| country_code |              20 |

### Numerical & Categorical Summary
|        | user_id                                   | subscription_start_date          | subscription_plan   | country_code   |
|:-------|:------------------------------------------|:---------------------------------|:--------------------|:---------------|
| count  | 90000                                     | 90000                            | 90000               | 89980          |
| unique | 90000                                     |                                  | 4                   | 218            |
| top    | user_524a96be-1872-40fd-b6c5-38cee925bbe4 |                                  | Higgsfield Ultimate | US             |
| freq   | 1                                         |                                  | 48000               | 17007          |
| mean   |                                           | 2023-10-09 06:03:23.517211+00:00 |                     |                |
| min    |                                           | 2023-08-26 00:00:07+00:00        |                     |                |
| 25%    |                                           | 2023-09-17 15:15:29.250000+00:00 |                     |                |
| 50%    |                                           | 2023-10-14 06:31:02+00:00        |                     |                |
| 75%    |                                           | 2023-10-30 17:23:05+00:00        |                     |                |
| max    |                                           | 2023-11-09 23:59:45+00:00        |                     |                |

### Top Value Distributions (Categorical)
**subscription_start_date**
| subscription_start_date   |   Count |
|:--------------------------|--------:|
| 2023-11-01 17:39:14+00:00 |       3 |
| 2023-10-15 13:44:26+00:00 |       3 |
| 2023-09-16 22:23:24+00:00 |       3 |
| 2023-11-02 01:49:01+00:00 |       3 |
| 2023-11-07 11:03:40+00:00 |       3 |
| 2023-10-30 22:55:42+00:00 |       3 |
| 2023-09-14 19:41:24+00:00 |       3 |
| 2023-08-29 04:29:05+00:00 |       2 |
| 2023-09-16 17:39:13+00:00 |       2 |
| 2023-09-22 00:26:26+00:00 |       2 |

**subscription_plan**
| subscription_plan   |   Count |
|:--------------------|--------:|
| Higgsfield Ultimate |   48000 |
| Higgsfield Basic    |   24000 |
| Higgsfield Creator  |   12000 |
| Higgsfield Pro      |    6000 |

**country_code**
| country_code   |   Count |
|:---------------|--------:|
| US             |   17007 |
| IN             |    4555 |
| DE             |    4281 |
| JP             |    3700 |
| GB             |    3613 |
| FR             |    3190 |
| KR             |    3036 |
| BR             |    2687 |
| TR             |    2379 |
| IT             |    2223 |

## Table: Purchases

**Shape:** 96424 rows, 5 columns

*No missing values.*

### Numerical & Categorical Summary
|        | user_id                                   | transaction_id                          | purchase_time                    | purchase_type       |   purchase_amount_dollars |
|:-------|:------------------------------------------|:----------------------------------------|:---------------------------------|:--------------------|--------------------------:|
| count  | 96424                                     | 96424                                   | 96424                            | 96424               |                96424      |
| unique | 77132                                     | 96424                                   |                                  | 6                   |                           |
| top    | user_d6cfd073-3008-45d3-91bf-f09903ae9f29 | ch_a1e98b75-2dce-45d8-9e6d-a71d2f35568e |                                  | Subscription Create |                           |
| freq   | 52                                        | 1                                       |                                  | 77002               |                           |
| mean   |                                           |                                         | 2023-10-13 10:53:43.773542+00:00 |                     |                   35.2185 |
| min    |                                           |                                         | 2023-08-26 00:00:07+00:00        |                     |                    0.56   |
| 25%    |                                           |                                         | 2023-09-23 18:54:45+00:00        |                     |                   10      |
| 50%    |                                           |                                         | 2023-10-19 19:39:27+00:00        |                     |                   35      |
| 75%    |                                           |                                         | 2023-11-01 15:55:12.250000+00:00 |                     |                   35      |
| max    |                                           |                                         | 2023-11-24 17:51:37+00:00        |                     |                  900      |
| std    |                                           |                                         |                                  |                     |                   26.4854 |

### Top Value Distributions (Categorical)
**purchase_time**
| purchase_time             |   Count |
|:--------------------------|--------:|
| 2023-11-07 11:03:40+00:00 |       3 |
| 2023-11-05 20:44:23+00:00 |       3 |
| 2023-09-14 19:41:24+00:00 |       3 |
| 2023-10-15 13:44:26+00:00 |       3 |
| 2023-11-01 17:39:14+00:00 |       3 |
| 2023-10-30 22:55:42+00:00 |       3 |
| 2023-11-02 01:49:01+00:00 |       3 |
| 2023-11-07 19:11:16+00:00 |       3 |
| 2023-09-26 18:23:40+00:00 |       2 |
| 2023-10-15 15:23:58+00:00 |       2 |

**purchase_type**
| purchase_type       |   Count |
|:--------------------|--------:|
| Subscription Create |   77002 |
| Subscription Update |    7812 |
| Credits package     |    7306 |
| Upsell              |    4122 |
| Gift                |     134 |
| Reactivation        |      48 |

**purchase_amount_dollars**
|   purchase_amount_dollars |   Count |
|--------------------------:|--------:|
|                        35 |   36242 |
|                         9 |   20607 |
|                        89 |   10024 |
|                        39 |    7628 |
|                        29 |    6293 |
|                        20 |    2317 |
|                         5 |    1893 |
|                        10 |    1702 |
|                        26 |    1580 |
|                        80 |    1125 |

## Table: Quizzes

**Shape:** 90004 rows, 9 columns

### Missing Values
|               |   Missing Count |
|:--------------|----------------:|
| source        |           21392 |
| flow_type     |           89965 |
| team_size     |           47372 |
| experience    |           44714 |
| usage_plan    |           44239 |
| frustration   |           44889 |
| first_feature |           21283 |
| role          |           66516 |

### Numerical & Categorical Summary
|        | user_id                                   | source    | flow_type   |   team_size | experience   | usage_plan   | frustration   | first_feature          | role    |
|:-------|:------------------------------------------|:----------|:------------|------------:|:-------------|:-------------|:--------------|:-----------------------|:--------|
| count  | 90004                                     | 68612     | 39          |       42632 | 45290        | 45765        | 45115         | 68721                  | 23488   |
| unique | 90000                                     | 533       | 3           |          13 | 4            | 7            | 12            | 21                     | 373     |
| top    | user_d95d86da-b4ef-4738-8809-b185fb7b9080 | instagram | invited     |           1 | beginner     | personal     | inconsistent  | Commercial & Ad Videos | creator |
| freq   | 2                                         | 18767     | 20          |       16538 | 24113        | 15387        | 7086          | 14656                  | 6948    |

### Top Value Distributions (Categorical)
**source**
| source       |   Count |
|:-------------|--------:|
| nan          |   21392 |
| instagram    |   18767 |
| youtube      |   11377 |
| friends      |    9802 |
| other        |    5663 |
| tiktok       |    5191 |
| google       |    4672 |
| ai-community |    3691 |
| chatgpt      |    2812 |
| twitter      |    2549 |

**flow_type**
| flow_type   |   Count |
|:------------|--------:|
| nan         |   89965 |
| invited     |      20 |
| personal    |      16 |
| team        |       3 |

**team_size**
| team_size   |   Count |
|:------------|--------:|
| nan         |   47372 |
| 1           |   16538 |
| solo        |   14902 |
| small       |    3728 |
| 2-10        |    3651 |
| 11-50       |     996 |
| growing     |     710 |
| 2001-5000   |     510 |
| 501-2000    |     394 |
| enterprise  |     339 |

**experience**
| experience   |   Count |
|:-------------|--------:|
| nan          |   44714 |
| beginner     |   24113 |
| intermediate |   13171 |
| expert       |    4176 |
| advanced     |    3830 |

**usage_plan**
| usage_plan   |   Count |
|:-------------|--------:|
| nan          |   44239 |
| personal     |   15387 |
| social       |   10069 |
| filmmaking   |    6820 |
| marketing    |    6611 |
| freelance    |    4915 |
| education    |    1960 |
| team         |       3 |

**frustration**
| frustration             |   Count |
|:------------------------|--------:|
| nan                     |   44889 |
| inconsistent            |    7086 |
| other                   |    5435 |
| High cost of top models |    4928 |
| high-cost               |    4430 |
| limited                 |    4299 |
| Other                   |    4264 |
| Inconsistent results    |    3777 |
| hard-prompt             |    3522 |
| Limited generations     |    2907 |

**first_feature**
| first_feature              |   Count |
|:---------------------------|--------:|
| nan                        |   21283 |
| Commercial & Ad Videos     |   14656 |
| Video Generations          |   11275 |
| video-creation             |    9529 |
| Cinematic Visuals          |    5811 |
| image-creation             |    4116 |
| Viral Social Media Content |    3107 |
| Realistic AI Avatars       |    3023 |
| Image Editing & Inpaint    |    2795 |
| consistent-character       |    2018 |

**role**
| role            |   Count |
|:----------------|--------:|
| nan             |   66516 |
| creator         |    6948 |
| filmmaker       |    3079 |
| designer        |    2893 |
| just-for-fun    |    2652 |
| brand-owner     |    1847 |
| marketer        |    1796 |
| founder         |    1283 |
| educator        |     872 |
| prompt-engineer |     867 |

## Table: Transactions

**Shape:** 178098 rows, 19 columns

### Missing Values
|                         |   Missing Count |
|:------------------------|----------------:|
| billing_address_country |            1115 |
| card_country            |               2 |
| bank_name               |           25580 |
| bank_country            |            2516 |
| is_prepaid              |             504 |
| is_virtual              |             504 |
| is_business             |             504 |
| failure_code            |           96424 |

### Numerical & Categorical Summary
|        | transaction_id                          | transaction_time                 |   amount_in_usd | billing_address_country   | card_3d_secure_support   | card_brand   | card_country   | card_funding   | cvc_check   | digital_wallet   |   is_3d_secure |   is_3d_secure_authenticated | payment_method_type   | bank_name                               | bank_country   |   is_prepaid |   is_virtual |   is_business | failure_code   |
|:-------|:----------------------------------------|:---------------------------------|----------------:|:--------------------------|:-------------------------|:-------------|:---------------|:---------------|:------------|:-----------------|---------------:|-----------------------------:|:----------------------|:----------------------------------------|:---------------|-------------:|-------------:|--------------:|:---------------|
| count  | 178098                                  | 178098                           |     178098      | 176983                    | 178098                   | 178098       | 178096         | 178098         | 178098      | 178098           |         178098 |                       178098 | 178098                | 152518                                  | 175582         |       177594 |       177594 |        177594 | 81674          |
| unique | 178098                                  |                                  |                 | 217                       | 4                        | 8            | 181            | 4              | 5           | 3                |              2 |                            2 | 1                     | 2683                                    | 174            |            2 |            2 |             2 | 7              |
| top    | ch_62af66bd-a44b-4ec7-8921-5ddf94f888e0 |                                  |                 | us                        | optional                 | visa         | us             | debit          | pass        | none             |              0 |                            0 | card                  | JCB INTERNATIONAL CREDIT CARD CO., LTD. | United States  |        False |        False |         False | card_declined  |
| freq   | 1                                       |                                  |                 | 31330                     | 113662                   | 89354        | 25223          | 89286          | 87542       | 169462           |         165661 |                       166145 | 178098                | 19148                                   | 84775          |       166186 |       166611 |        159098 | 78892          |
| mean   |                                         | 2023-10-06 12:24:22.317746+00:00 |         39.8672 |                           |                          |              |                |                |             |                  |                |                              |                       |                                         |                |              |              |               |                |
| min    |                                         | 2023-08-26 00:00:07+00:00        |          0.56   |                           |                          |              |                |                |             |                  |                |                              |                       |                                         |                |              |              |               |                |
| 25%    |                                         | 2023-09-11 18:12:27+00:00        |          9      |                           |                          |              |                |                |             |                  |                |                              |                       |                                         |                |              |              |               |                |
| 50%    |                                         | 2023-10-11 06:56:11.500000+00:00 |         35      |                           |                          |              |                |                |             |                  |                |                              |                       |                                         |                |              |              |               |                |
| 75%    |                                         | 2023-10-29 20:52:55.500000+00:00 |         35      |                           |                          |              |                |                |             |                  |                |                              |                       |                                         |                |              |              |               |                |
| max    |                                         | 2023-11-24 17:51:37+00:00        |       6239.17   |                           |                          |              |                |                |             |                  |                |                              |                       |                                         |                |              |              |               |                |
| std    |                                         |                                  |         73.7539 |                           |                          |              |                |                |             |                  |                |                              |                       |                                         |                |              |              |               |                |

### Top Value Distributions (Categorical)
**transaction_time**
| transaction_time          |   Count |
|:--------------------------|--------:|
| 2023-08-30 10:05:48+00:00 |       3 |
| 2023-10-14 16:59:33+00:00 |       3 |
| 2023-11-07 16:29:18+00:00 |       3 |
| 2023-09-10 11:05:41+00:00 |       3 |
| 2023-09-11 10:48:45+00:00 |       3 |
| 2023-11-05 10:36:55+00:00 |       3 |
| 2023-11-06 14:13:30+00:00 |       3 |
| 2023-10-21 14:16:41+00:00 |       3 |
| 2023-10-15 13:44:26+00:00 |       3 |
| 2023-11-07 16:15:57+00:00 |       3 |

**amount_in_usd**
|   amount_in_usd |   Count |
|----------------:|--------:|
|              35 |   60004 |
|               9 |   41537 |
|              89 |   18480 |
|              39 |   12170 |
|              29 |   11293 |
|              20 |    5600 |
|              26 |    4128 |
|              80 |    3569 |
|               5 |    2459 |
|              10 |    2332 |

**billing_address_country**
| billing_address_country   |   Count |
|:--------------------------|--------:|
| us                        |   31330 |
| jp                        |   21192 |
| in                        |    6593 |
| fr                        |    6556 |
| de                        |    5854 |
| hk                        |    5650 |
| kh                        |    5508 |
| vn                        |    5049 |
| kr                        |    4752 |
| gb                        |    4697 |

**card_3d_secure_support**
| card_3d_secure_support   |   Count |
|:-------------------------|--------:|
| optional                 |  113662 |
| recommended              |   41864 |
| not_supported            |   20314 |
| required                 |    2258 |

**card_brand**
| card_brand   |   Count |
|:-------------|--------:|
| visa         |   89354 |
| mc           |   63139 |
| jcb          |   19756 |
| amex         |    3889 |
| dscvr        |    1174 |
| link         |     499 |
| cup          |     228 |
| diners       |      59 |

**card_country**
| card_country   |   Count |
|:---------------|--------:|
| us             |   25223 |
| jp             |   20450 |
| fr             |   11425 |
| es             |    7512 |
| de             |    6493 |
| gb             |    6234 |
| in             |    6139 |
| tr             |    5892 |
| it             |    5024 |
| kr             |    4563 |

**card_funding**
| card_funding   |   Count |
|:---------------|--------:|
| debit          |   89286 |
| credit         |   77653 |
| prepaid        |   10657 |
| unknown        |     502 |

**cvc_check**
| cvc_check    |   Count |
|:-------------|--------:|
| pass         |   87542 |
| unavailable  |   50649 |
| not_provided |   36524 |
| fail         |    2485 |
| unchecked    |     898 |

**digital_wallet**
| digital_wallet   |   Count |
|:-----------------|--------:|
| none             |  169462 |
| apple_pay        |    8453 |
| android_pay      |     183 |

**is_3d_secure**
| is_3d_secure   |   Count |
|:---------------|--------:|
| False          |  165661 |
| True           |   12437 |

**is_3d_secure_authenticated**
| is_3d_secure_authenticated   |   Count |
|:-----------------------------|--------:|
| False                        |  166145 |
| True                         |   11953 |

**payment_method_type**
| payment_method_type   |   Count |
|:----------------------|--------:|
| card                  |  178098 |

**bank_name**
| bank_name                                          |   Count |
|:---------------------------------------------------|--------:|
| nan                                                |   25580 |
| JCB INTERNATIONAL CREDIT CARD CO., LTD.            |   19148 |
| VTB BANK OJSC                                      |    3574 |
| REVOLUT BANK UAB                                   |    3469 |
| CAJA DE AHORROS Y PENSIONES DE BARCElanA(LA CAIXA) |    2959 |
| WESTPAC BANKING CORPORATION                        |    2305 |
| JPMORGAN CHASE BANK, N.A.                          |    2204 |
| LA BANQUE POSTALE                                  |    2162 |
| CHASE BANK USA, N.A.                               |    1872 |
| KASPI BANK JSC                                     |    1463 |

**bank_country**
| bank_country   |   Count |
|:---------------|--------:|
| United States  |   84775 |
| France         |    8300 |
| Spain          |    6708 |
| Canada         |    5084 |
| Turkey         |    4334 |
| United Kingdom |    4226 |
| Italy          |    3150 |
| India          |    2834 |
| Brazil         |    2782 |
| Australia      |    2772 |

**is_prepaid**
|   is_prepaid |   Count |
|-------------:|--------:|
|            0 |  166186 |
|            1 |   11408 |
|          nan |     504 |

**is_virtual**
|   is_virtual |   Count |
|-------------:|--------:|
|            0 |  166611 |
|            1 |   10983 |
|          nan |     504 |

**is_business**
|   is_business |   Count |
|--------------:|--------:|
|             0 |  159098 |
|             1 |   18496 |
|           nan |     504 |

**failure_code**
| failure_code            |   Count |
|:------------------------|--------:|
| nan                     |   96424 |
| card_declined           |   78892 |
| incorrect_cvc           |     972 |
| incorrect_number        |     765 |
| expired_card            |     519 |
| processing_error        |     378 |
| invalid_cvc             |     127 |
| authentication_required |      21 |

---

# EDA Summary for 'TEST' Split

## Table: Users

**Shape:** 7000 rows, 1 columns

*No missing values.*

### Numerical & Categorical Summary
|        | user_id                                   |
|:-------|:------------------------------------------|
| count  | 7000                                      |
| unique | 7000                                      |
| top    | user_d318c0bc-2d30-4fdd-a984-92565fe7b3ba |
| freq   | 1                                         |

### Top Value Distributions (Categorical)
## Table: Generations

**Shape:** 1078213 rows, 12 columns

### Missing Values
|                     |   Missing Count |
|:--------------------|----------------:|
| created_at          |               1 |
| completed_at        |          115950 |
| failed_at           |          968262 |
| credit_cost         |          910908 |
| resolution          |          570872 |
| aspect_ration       |           66633 |
| duration            |          910008 |
| processing_time_sec |          115951 |

### Numerical & Categorical Summary
|        | user_id                                   | generation_id                            | created_at                       | completed_at                     | failed_at                        | status    |   credit_cost | generation_type   | resolution   | aspect_ration   |     duration |   processing_time_sec |
|:-------|:------------------------------------------|:-----------------------------------------|:---------------------------------|:---------------------------------|:---------------------------------|:----------|--------------:|:------------------|:-------------|:----------------|-------------:|----------------------:|
| count  | 1078213                                   | 1078213                                  | 1078212                          | 962263                           | 109951                           | 1078213   |     167305    | 1078213           | 507341       | 1011580         | 168205       |      962262           |
| unique | 6687                                      | 1078213                                  |                                  |                                  |                                  | 7         |               | 21                | 11           | 12              |              |                       |
| top    | user_598247fb-79f7-45a0-b3c6-812fdb1dcb68 | gen_a7aaa45b-dc9c-4d51-ba6d-7ddf059c09c4 |                                  |                                  |                                  | completed |               | image_model_1     | 2k           | 9:16            |              |                       |
| freq   | 9075                                      | 1                                        |                                  |                                  |                                  | 962424    |               | 423545            | 254240       | 314781          |              |                       |
| mean   |                                           |                                          | 2023-11-21 14:14:48.901024+00:00 | 2023-11-21 16:38:52.063638+00:00 | 2023-11-20 21:14:01.433249+00:00 |           |       1335.79 |                   |              |                 |      8.08521 |         136.39        |
| min    |                                           |                                          | 2023-11-09 03:55:46.682622+00:00 | 2023-11-09 03:56:32.350825+00:00 | 2023-11-09 05:29:12.965074+00:00 |           |          0    |                   |              |                 |      1       |           6.53951     |
| 25%    |                                           |                                          | 2023-11-16 21:02:52.567893+00:00 | 2023-11-16 22:33:40.563603+00:00 | 2023-11-16 08:05:56.937874+00:00 |           |        625    |                   |              |                 |      5       |          42.1408      |
| 50%    |                                           |                                          | 2023-11-21 01:54:49.178576+00:00 | 2023-11-21 05:58:54.475805+00:00 | 2023-11-20 11:11:05.068690+00:00 |           |       1000    |                   |              |                 |      6.284   |          69.4237      |
| 75%    |                                           |                                          | 2023-11-25 16:35:59.220915+00:00 | 2023-11-25 19:25:12.660557+00:00 | 2023-11-24 10:39:12.786880+00:00 |           |       1750    |                   |              |                 |     10       |         124.832       |
| max    |                                           |                                          | 2023-12-08 18:26:43.266104+00:00 | 2023-12-08 18:27:25.040938+00:00 | 2023-12-08 17:46:10.257957+00:00 |           |      17500    |                   |              |                 |     40.7277  |           1.26575e+06 |
| std    |                                           |                                          |                                  |                                  |                                  |           |       1145.95 |                   |              |                 |      4.18951 |        1397.58        |

### Top Value Distributions (Categorical)
**created_at**
| created_at                       |   Count |
|:---------------------------------|--------:|
| 2023-11-21 17:21:22.892699+00:00 |      13 |
| 2023-11-21 12:56:57.211313+00:00 |      13 |
| 2023-11-21 13:37:54.521973+00:00 |      13 |
| 2023-11-21 18:02:27.258869+00:00 |      13 |
| 2023-11-21 16:52:23.289573+00:00 |      13 |
| 2023-11-21 17:42:12.621984+00:00 |      13 |
| 2023-11-20 02:33:03.338965+00:00 |      13 |
| 2023-11-20 02:33:03.339008+00:00 |      13 |
| 2023-11-20 02:33:03.339034+00:00 |      13 |
| 2023-11-20 04:00:56.393905+00:00 |      13 |

**completed_at**
| completed_at                     |   Count |
|:---------------------------------|--------:|
| NaT                              |  115950 |
| 2023-11-21 17:22:39.162672+00:00 |      13 |
| 2023-11-21 12:57:39.080687+00:00 |      13 |
| 2023-11-21 13:38:36.987308+00:00 |      13 |
| 2023-11-21 18:03:10.139478+00:00 |      13 |
| 2023-11-21 16:53:07.116932+00:00 |      13 |
| 2023-11-21 17:42:52.095246+00:00 |      13 |
| 2023-11-20 02:34:08.682111+00:00 |      13 |
| 2023-11-20 02:34:10.933260+00:00 |      13 |
| 2023-11-20 02:33:45.228457+00:00 |      13 |

**failed_at**
| failed_at                        |   Count |
|:---------------------------------|--------:|
| NaT                              |  968262 |
| 2023-11-25 15:48:32.431665+00:00 |      11 |
| 2023-11-19 14:23:50.659035+00:00 |       7 |
| 2023-11-19 14:13:51.809920+00:00 |       7 |
| 2023-11-17 14:22:29.571245+00:00 |       2 |
| 2023-11-19 14:54:13.260401+00:00 |       2 |
| 2023-11-15 19:20:02.230128+00:00 |       2 |
| 2023-11-15 17:54:35.994706+00:00 |       2 |
| 2023-11-17 15:30:31.373942+00:00 |       2 |
| 2023-11-17 18:37:15.880075+00:00 |       2 |

**status**
| status      |   Count |
|:------------|--------:|
| completed   |  962424 |
| nsfw        |   63946 |
| failed      |   45758 |
| canceled    |    5898 |
| queued      |     168 |
| in_progress |      16 |
| waiting     |       3 |

**credit_cost**
|   credit_cost |   Count |
|--------------:|--------:|
|           nan |  910908 |
|          1000 |   14696 |
|           600 |   12051 |
|           500 |   10951 |
|           875 |    8751 |
|          2250 |    7211 |
|             0 |    7182 |
|           900 |    7171 |
|          2000 |    6165 |
|          1200 |    5329 |

**generation_type**
| generation_type   |   Count |
|:------------------|--------:|
| image_model_1     |  423545 |
| image_model_2     |  203169 |
| image_model_9     |  142368 |
| video_model_8     |   67801 |
| image_model_3     |   55856 |
| image_model_6     |   31855 |
| image_model_4     |   25725 |
| video_model_3     |   21535 |
| video_model_6     |   20327 |
| video_model_1     |   17177 |

**resolution**
| resolution   |   Count |
|:-------------|--------:|
| nan          |  570872 |
| 2k           |  254240 |
| 1k           |  132020 |
| 4k           |   81493 |
| 720p         |   28933 |
| 1080p        |    3566 |
| 480p         |    1921 |
| 1080         |    1664 |
| 768          |    1544 |
| 720          |    1391 |

**aspect_ration**
| aspect_ration   |   Count |
|:----------------|--------:|
| 9:16            |  314781 |
| 3:4             |  258095 |
| 16:9            |  209119 |
| 1:1             |   84244 |
| nan             |   66633 |
| auto            |   32867 |
| 4:5             |   29964 |
| 4:3             |   25576 |
| 2:3             |   21854 |
| 21:9            |   21455 |

**duration**
|   duration |   Count |
|-----------:|--------:|
|        nan |  910008 |
|          5 |   49472 |
|         10 |   20026 |
|          6 |   14848 |
|         15 |   13072 |
|          8 |   12533 |
|          4 |    9311 |
|         12 |    8303 |
|          3 |    7580 |
|          9 |    6524 |

**processing_time_sec**
|   processing_time_sec |   Count |
|----------------------:|--------:|
|              nan      |  115951 |
|               76.27   |      13 |
|               41.8694 |      13 |
|               42.4653 |      13 |
|               42.8806 |      13 |
|               43.8274 |      13 |
|               39.4733 |      13 |
|               65.3431 |      13 |
|               67.5943 |      13 |
|               41.8894 |      13 |

## Table: Properties

**Shape:** 7000 rows, 4 columns

### Missing Values
|              |   Missing Count |
|:-------------|----------------:|
| country_code |               2 |

### Numerical & Categorical Summary
|        | user_id                                   | subscription_start_date          | subscription_plan   | country_code   |
|:-------|:------------------------------------------|:---------------------------------|:--------------------|:---------------|
| count  | 7000                                      | 7000                             | 7000                | 6998           |
| unique | 7000                                      |                                  | 4                   | 153            |
| top    | user_3bb244bf-2efa-4cee-b5a7-a7b395e6e687 |                                  | Higgsfield Ultimate | US             |
| freq   | 1                                         |                                  | 3200                | 1299           |
| mean   |                                           | 2023-11-16 15:14:05.424571+00:00 |                     |                |
| min    |                                           | 2023-11-10 00:09:15+00:00        |                     |                |
| 25%    |                                           | 2023-11-13 07:18:11.250000+00:00 |                     |                |
| 50%    |                                           | 2023-11-16 09:39:05+00:00        |                     |                |
| 75%    |                                           | 2023-11-19 23:01:37+00:00        |                     |                |
| max    |                                           | 2023-11-23 23:56:03+00:00        |                     |                |

### Top Value Distributions (Categorical)
**subscription_start_date**
| subscription_start_date   |   Count |
|:--------------------------|--------:|
| 2023-11-13 22:11:52+00:00 |       2 |
| 2023-11-16 10:50:25+00:00 |       2 |
| 2023-11-11 04:38:19+00:00 |       2 |
| 2023-11-14 14:26:29+00:00 |       2 |
| 2023-11-15 01:09:38+00:00 |       2 |
| 2023-11-16 14:46:57+00:00 |       2 |
| 2023-11-11 13:36:26+00:00 |       2 |
| 2023-11-11 17:44:03+00:00 |       2 |
| 2023-11-22 15:38:21+00:00 |       2 |
| 2023-11-14 11:51:29+00:00 |       2 |

**subscription_plan**
| subscription_plan   |   Count |
|:--------------------|--------:|
| Higgsfield Ultimate |    3200 |
| Higgsfield Basic    |    2000 |
| Higgsfield Pro      |    1600 |
| Higgsfield Creator  |     200 |

**country_code**
| country_code   |   Count |
|:---------------|--------:|
| US             |    1299 |
| IN             |     630 |
| DE             |     365 |
| GB             |     330 |
| KR             |     268 |
| FR             |     267 |
| BR             |     189 |
| CA             |     189 |
| UA             |     182 |
| TR             |     175 |

## Table: Purchases

**Shape:** 8825 rows, 5 columns

*No missing values.*

### Numerical & Categorical Summary
|        | user_id                                   | transaction_id                          | purchase_time                    | purchase_type       |   purchase_amount_dollars |
|:-------|:------------------------------------------|:----------------------------------------|:---------------------------------|:--------------------|--------------------------:|
| count  | 8825                                      | 8825                                    | 8825                             | 8825                |                 8825      |
| unique | 6999                                      | 8825                                    |                                  | 5                   |                           |
| top    | user_6c26ee22-c16a-4f86-8a80-050b4766adaf | ch_7d97fc3f-56ae-40bc-92cc-4a97a5f53bdd |                                  | Subscription Create |                           |
| freq   | 23                                        | 1                                       |                                  | 6999                |                           |
| mean   |                                           |                                         | 2023-11-17 16:20:42.149121+00:00 |                     |                   34.4797 |
| min    |                                           |                                         | 2023-11-10 00:09:15+00:00        |                     |                    0.84   |
| 25%    |                                           |                                         | 2023-11-13 14:59:42+00:00        |                     |                    9      |
| 50%    |                                           |                                         | 2023-11-17 12:47:33+00:00        |                     |                   29      |
| 75%    |                                           |                                         | 2023-11-20 22:01:59+00:00        |                     |                   49      |
| max    |                                           |                                         | 2023-12-07 20:03:42+00:00        |                     |                  286      |
| std    |                                           |                                         |                                  |                     |                   33.0068 |

### Top Value Distributions (Categorical)
**purchase_time**
| purchase_time             |   Count |
|:--------------------------|--------:|
| 2023-11-21 23:04:33+00:00 |       2 |
| 2023-11-23 19:50:29+00:00 |       2 |
| 2023-11-16 14:22:11+00:00 |       2 |
| 2023-11-15 20:16:55+00:00 |       2 |
| 2023-11-13 14:23:16+00:00 |       2 |
| 2023-11-18 15:35:26+00:00 |       2 |
| 2023-11-21 15:56:42+00:00 |       2 |
| 2023-11-15 18:21:17+00:00 |       2 |
| 2023-11-22 15:38:21+00:00 |       2 |
| 2023-11-17 20:38:36+00:00 |       2 |

**purchase_type**
| purchase_type       |   Count |
|:--------------------|--------:|
| Subscription Create |    6999 |
| Credits package     |    1171 |
| Subscription Update |     383 |
| Upsell              |     270 |
| Gift                |       2 |

**purchase_amount_dollars**
|   purchase_amount_dollars |   Count |
|--------------------------:|--------:|
|                        49 |    2447 |
|                         9 |    1993 |
|                        29 |    1454 |
|                        39 |     436 |
|                        35 |     373 |
|                        10 |     329 |
|                        20 |     297 |
|                         5 |     253 |
|                        40 |     203 |
|                        19 |     115 |

## Table: Quizzes

**Shape:** 7000 rows, 9 columns

### Missing Values
|               |   Missing Count |
|:--------------|----------------:|
| source        |            1090 |
| flow_type     |            7000 |
| team_size     |            3971 |
| experience    |            1339 |
| usage_plan    |            1281 |
| frustration   |            1430 |
| first_feature |            1088 |
| role          |            6685 |

### Numerical & Categorical Summary
|        | user_id                                   | source    |   flow_type | team_size   | experience   | usage_plan   | frustration   | first_feature     | role    |
|:-------|:------------------------------------------|:----------|------------:|:------------|:-------------|:-------------|:--------------|:------------------|:--------|
| count  | 7000                                      | 5910      |           0 | 3029        | 5661         | 5719         | 5570          | 5912              | 315     |
| unique | 7000                                      | 23        |             | 13          | 4            | 6            | 12            | 19                | 15      |
| top    | user_a80ae0fb-d842-4d71-a60a-7043d9021991 | instagram |             | solo        | beginner     | personal     | inconsistent  | Video Generations | creator |
| freq   | 1                                         | 1396      |             | 2096        | 3062         | 1711         | 1374          | 1888              | 93      |
| mean   |                                           |           |             |             |              |              |               |                   |         |
| std    |                                           |           |             |             |              |              |               |                   |         |
| min    |                                           |           |             |             |              |              |               |                   |         |
| 25%    |                                           |           |             |             |              |              |               |                   |         |
| 50%    |                                           |           |             |             |              |              |               |                   |         |
| 75%    |                                           |           |             |             |              |              |               |                   |         |
| max    |                                           |           |             |             |              |              |               |                   |         |

### Top Value Distributions (Categorical)
**source**
| source       |   Count |
|:-------------|--------:|
| instagram    |    1396 |
| youtube      |    1098 |
| nan          |    1090 |
| other        |     746 |
| friends      |     708 |
| tiktok       |     499 |
| ai-community |     470 |
| google       |     412 |
| chatgpt      |     228 |
| twitter      |     144 |

**flow_type**
|   flow_type |   Count |
|------------:|--------:|
|         nan |    7000 |

**team_size**
| team_size   |   Count |
|:------------|--------:|
| nan         |    3971 |
| solo        |    2096 |
| small       |     486 |
| 1           |     244 |
| growing     |      82 |
| 2-10        |      48 |
| midsize     |      28 |
| enterprise  |      21 |
| 11-50       |      11 |
| 2001-5000   |       4 |

**experience**
| experience   |   Count |
|:-------------|--------:|
| beginner     |    3062 |
| intermediate |    1685 |
| nan          |    1339 |
| advanced     |     466 |
| expert       |     448 |

**usage_plan**
| usage_plan   |   Count |
|:-------------|--------:|
| personal     |    1711 |
| nan          |    1281 |
| social       |    1217 |
| filmmaking   |     975 |
| marketing    |     902 |
| freelance    |     632 |
| education    |     282 |

**frustration**
| frustration             |   Count |
|:------------------------|--------:|
| nan                     |    1430 |
| inconsistent            |    1374 |
| high-cost               |    1083 |
| other                   |    1015 |
| limited                 |     817 |
| hard-prompt             |     675 |
| confusing               |     322 |
| High cost of top models |      85 |
| Other                   |      66 |
| Inconsistent results    |      48 |

**first_feature**
| first_feature              |   Count |
|:---------------------------|--------:|
| Video Generations          |    1888 |
| Commercial & Ad Videos     |    1665 |
| nan                        |    1088 |
| Realistic AI Avatars       |     666 |
| Cinematic Visuals          |     468 |
| Viral Social Media Content |     319 |
| Image Editing & Inpaint    |     315 |
| video-creation             |     130 |
| Storyboarding              |     111 |
| Lipsync & Talking Avatars  |      77 |

**role**
| role            |   Count |
|:----------------|--------:|
| nan             |    6685 |
| creator         |      93 |
| filmmaker       |      50 |
| designer        |      46 |
| just-for-fun    |      45 |
| marketer        |      19 |
| brand-owner     |      19 |
| prompt-engineer |      17 |
| founder         |      10 |
| developer       |       7 |

## Table: Transactions

**Shape:** 11822 rows, 19 columns

### Missing Values
|                         |   Missing Count |
|:------------------------|----------------:|
| billing_address_country |              68 |
| bank_name               |            2145 |
| bank_country            |             194 |
| is_prepaid              |              62 |
| is_virtual              |              62 |
| is_business             |              62 |
| failure_code            |            8825 |

### Numerical & Categorical Summary
|        | transaction_id                          | transaction_time                 |   amount_in_usd | billing_address_country   | card_3d_secure_support   | card_brand   | card_country   | card_funding   | cvc_check   | digital_wallet   |   is_3d_secure |   is_3d_secure_authenticated | payment_method_type   | bank_name     | bank_country   |   is_prepaid |   is_virtual |   is_business | failure_code   |
|:-------|:----------------------------------------|:---------------------------------|----------------:|:--------------------------|:-------------------------|:-------------|:---------------|:---------------|:------------|:-----------------|---------------:|-----------------------------:|:----------------------|:--------------|:---------------|-------------:|-------------:|--------------:|:---------------|
| count  | 11822                                   | 11822                            |      11822      | 11754                     | 11822                    | 11822        | 11822          | 11822          | 11822       | 11822            |          11822 |                        11822 | 11822                 | 9677          | 11628          |        11760 |        11760 |         11760 | 2997           |
| unique | 11822                                   |                                  |                 | 154                       | 4                        | 8            | 134            | 4              | 5           | 3                |              2 |                            2 | 1                     | 1084          | 126            |            2 |            2 |             2 | 7              |
| top    | ch_2171f704-7f24-4003-bcb1-5ee3c87c8469 |                                  |                 | us                        | optional                 | visa         | us             | debit          | pass        | none             |              0 |                            0 | card                  | VTB BANK OJSC | United States  |        False |        False |         False | card_declined  |
| freq   | 1                                       |                                  |                 | 2089                      | 11133                    | 6248         | 1853           | 6335           | 6500        | 10950            |          10363 |                        10399 | 11822                 | 395           | 5390           |        11012 |        10690 |         10274 | 2819           |
| mean   |                                         | 2023-11-17 18:27:00.129927+00:00 |         41.4474 |                           |                          |              |                |                |             |                  |                |                              |                       |               |                |              |              |               |                |
| min    |                                         | 2023-11-10 00:09:15+00:00        |          0.91   |                           |                          |              |                |                |             |                  |                |                              |                       |               |                |              |              |               |                |
| 25%    |                                         | 2023-11-13 16:19:34.500000+00:00 |         10.89   |                           |                          |              |                |                |             |                  |                |                              |                       |               |                |              |              |               |                |
| 50%    |                                         | 2023-11-17 14:32:04.500000+00:00 |         31.9    |                           |                          |              |                |                |             |                  |                |                              |                       |               |                |              |              |               |                |
| 75%    |                                         | 2023-11-21 00:57:54+00:00        |         49      |                           |                          |              |                |                |             |                  |                |                              |                       |               |                |              |              |               |                |
| max    |                                         | 2023-12-07 20:03:42+00:00        |       2375.7    |                           |                          |              |                |                |             |                  |                |                              |                       |               |                |              |              |               |                |
| std    |                                         |                                  |         73.107  |                           |                          |              |                |                |             |                  |                |                              |                       |               |                |              |              |               |                |

### Top Value Distributions (Categorical)
**transaction_time**
| transaction_time          |   Count |
|:--------------------------|--------:|
| 2023-11-23 19:50:29+00:00 |       2 |
| 2023-11-15 01:09:38+00:00 |       2 |
| 2023-11-15 18:21:17+00:00 |       2 |
| 2023-11-15 10:29:28+00:00 |       2 |
| 2023-11-15 10:17:46+00:00 |       2 |
| 2023-11-15 20:16:55+00:00 |       2 |
| 2023-11-15 14:34:02+00:00 |       2 |
| 2023-11-15 17:40:57+00:00 |       2 |
| 2023-11-20 16:09:24+00:00 |       2 |
| 2023-11-20 16:17:41+00:00 |       2 |

**amount_in_usd**
|   amount_in_usd |   Count |
|----------------:|--------:|
|           49    |    2507 |
|            9    |    1924 |
|           29    |    1590 |
|           39    |     560 |
|           35    |     470 |
|           20    |     307 |
|           10    |     289 |
|            5    |     262 |
|           40    |     201 |
|           59.29 |     156 |

**billing_address_country**
| billing_address_country   |   Count |
|:--------------------------|--------:|
| us                        |    2089 |
| in                        |    1010 |
| de                        |     547 |
| kr                        |     515 |
| gb                        |     508 |
| fr                        |     408 |
| ca                        |     376 |
| it                        |     320 |
| ua                        |     309 |
| es                        |     303 |

**card_3d_secure_support**
| card_3d_secure_support   |   Count |
|:-------------------------|--------:|
| optional                 |   11133 |
| recommended              |     300 |
| required                 |     275 |
| not_supported            |     114 |

**card_brand**
| card_brand   |   Count |
|:-------------|--------:|
| visa         |    6248 |
| mc           |    4902 |
| amex         |     430 |
| dscvr        |     116 |
| link         |      62 |
| jcb          |      41 |
| cup          |      12 |
| diners       |      11 |

**card_country**
| card_country   |   Count |
|:---------------|--------:|
| us             |    1853 |
| in             |    1012 |
| gb             |     630 |
| de             |     541 |
| kr             |     517 |
| fr             |     439 |
| ca             |     377 |
| it             |     360 |
| ua             |     336 |
| es             |     302 |

**card_funding**
| card_funding   |   Count |
|:---------------|--------:|
| debit          |    6335 |
| credit         |    4798 |
| prepaid        |     627 |
| unknown        |      62 |

**cvc_check**
| cvc_check    |   Count |
|:-------------|--------:|
| pass         |    6500 |
| not_provided |    4571 |
| unavailable  |     465 |
| fail         |     224 |
| unchecked    |      62 |

**digital_wallet**
| digital_wallet   |   Count |
|:-----------------|--------:|
| none             |   10950 |
| apple_pay        |     762 |
| android_pay      |     110 |

**is_3d_secure**
| is_3d_secure   |   Count |
|:---------------|--------:|
| False          |   10363 |
| True           |    1459 |

**is_3d_secure_authenticated**
| is_3d_secure_authenticated   |   Count |
|:-----------------------------|--------:|
| False                        |   10399 |
| True                         |    1423 |

**payment_method_type**
| payment_method_type   |   Count |
|:----------------------|--------:|
| card                  |   11822 |

**bank_name**
| bank_name                          |   Count |
|:-----------------------------------|--------:|
| nan                                |    2145 |
| VTB BANK OJSC                      |     395 |
| REVOLUT BANK UAB                   |     305 |
| YES BANK, LTD.                     |     229 |
| JPMORGAN CHASE BANK, N.A.          |     224 |
| WESTPAC BANKING CORPORATION        |     175 |
| JSC UNIVERSAL BANK                 |     164 |
| CHASE BANK USA, N.A.               |     150 |
| SUMITOMO MITSUI CARD COMPANY, LTD. |     145 |
| BANK OF AMERICA                    |     120 |

**bank_country**
| bank_country   |   Count |
|:---------------|--------:|
| United States  |    5390 |
| India          |     421 |
| Canada         |     398 |
| United Kingdom |     397 |
| Ukraine        |     284 |
| Australia      |     263 |
| Spain          |     250 |
| Germany        |     215 |
| Turkey         |     206 |
| Japan          |     200 |

**is_prepaid**
|   is_prepaid |   Count |
|-------------:|--------:|
|            0 |   11012 |
|            1 |     748 |
|          nan |      62 |

**is_virtual**
|   is_virtual |   Count |
|-------------:|--------:|
|            0 |   10690 |
|            1 |    1070 |
|          nan |      62 |

**is_business**
|   is_business |   Count |
|--------------:|--------:|
|             0 |   10274 |
|             1 |    1486 |
|           nan |      62 |

**failure_code**
| failure_code            |   Count |
|:------------------------|--------:|
| nan                     |    8825 |
| card_declined           |    2819 |
| incorrect_cvc           |      76 |
| incorrect_number        |      38 |
| expired_card            |      29 |
| processing_error        |      22 |
| invalid_cvc             |      10 |
| authentication_required |       3 |

---

