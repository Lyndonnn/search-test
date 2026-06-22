# Answer Equivalence Verifier v3 Summary

## Overall

| split | n | supported_answer_v2_positive_rate | supported_answer_hard_v3_positive_rate | supported_answer_soft_mean | semantic_equivalent_rate | answer_correct_score_mean | judge_uncertain_or_malformed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1832 | 0.0212882096069869 | 0.025655021834061136 | 0.0749831878744728 | 0.03275109170305677 | 0.0866007336762893 | 0.002183406113537118 |
| dev | 392 | 0.015306122448979591 | 0.01020408163265306 | 0.07280075425020119 | 0.015306122448979591 | 0.09340841361087086 | 0.007653061224489796 |
| test | 256 | 0.01171875 | 0.015625 | 0.058743505079616606 | 0.0390625 | 0.08950558688161572 | 0.00390625 |

## Breakdown By Answer Type

| split | answer_type | n | hard_rate | soft_mean | semantic_equivalent_rate | answer_correct_score_mean |
| --- | --- | --- | --- | --- | --- | --- |
| dev | address | 30 | 0.0 | 0.16334121540744392 | 0.0 | 0.27802534559116715 |
| dev | company_id | 75 | 0.013333333333333334 | 0.013333333333333334 | 0.013333333333333334 | 0.013333333333333334 |
| dev | count | 53 | 0.018867924528301886 | 0.018867924528301886 | 0.018867924528301886 | 0.018867924528301886 |
| dev | date | 8 | 0.0 | 0.0 | 0.0 | 0.0 |
| dev | email | 20 | 0.0 | 0.0 | 0.0 | 0.0 |
| dev | entity_name | 33 | 0.06060606060606061 | 0.3417362152290382 | 0.06060606060606061 | 0.39114622208585353 |
| dev | other | 15 | 0.0 | 0.41139393939393937 | 0.06666666666666667 | 0.5277272727272727 |
| dev | phone | 131 | 0.0 | 0.0 | 0.007633587786259542 | 0.007633587786259542 |
| dev | price | 12 | 0.0 | 0.0 | 0.0 | 0.0 |
| dev | title | 15 | 0.0 | 0.2792970006925466 | 0.0 | 0.29677355653227383 |
| test | address | 24 | 0.0 | 0.2594238243294162 | 0.041666666666666664 | 0.42508331596875976 |
| test | company_id | 23 | 0.08695652173913043 | 0.08695652173913043 | 0.08695652173913043 | 0.08695652173913043 |
| test | count | 32 | 0.0625 | 0.0625 | 0.0625 | 0.0625 |
| test | date | 12 | 0.0 | 0.0 | 0.0 | 0.0 |
| test | email | 12 | 0.0 | 0.0 | 0.0 | 0.0 |
| test | entity_name | 17 | 0.0 | 0.1829164299752535 | 0.23529411764705882 | 0.38045947158214016 |
| test | other | 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| test | phone | 118 | 0.0 | 0.0 | 0.0 | 0.0 |
| test | price | 11 | 0.0 | 0.0 | 0.0 | 0.0 |
| test | title | 6 | 0.0 | 0.28376436781609193 | 0.16666666666666666 | 0.37393660692450126 |
| train | address | 131 | 0.0 | 0.2344814947528816 | 0.05343511450381679 | 0.30152639117501306 |
| train | company_id | 225 | 0.017777777777777778 | 0.017777777777777778 | 0.017777777777777778 | 0.017777777777777778 |
| train | count | 220 | 0.1 | 0.1 | 0.1 | 0.1 |
| train | date | 56 | 0.05357142857142857 | 0.05357142857142857 | 0.05357142857142857 | 0.05357142857142857 |
| train | description | 4 | 0.0 | 0.05 | 0.0 | 0.27420634920634923 |
| train | email | 103 | 0.0 | 0.0 | 0.0 | 0.0 |
| train | entity_name | 118 | 0.025423728813559324 | 0.2422690805562084 | 0.059322033898305086 | 0.28780003677840676 |
| train | other | 95 | 0.07368421052631578 | 0.28582295527024315 | 0.08421052631578947 | 0.32028108565158225 |
| train | phone | 655 | 0.004580152671755725 | 0.004580152671755725 | 0.004580152671755725 | 0.004580152671755725 |
| train | price | 154 | 0.006493506493506494 | 0.006493506493506494 | 0.012987012987012988 | 0.012987012987012988 |
| train | title | 71 | 0.056338028169014086 | 0.249453410100014 | 0.056338028169014086 | 0.27702329545714915 |

## Potential False Positive Examples



## Potential False Negative Examples

| split | sample_id | answer_type | gold | pred | soft | failure |
| --- | --- | --- | --- | --- | --- | --- |
| train | pix2fact_6fdb083f22 | address | 3231 Canton Road, Tsim Sha Tsui, Hong Kong | 22 Canton Road, Tsim Sha Tsui, Kowloon, Hong Kong | 0.845 | judge_uncertain |
| train | pix2fact_d964cbd2fc | phone | 2025 | 2025年 | 0.000 | format_mismatch |
| train | pix2fact_071f656f40 | company_id | 10 pm | 10 PM every night | 0.000 | wrong_value |
| train | pix2fact_28edaa0409 | address | 10:00 | 10:00-17:30 | 0.900 | wrong_entity |
| train | pix2fact_28edaa0409 | company_id | 10:00 | Monday and Tuesday 10:00 - 17:00 Wednesday to Sunday 9:00 - 18:00 | 0.000 | wrong_value |
| train | pix2fact_681e777473 | address | 10:00 | 10：00-22：00 | 0.900 | wrong_entity |
| train | pix2fact_21ebcaa5ad | company_id | 601933.SH | 601933 | 0.000 | wrong_value |
| train | pix2fact_21ebcaa5ad | company_id | 601933.SH | 601933 | 0.000 | wrong_value |
| train | pix2fact_21ebcaa5ad | company_id | 601933.SH | 601933 | 0.000 | wrong_value |
| train | pix2fact_9f2e4a61c0 | price | 500 USD | $500 | 0.000 | wrong_value |
| train | pix2fact_9f2e4a61c0 | price | 500 USD | $500 USD | 0.000 | wrong_value |
| train | pix2fact_ac5a3ac4ba | company_id | 10AM | 10 am to 6 pm | 0.000 | wrong_value |
| dev | pix2fact_9434eeee4b | date | 10:00 | 10:00-20:00 | 0.000 | wrong_value |
| dev | pix2fact_daf350b1d5 | company_id | 10 am | 10 am to 11 pm | 0.000 | wrong_value |
| dev | pix2fact_d7063afcae | company_id | 12:00 PM | 12:00 PM (noon) | 0.000 | wrong_value |
| test | pix2fact_243d684e7d | count | 9 days | 29 days | 0.000 | wrong_value |

## Judge Uncertain Examples

| split | sample_id | answer_type | gold | pred | judge_score | reason |
| --- | --- | --- | --- | --- | --- | --- |
| train | pix2fact_ee0d7ec2ae | phone | 〒605-0073, 257-1 Gionmachi Kitagawa, Higashiyama-ku, Kyōto-shi, Kyōto-fu, Japan | 4-26-10 Oike, Kita-ku, Kyoto-shi, Kyoto, 600-8335, Japan | 0.000 | invalid_judge_json: { "semantic_equivalent": false, "evidence_supports_prediction": false, "missing_key_information": tr |
| train | pix2fact_6fdb083f22 | address | 3231 Canton Road, Tsim Sha Tsui, Hong Kong | 22 Canton Road, Tsim Sha Tsui, Kowloon, Hong Kong | 0.400 | 虽然两个地址都在尖沙咀的 Canton Road,但具体门牌号不同。黄金答案是3231号,而预测答案是22号。 |
| train | pix2fact_abd02d5a53 | title | Les missionnaires botaniques en Chine | The Future of China's Green Transition | 0.000 | invalid_judge_json: {      "semantic_equivalent": false,    "evidence_supports_prediction": true,    "missing_key_inform |
| train | pix2fact_54b7438347 | entity_name | Pineapple Filled Butter Cookies | Pineapple Traditional Butter Cookies | 0.500 | While both answers mention 'Pineapple' and 'Butter Cookies', the gold answer specifies 'FILLED' which is not included in |
| dev | pix2fact_bc331ccd8f | entity_name | Domori | Cadbury's Dairy Milk (Dairy Milk) | 0.000 | invalid_judge_json: { "semantic_equivalent": false, "evidence_supports_prediction": false, "missing_key_information": fa |
| dev | pix2fact_926bc1ea82 | company_id | ThinkPad T14 Gen 5 Intel (14") | ThinkPad T480s 2599.99美元；ThinkPad T470s 2699.99美元。 | 0.000 | invalid_judge_json: { "semantic_equivalent": false, "evidence_supports_prediction": false, "missing_key_information": tr |
| dev | pix2fact_926bc1ea82 | phone | ThinkPad T14 Gen 5 Intel (14") | S430 2199 USD | 0.000 | invalid_judge_json: { "semantic_equivalent": false, "evidence_supports_prediction": true, "missing_key_information": fal |
| test | pix2fact_a1b1e2adb8 | title | 11:00–21:00 | 营业时间是：周一至周日 10:00-21:00 周一休息 | 0.500 | 虽然提供的预测答案与证据中的星期二营业时间相符,但黄金答案和预测答案在表述上不一致。黄金答案直接给出了具体的时间,而预测答案则包含了其他未被问及的信息,如一周的营业时间和周一休息日。 |
