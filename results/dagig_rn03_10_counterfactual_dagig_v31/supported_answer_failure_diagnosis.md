# Supported-Answer Failure Diagnosis

## Old vs v2 Supported-Answer Rates

| split | n | old_supported_answer | supported_answer_v2 | old_false_v2_true | old_true_v2_false |
| --- | --- | --- | --- | --- | --- |
| train | 1832 | 0.09989082969432314 | 0.0212882096069869 | 0.0032751091703056767 | 0.08187772925764192 |
| dev | 392 | 0.10204081632653061 | 0.015306122448979591 | 0.007653061224489796 | 0.09438775510204081 |
| test | 256 | 0.08203125 | 0.01171875 | 0.00390625 | 0.07421875 |

## Case Sampling Summary

| split | q25 | q75 | high_DAGIG_false_supported_answer | low_DAGIG_true_supported_answer | evidence_support_true_but_answer_wrong | answer_correct_but_evidence_support_false | examples_written |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dev | -0.08654705588389142 | 0.9448574059569955 | 84 | 7 | 264 | 7 | 74 |
| test | -0.12166583167975341 | 0.5631775893895881 | 61 | 4 | 165 | 2 | 66 |

## Failure Cause Breakdown

| split | failure_cause | count | rate_among_old_unsupported |
| --- | --- | --- | --- |
| train | answer_string_or_value_mismatch | 1128 | 0.6840509399636143 |
| train | query_retrieves_support_but_evidence_selection_bad | 180 | 0.1091570648878108 |
| train | answer_format_mismatch | 135 | 0.0818677986658581 |
| train | evidence_not_supporting_answer | 96 | 0.05821710127349909 |
| train | answer_extracted_incorrectly | 75 | 0.04548211036992116 |
| train | correct_evidence_but_wrong_final_answer | 24 | 0.014554275318374773 |
| train | evidence_verifier_too_weak | 6 | 0.0036385688295936932 |
| train | correct_answer_but_unsupported_evidence | 5 | 0.0030321406913280777 |
| dev | answer_string_or_value_mismatch | 236 | 0.6704545454545454 |
| dev | query_retrieves_support_but_evidence_selection_bad | 40 | 0.11363636363636363 |
| dev | evidence_not_supporting_answer | 24 | 0.06818181818181818 |
| dev | answer_format_mismatch | 23 | 0.06534090909090909 |
| dev | answer_extracted_incorrectly | 20 | 0.056818181818181816 |
| dev | correct_answer_but_unsupported_evidence | 3 | 0.008522727272727272 |
| dev | correct_evidence_but_wrong_final_answer | 3 | 0.008522727272727272 |
| dev | evidence_verifier_too_weak | 3 | 0.008522727272727272 |
| test | answer_string_or_value_mismatch | 161 | 0.6851063829787234 |
| test | evidence_not_supporting_answer | 24 | 0.10212765957446808 |
| test | query_retrieves_support_but_evidence_selection_bad | 24 | 0.10212765957446808 |
| test | answer_format_mismatch | 14 | 0.059574468085106386 |
| test | answer_extracted_incorrectly | 9 | 0.03829787234042553 |
| test | correct_evidence_but_wrong_final_answer | 1 | 0.00425531914893617 |
| test | correct_answer_but_unsupported_evidence | 1 | 0.00425531914893617 |
| test | evidence_verifier_too_weak | 1 | 0.00425531914893617 |
