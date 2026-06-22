# DAG-IG v1 Failure Diagnosis

## Predictive Edge Credits

| split | comparison | spearman | auc | top_bottom_gap |
| --- | --- | --- | --- | --- |
| train | R_ground vs IoU | 0.5514200523848587 |  | 0.2896905071774639 |
| train | R_ground vs center-hit | 0.5504590359785035 | 0.8289357212471277 | 0.6943231441048034 |
| dev | R_ground vs IoU | 0.49042859454876014 |  | 0.24691815246991733 |
| dev | R_ground vs center-hit | 0.3995751905969366 | 0.735810534912209 | 0.5612244897959183 |
| dev | R_evidence vs evidence_support | 0.30375624799267154 | 0.6569790491293342 | 0.2959183673469388 |
| test | R_ground vs IoU | 0.5219356807774271 |  | 0.22895572973805337 |
| test | R_ground vs center-hit | 0.5224432612163559 | 0.8087820512820513 | 0.640625 |
| test | R_evidence vs evidence_support | 0.40718328252698804 | 0.7049512987012987 | 0.453125 |

## Weak Edge Credits

| split | comparison | spearman | auc | top_bottom_gap |
| --- | --- | --- | --- | --- |
| train | DAGIG_total vs R@5 | 0.2191956435499467 | 0.630158047183276 | 0.28285077951002224 |
| train | DAGIG_total vs evidence_support | 0.20563140010358996 | 0.6279118674552938 | 0.24454148471615722 |
| train | DAGIG_total vs answer F1 | 0.05959912458370069 |  | 0.027098376784893866 |
| train | R_search vs R@5 | 0.2367650527591232 | 0.626685679204097 | 0.2962138084632516 |
| train | R_search vs MRR | 0.21775573634122214 |  | 0.23010524896766577 |
| dev | DAGIG_total vs R@5 | 0.11585869131348628 | 0.5690376569037657 | 0.12631578947368421 |
| dev | DAGIG_total vs evidence_support | 0.13034239852867754 | 0.5814400292763259 | 0.09183673469387754 |
| dev | DAGIG_total vs answer F1 | 0.03661677083877286 |  | 0.018143960442722068 |
| dev | R_search vs R@5 | 0.031121432260168155 | 0.5164167828916782 | 0.0 |
| dev | R_search vs MRR | 0.022243104221482568 |  | 0.005385280343657484 |
| test | DAGIG_total vs R@5 | 0.2309358275718541 | 0.6347676531671859 | 0.2419354838709678 |
| test | DAGIG_total vs evidence_support | 0.20549226937205556 | 0.6248647186147186 | 0.171875 |
| test | DAGIG_total vs answer F1 | -0.055764381261853756 |  | -0.0670763652751944 |
| test | R_search vs R@5 | 0.18154160270165307 | 0.5942367601246106 | 0.3225806451612903 |
| test | R_search vs MRR | 0.13975319300698488 |  | 0.18683448775428863 |

## Generic Process vs Counterfactual DAG-IG v1

| split | metric | generic | counterfactual | delta_cf_minus_generic |
| --- | --- | --- | --- | --- |
| dev | R@5 | 0.6354166666666666 | 0.6185567010309279 | -0.01685996563573877 |
| dev | MRR | 0.5256219027454987 | 0.5132327857174049 | -0.012389117028093755 |
| dev | evidence_support | 0.7551020408163265 | 0.7346938775510204 | -0.020408163265306034 |
| test | R@5 | 0.578125 | 0.5714285714285714 | -0.006696428571428603 |
| test | MRR | 0.47003739818295737 | 0.45708287813550974 | -0.012954520047447626 |
| test | evidence_support | 0.671875 | 0.65625 | -0.015625 |

## Diagnosis

- Strongest part: grounding edge; R_ground predicts IoU/center-hit.
- Moderately useful part: evidence edge.
- Weak part: search edge and total DAG-IG score, especially on dev.
- Generic-process matches or beats v1 because it directly rewards retrieval and evidence support, while v1 search credit is a weak lexical proxy.
- More training is not justified until counterfactual quality, search counterfactuals, and evidence verification are fixed.
