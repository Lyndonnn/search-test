## Data Notes

This repository intentionally checks in only the small sanity dataset:

- `mini_data.pq`: a tiny veRL-format parquet used for M0 sanity/debug runs
- `data_example.ipynb`: a notebook showing the expected veRL parquet schema

The larger `mmsearch_r1_infoseek_sub_2k.parquet` file is not versioned in this repo.
If you need a larger training/eval parquet, prepare it locally from the upstream data
source and place it under `mmsearch_r1/data/` or another local path.

Why this file is not included:

- It is large enough to make repository cloning fragile
- A previous Git LFS pointer for that file broke fresh clones because the LFS object
  was unavailable on the server

Recommended workflow:

1. Use `mini_data.pq` for sanity checks
2. Generate or download your own larger parquet locally
3. Keep that local parquet out of Git unless you have a verified storage plan
