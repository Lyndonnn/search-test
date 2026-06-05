# Third-Party Reference Repositories

The DAG-IG scaffold expects reference repositories to live here when network access is available:

- `IG-Search`
- `multimodal-search-r1`
- `MMSearch-Plus`
- `lmms-eval`
- `mmsearch_r1_verl`: exact veRL commit pinned by the original MMSearch-R1 repository

Run:

```bash
bash projects/dagig_mmsearch/scripts/clone_third_party.sh
```

In the current sandbox, GitHub DNS resolution failed and escalated network approval timed out, so the references were not cloned. The local repository already includes `mmsearch_r1/` and `verl/`, which are sufficient for the current DAG-IG-Lite smoke prototype.

For the paper-baseline MMSearch-R1 training path, do not use the newer vendored
`./verl`. Run `make mmsearch_setup_baseline`; it checks out commit
`8e9e73723fd1cc729bedb3bbcf915060afbda91d` into
`third_party/mmsearch_r1_verl/` and creates the isolated `.venv-mmsearch-r1`.
