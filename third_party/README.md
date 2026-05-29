# Third-Party Reference Repositories

The DAG-IG scaffold expects reference repositories to live here when network access is available:

- `IG-Search`
- `multimodal-search-r1`
- `MMSearch-Plus`
- `lmms-eval`

Run:

```bash
bash projects/dagig_mmsearch/scripts/clone_third_party.sh
```

In the current sandbox, GitHub DNS resolution failed and escalated network approval timed out, so the references were not cloned. The local repository already includes `mmsearch_r1/` and `verl/`, which are sufficient for the current DAG-IG-Lite smoke prototype.

