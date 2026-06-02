PYTHON ?= python3
DAGIG_ROOT := projects/dagig_mmsearch
DAGIG_SRC := $(DAGIG_ROOT)/src
export PYTHONPATH := $(DAGIG_SRC):$(PYTHONPATH)

.PHONY: setup audit prepare_data prepare_real_data build_indexes smoke autodl_check hf_probe reference_logprob_smoke reference_ablation agent_rollout_smoke model_agent_rollout eval_nosearch eval_prompted train_outcome train_local_ig train_dagig_lite eval_all make_tables make_figures

setup:
	bash $(DAGIG_ROOT)/scripts/setup_autodl_a800.sh

audit:
	mkdir -p REPO_AUDIT
	find . -path './.git' -prune -o -path './.pycache' -prune -o -path './verl/.git' -prune -o -print > REPO_AUDIT/tree.txt

prepare_data:
	bash $(DAGIG_ROOT)/scripts/prepare_data.sh

prepare_real_data:
	bash $(DAGIG_ROOT)/scripts/prepare_real_vqa_small.sh

build_indexes:
	bash $(DAGIG_ROOT)/scripts/build_indexes.sh

smoke:
	$(PYTHON) -m unittest discover -s $(DAGIG_ROOT)/tests -p 'test_*.py'

autodl_check:
	bash $(DAGIG_ROOT)/scripts/check_gpu_env.sh

hf_probe:
	bash $(DAGIG_ROOT)/scripts/check_hf_access.sh

reference_logprob_smoke:
	bash $(DAGIG_ROOT)/scripts/run_reference_logprob_smoke.sh

reference_ablation:
	bash $(DAGIG_ROOT)/scripts/run_reference_ablation.sh

agent_rollout_smoke:
	bash $(DAGIG_ROOT)/scripts/run_agent_rollout_smoke.sh

model_agent_rollout:
	bash $(DAGIG_ROOT)/scripts/run_model_agent_rollout.sh

eval_nosearch:
	bash $(DAGIG_ROOT)/scripts/run_direct_vqa.sh

eval_prompted:
	bash $(DAGIG_ROOT)/scripts/run_prompted_search.sh

train_outcome:
	bash $(DAGIG_ROOT)/scripts/run_outcome_rl.sh

train_local_ig:
	bash $(DAGIG_ROOT)/scripts/run_local_ig.sh

train_dagig_lite:
	bash $(DAGIG_ROOT)/scripts/run_dagig_lite.sh

eval_all:
	bash $(DAGIG_ROOT)/scripts/run_eval_all.sh

make_tables:
	bash $(DAGIG_ROOT)/scripts/make_tables.sh

make_figures:
	bash $(DAGIG_ROOT)/scripts/make_figures.sh
