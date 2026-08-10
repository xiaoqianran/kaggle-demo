.PHONY: smoke push status logs
smoke:
	./scripts/auto_fix_loop.sh notebooks/Grok-infra-t4x2-smoke
push:
	./scripts/run_on_kaggle_poll.sh notebooks/Grok-infra-t4x2-smoke
status:
	python3.11 -m kaggle kernels status shuhuaqqq/grok-infra-t4x2-smoke
logs:
	python3.11 -m kaggle kernels logs shuhuaqqq/grok-infra-t4x2-smoke | tail -100
