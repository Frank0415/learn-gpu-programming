.PHONY: all correctness performance sweep profile

all:
	CXX=/usr/bin/g++-15 uv run ../02-matmul-test/main.py main.py

correctness:
	CXX=/usr/bin/g++-15 uv run ../02-matmul-test/main.py main.py --mode correctness

performance: sweep

sweep:
	CXX=/usr/bin/g++-15 uv run python run_performance_sweep.py

profile:
	SUDO_ASKPASS=/usr/bin/ksshaskpass SSH_ASKPASS_REQUIRE=force CXX=/usr/bin/g++-15 uv run python run_performance_sweep.py --profile --sudo-ncu
