SHELL:=/bin/bash
export SHELLOPTS:=errexit:pipefail
.SECONDARY:



test-target: test-2020-target  test-2022-target
test-2020-target:
	python3 hipe_evaluation/tests/unittest_eval.py -v

test-2022-target:
	python3 hipe_evaluation/tests/unittest_eval_2022.py -v

test-clean:
	rm -v hipe_evaluation/tests/data/*.tmp
