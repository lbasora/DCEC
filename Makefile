
#include config

PYTHON=python3


DATA_PATH = ./hypers

JOBS = $(foreach i,$(shell seq 0 10),$(DATA_PATH)/$i.csv)

all: $(JOBS)

$(DATA_PATH)/%:
	@mkdir -p $(DATA_PATH)/
	@echo "===Searching for hyperparameter==="
	@echo $@
	$(PYTHON) ./dcec_test.py --search $(notdir $(basename $@)) --csvlog $@

