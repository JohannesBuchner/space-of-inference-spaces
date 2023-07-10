PYTHON := python3

.PHONY: all help # rules that do not correspond to a output file
.SUFFIXES: # disable built-in rules
.SECONDARY: # do not delete intermediate products

OUTPUTS := evaluateproblems.tex evaluateproblems_spacestructure.pdf evaluateproblems.pdf evaluateproblems_volcurve.pdf reference-run-output.tar.gz

all: ${OUTPUTS}
clean: 
	rm -f ${OUTPUTS}

evaluateproblems.tex evaluateproblems_spacestructure.pdf evaluateproblems.pdf evaluateproblems_volcurve.pdf: evaluateproblems.py problems.txt
	${PYTHON} $^

reference-run-output.tar.gz: problems.txt
	grep -v '^#' < problems.txt |cut -d '	' -f3 |xargs tar -czvf $@
