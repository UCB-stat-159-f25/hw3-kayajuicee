ENV_NAME := ligo159

.PHONY: env html clean

env:
	@if [ -f environment.yml ]; then \
		echo "Creating/Updating conda env $(ENV_NAME) from environment.yml"; \
		conda env create -n $(ENV_NAME) -f environment.yml 2>/dev/null || \
		conda env update -n $(ENV_NAME) -f environment.yml; \
	else \
		echo "No environment.yml found. Skipping."; \
	fi
	python -m pip install -e .
	python -m pip install mystmd

html:
	myst build --html

clean:
	rm -rf figures/* audio/* _build
