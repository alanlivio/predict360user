.PHONY: setup build serve

setup:
	pip install zensical nbconvert

build:
	python -m jupyter nbconvert --to html docs/tutorial.ipynb
	zensical build

serve:
	python -m jupyter nbconvert --to html docs/tutorial.ipynb
	zensical serve
