.PHONY: deps setup build serve clean

deps: setup

setup:
	pip install zensical nbconvert

build:
	python -m jupyter nbconvert --to html docs/tutorial.ipynb
	zensical build

serve:
	python -m jupyter nbconvert --to html docs/tutorial.ipynb
	zensical serve

clean:
	rm -rf site docs/tutorial.html

