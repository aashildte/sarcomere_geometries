#!/bin/bash

python generate_geometry.py 0 1 True
python generate_geometry.py 0 2 True

for i in 1 2 3 4 5 6 7 8
do
	python generate_geometry.py $i 1 False
	python generate_geometry.py $i 2 False
done
