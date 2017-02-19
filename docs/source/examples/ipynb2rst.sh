#! /bin/bash

cp -i ../../../examples/*/*.ipynb .
jupyter nbconvert --to rst *.ipynb
rm *.ipynb

exit 0
