#!/bin/bash
rm -rf docs/*
pdoc --html -o docs  --config show_source_code=False --force deep_mri
mv docs/deep_mri/* docs/
rm -d docs/deep_mri
