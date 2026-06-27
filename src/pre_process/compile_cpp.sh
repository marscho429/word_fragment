#!/bin/bash

set -e

echo "Checking for pybind11..."
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "pybind11 not found. Installing..."
    pip install pybind11
fi

echo "Compiling protein_score_cpp.cpp..."
c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    src/pre_process/protein_score_cpp.cpp \
    -o src/pre_process/protein_score_cpp$(python3-config --extension-suffix)

echo "Compilation complete!"