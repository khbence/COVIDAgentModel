#!/bin/bash
if [ -f CMakeCache.txt ]; then
    rm CMakeCache.txt
fi
cmake -DCMAKE_BUILD_TYPE=Release ..
