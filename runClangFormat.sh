#!/bin/bash

find src -type f -name "*.cpp" -execdir clang-format -i '{}' ';'
find include -type f -name "*.h" -execdir clang-format -i '{}' ';'