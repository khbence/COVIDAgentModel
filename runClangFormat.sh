#!/bin/bash

function runClang() {
    files=("$@")
    for file in "${files[@]}"
    do
        clang-format -i $file
    done
}

readarray -d '' cppFiles < <(find src -type f -name "*.cpp")
readarray -d '' hFiles < <(find include -type f -name "*.h")

runClang "${cppFiles[@]}"
runClang "${hFiles[@]}"