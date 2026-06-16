#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

executables=(
  01_class_object
  02_raii_unique_ptr
  03_inheritance_polymorphism
  04_template
  05_stl_lambda
  06_modern_cpp
  07_move_semantics
  08_all_in_one
)

for exe in "${executables[@]}"; do
  echo "================ ${exe} ================"
  "${BUILD_DIR}/${exe}"
  echo
done
