#!/bin/bash

case "${1:-all}" in
  all)
    .venv/bin/pytest tests/ -v
    ;;
  fast)
    .venv/bin/pytest tests/ -v --ignore=tests/test_evaluation.py
    ;;
  *)
    .venv/bin/pytest "tests/test_${1}.py" -v "${@:2}"
    ;;
esac
