#!/bin/bash
set -e

SUBMISSION_DIR="cloudops-env-submission"
ZIP_NAME="cloudops-env-submission.zip"

rm -rf "$SUBMISSION_DIR" "$ZIP_NAME"
mkdir -p "$SUBMISSION_DIR/env"

cp openenv.yaml      "$SUBMISSION_DIR/"
cp openenv.json      "$SUBMISSION_DIR/"
cp Dockerfile        "$SUBMISSION_DIR/"
cp server.py         "$SUBMISSION_DIR/"
cp inference.py      "$SUBMISSION_DIR/"
cp requirements.txt  "$SUBMISSION_DIR/"
cp README.md         "$SUBMISSION_DIR/"
cp validate-local.sh "$SUBMISSION_DIR/"
cp env/__init__.py   "$SUBMISSION_DIR/env/"
cp env/cloudops_env.py "$SUBMISSION_DIR/env/"

zip -r "$ZIP_NAME" "$SUBMISSION_DIR/"

FILE_COUNT=$(find "$SUBMISSION_DIR" -type f | wc -l | tr -d ' ')
echo "✅ READY: $ZIP_NAME ($FILE_COUNT files)"
