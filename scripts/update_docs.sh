#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SCHEMA_PATH="$SCRIPT_DIR/../MELD/resources/contract.schema.json"
DOCS_BASE_PATH="$SCRIPT_DIR/../docs"
SCHEMA_DOC_PATH="$DOCS_BASE_PATH/generated/contract_schema_reference.md"
MODEL_OUTPUT_PATH="$SCRIPT_DIR/../MELD/ModelManager/generated"
MODEL_TEMP_ROOT=$(mktemp -d)
MODEL_TEMP_PATH="$MODEL_TEMP_ROOT/generated"

cleanup() {
    rm -rf "$MODEL_TEMP_ROOT"
}

trap cleanup EXIT

mkdir -p "$DOCS_BASE_PATH/generated" "$MODEL_TEMP_PATH"

if ! python -m pip show json-schema-for-humans >/dev/null 2>&1; then
    echo "Error: 'json-schema-for-humans' is not installed."
    echo "Install it with: pip install json-schema-for-humans"
    exit 1
fi

if ! python -m pip show datamodel-code-generator >/dev/null 2>&1; then
    echo "Error: 'datamodel-code-generator' is not installed."
    echo "Install it with: pip install datamodel-code-generator"
    exit 1
fi

generate-schema-doc --config template_name=md \
                     --config show_breadcrumbs=false \
                     --config examples_as_yaml \
                     --config no_show_heading_number \
                     "$SCHEMA_PATH" \
                     "$SCHEMA_DOC_PATH"

datamodel-codegen \
    --input "$SCHEMA_PATH" \
    --input-file-type jsonschema \
    --output "$MODEL_TEMP_PATH" \
    --output-model-type dataclasses.dataclass \
    --module-split-mode single \
    --all-exports-scope recursive \
    --all-exports-collision-strategy minimal-prefix \
    --use-exact-imports \
    --strict-refs \
    --disable-timestamp

rm -rf "$MODEL_OUTPUT_PATH"
mv "$MODEL_TEMP_PATH" "$MODEL_OUTPUT_PATH"
