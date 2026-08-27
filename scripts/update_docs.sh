SCHEMA_PATH="../MELD/resources/contract.schema.json"
DOCS_BASE_PATH="../docs"
SCHEMA_DOC_PATH="$DOCS_BASE_PATH/contract_schema_reference.md"

if ! python -m pip show json-schema-for-humans >/dev/null 2>&1; then
    echo "Error: 'json-schema-for-humans' is not installed."
    echo "Install it with: pip install json-schema-for-humans"
    exit 1
fi

generate-schema-doc --config template_name=md \
                     --config show_breadcrumbs=false \
                     --config examples_as_yaml \
                     --config no_show_heading_number \
                     "$SCHEMA_PATH" \
                     "$SCHEMA_DOC_PATH"