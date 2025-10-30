#!/bin/bash

# NOTE: meant to be used in the output of gen_adversarial_words.sh

# A script to read a file, convert all characters to lowercase, sort the lines,
# deduplicate identical lines, and replace the original file content with the result.

# Check if a filename was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

FILE="$1"
TEMP_FILE=$(mktemp)

# Check if the file exists
if [ ! -f "$FILE" ]; then
    echo "Error: File '$FILE' not found."
    exit 1
fi

echo "Processing $FILE: converting to lowercase, sorting, and deduplicating..."

# 1. Read the file
# 2. Convert all uppercase characters to lowercase (for case-insensitive comparison)
# 3. Sort the lines
# 4. Deduplicate (remove identical lines)
# 5. Redirect the final output into the temporary file
cat "$FILE" | tr '[:upper:]' '[:lower:]' | sort | uniq > "$TEMP_FILE"

# Replace the original file with the processed temporary file
# The 'mv' command handles the replacement safely.
mv "$TEMP_FILE" "$FILE"

echo "Done. '$FILE' has been updated in place."
exit 0
