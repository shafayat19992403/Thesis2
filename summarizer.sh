#!/bin/bash

DIR_PATH="Figures/ConfigTexts"
OUTPUT_PATH="Figures/ConfigTexts/OutputTexts"

# Find .txt files with lines containing "E1" or "E2"
grep -rE "E1|E2|E3" "$DIR_PATH"/*.txt > "$OUTPUT_PATH"/error_log.txt

# Find .txt files with lines containing "S1" and print it in success.txt
grep -rE "S1|S0|S2" "$DIR_PATH"/*.txt > "$OUTPUT_PATH"/success_log.txt


# only keep the first line from every file in success_log.txt
awk '!a[$1]++' "$OUTPUT_PATH"/success_log.txt > "$OUTPUT_PATH"/success_log_unique.txt