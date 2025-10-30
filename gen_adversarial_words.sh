#!/bin/bash

# create adversarial words, meant to be fed into a TTS stage

WW="alexa"

python adversarial_samples.py \
    -w "${WW}" \
    -o "/run/media/miro/endeavouros/ww/adversarial/${WW}.txt" \
    -n 20 \
    -u "http://100.88.41.41:11434" \
    -m "gemma3:4b" \
    --llm-weight 1.0 \
    --grapheme-weight 0.0