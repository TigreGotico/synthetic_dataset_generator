#!/bin/bash


chatterbox_bulk_tts \
    --output-path "/run/media/miro/endeavouros/ww/vc_synth_output/hey_mycroft" \
    --voices-path "/run/media/miro/endeavouros/ww/hf_datasets/spoken_words_en" \
    --exaggeration-range 0.4 0.9 0.1 \
    "hey mycroft"