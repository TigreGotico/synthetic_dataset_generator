#!/bin/bash

# NOTE: meant to be used in the output of ovos_ww_synth.sh / record_dataset.py / chatterbox_bulk_tts

# augment an existing dataset by revoicing it

WW="hey_mycroft"

mkdir -p ./logs

chatterbox_bulk_vc \
    --output-path "/run/media/miro/endeavouros/ww/vc_output/${WW}" \
    --audios-path "/run/media/miro/endeavouros/ww/synth_output/en/${WW}" \
    --voices-path "/run/media/miro/endeavouros/ww/hf_datasets/spoken_words_en" \
    --n-random 2