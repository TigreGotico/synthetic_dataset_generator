#!/bin/bash

# --- ⚙️ Configuration ---
# Max concurrent jobs
MAX_JOBS=2

# --- 🧹 Cleanup Function ---
# This function is executed when the script receives a SIGINT (Ctrl+C)
cleanup() {
  echo ""
  echo "--- 🛑 Interrupted (Ctrl+C) ---"
  # Use 'jobs -p' to get the PIDs of all background jobs started by this shell
  if jobs -r > /dev/null; then
    echo "Killing ongoing background jobs..."
    kill $(jobs -p) 2>/dev/null
    # Wait a moment for processes to terminate
    sleep 1
  fi
  echo "Exiting script."
  exit 1 # Exit with a non-zero status to indicate an error/interruption
}

# --- 🚨 Trap SIGINT (Ctrl+C) ---
# Set the cleanup function to be executed upon receiving SIGINT
trap cleanup SIGINT

# --- ℹ️ Help Text Function ---
show_help() {
  echo "Usage: $0 <LANG> <WAKE_WORD_1> [WAKE_WORD_2] [WAKE_WORD_N...]"
  echo ""
  echo "Synthesize wake word data for a given language."
  echo ""
  echo "Arguments:"
  echo "  <LANG>          The two-letter language code (e.g., en, es, pt)."
  echo "  <WAKE_WORD_N>   One or more wake words to process."
  echo ""
  echo "Example:"
  echo "  $0 en 'hey_floyd' 'hey_ziggy' 'hey_robin'"
}

# --- 🔍 Argument Validation ---
if [ "$#" -lt 2 ]; then
  show_help
  exit 1
fi

# Set the language from the first argument
LANG="$1"
# Shift the arguments so that $@ now contains only the wake words
shift

# The remaining arguments are the wake words
WW_LIST=("$@")

# --- 🛠️ Core Functions ---

# Function to run preprocessing for a single wake word
process_ww() {
  WW="$1"
  LOG_DIR="logs/$LANG"
  mkdir -p "$LOG_DIR"

  echo "🔹 Starting ${WW} ($LANG) ... detailed logs -> ${LOG_DIR}/synth_${WW}.log"

  OUTPUT_PATH="/run/media/miro/endeavouros/ww/synth_output/$LANG/${WW}"

  python ovos_ww_synth.py \
      -w "${WW}"  \
      -l "${LANG}"  \
      -o "${OUTPUT_PATH}"  \
      -n 50  \
      --edge  \
      --google  \
      --vc-refs "/run/media/miro/endeavouros/ww/hf_datasets/spoken_words_en" \
     >"${LOG_DIR}/synth_${WW}.log" 2>&1

  if [ $? -eq 0 ]; then
    echo "✅ Done: $WW"
  else
    # Check if the process failed due to being killed by SIGINT
    # $? is 130 if a background job is killed by SIGINT
    if [ $? -eq 130 ]; then
      echo "⏸️ Aborted: $WW (Interrupted)"
    else
      echo "❌ Failed: $WW (Check ${LOG_DIR}/synth_${WW}.log for details)"
    fi
  fi
}

# --- 🚀 Main Execution ---

echo "--- Starting Synthesis ---"
echo "Language: ${LANG}"
echo "Wake Words: ${WW_LIST[@]}"
echo "Max Concurrent Jobs: ${MAX_JOBS}"
echo "--------------------------"

# Run jobs in parallel (max $MAX_JOBS at a time)
for WW in "${WW_LIST[@]}"; do
  process_ww "$WW" &
  # Limit parallel jobs
  while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
    sleep 1
  done
done

# Wait for all background jobs to finish
wait

echo "🎉 All wake words synthesized (or failed/interrupted)! Check logs for full status."