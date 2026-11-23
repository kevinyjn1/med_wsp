#!/bin/bash
# Queue of GPU sweeps that do not need Astra, in priority order. Run from anywhere:  bash scripts/queue_sweeps.sh
#
# GPU locks (wsp/gpu_lock.py): the patching and looping scripts take the locks themselves, once per question or
# configuration, so an Astra lab pass can run in between. Do not wrap those in the lock command: a wrapper holds the
# lock for the whole run (on 2026-11-23 that kept a lab session waiting for 6.5 hours). The shorter scoring and
# generation steps still run under the "batch" lock as a whole.
# Each step writes logs/<name>.log and ends it with "exit <code>". A failed step does not stop the queue.
# Every job keeps the Mac awake while on AC power (wsp/keep_awake.py); a closed lid or an empty battery still stops it.
#
# This is bash on purpose: the command prefix is an array. In zsh an unquoted "$L cmd" is NOT split into words.
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

PY=.venv/bin/python
BATCH=("$PY" -m wsp.gpu_lock --name batch --)

step() { # step <log name> <command...>
  local log="logs/$1.log"
  shift
  echo "[$(date +%H:%M)] start: $*"
  "$@" >"$log" 2>&1
  local code=$?
  echo "exit $code" >>"$log"
  echo "[$(date +%H:%M)] exit $code: $log"
}

# EXP-05b: copy one layer's weights from MediPhi-Instruct into Phi-3.5 (bare prompt), then activation patching.
step exp05_patch_bare_weights "$PY" experiments/exp_05_medical_circuit/patch.py \
  --donor mediphi_instruct --style bare --per-set 60 --mode weights
step exp05_patch_bare_activation "$PY" experiments/exp_05_medical_circuit/patch.py \
  --donor mediphi_instruct --style bare --per-set 60 --mode activation

# EXP-07: layer-looping scan on Phi-3.5, 600 dev questions.
step exp07_scan "$PY" experiments/exp_07_layer_looping/run.py \
  --model phi --split dev --limit 600 --sweep defaults position

# EXP-04c control: the bare question inside the chat template, for the two medical models.
for model in mediphi mediphi_instruct; do
  step "exp04_lmeval_chat_$model" "${BATCH[@]}" "$PY" experiments/exp_04_medqa_baseline/run_lmeval_style.py \
    --model "$model" --variants chat
done

# EXP-03b: does MIT's published Qwen edit change behaviour here? 48 GSM8K problems, base vs edited.
for model in base p50c; do
  step "exp03_eval_${model}_gsm8k" "${BATCH[@]}" "$PY" experiments/exp_03_mit8_math/eval_mlx.py \
    --model "$model" --set gsm8k --n 48 --max-tokens 16384 --batch-size 12
done

echo "=== queue done ==="
grep -H -E "^exit|Traceback" logs/exp05_patch_bare_weights.log logs/exp05_patch_bare_activation.log logs/exp07_scan.log \
  logs/exp04_lmeval_chat_mediphi.log logs/exp04_lmeval_chat_mediphi_instruct.log \
  logs/exp03_eval_base_gsm8k.log logs/exp03_eval_p50c_gsm8k.log
