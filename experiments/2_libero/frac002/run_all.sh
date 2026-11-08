#!/usr/bin/env bash
# EO-1 on the flex-pi frac002 split (one demo per task, the 40 demos of the flex-pi / pi0.5 / FOCA one-demo runs):
# fine-tune IPEC-COMMUNITY/EO-1-3B with EO-1's LIBERO recipe (experiments/2_libero/train.sh: global batch 512, lr 1e-4,
# merger 1e-4, vision 2e-5, cosine, warm-up 3 %, chunk 8) for 20 epochs over the 6,758 frames (~264 updates), then
# evaluate the final checkpoint on the four LIBERO suites (50 trials per task, replan 8, seed 7). Each suite is split
# over CLIENTS_PER_SUITE eval processes (each loads the model, ~15 GB in fp32), spread over the GPUs.
#
#   bash experiments/2_libero/frac002/run_all.sh                 # seed 42
#   SMOKE=1 bash experiments/2_libero/frac002/run_all.sh         # smoke test: 20 updates, 1 trial/task
#   STOP_POD=1 bash experiments/2_libero/frac002/run_all.sh      # RunPod: stop the pod when everything is done
#
# Run from the repo root inside tmux, after pod_setup.sh and `source /workspace/env.sh`. Idempotent: an evaluated seed
# is skipped, a trained seed is not retrained. Results: results/frac002/<exp>/<suite>_<k>/results.json and
# results/frac002/summary.tsv (exp, suite, successes, episodes, %).
set -euo pipefail

SEEDS="${SEEDS:-42}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
NUM_GPUS=$(nvidia-smi -L | wc -l)
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"  # EO-1 LIBERO recipe: 64 per device x 8 GPUs
PER_DEVICE="${PER_DEVICE:-64}"
EPOCHS="${EPOCHS:-20}"
ATTN="${ATTN:-flash_attention_2}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
NUM_WORKERS="${NUM_WORKERS:-8}"      # data loader processes per GPU
TRIALS="${TRIALS:-50}"
REPLAN="${REPLAN:-8}"
EVAL_SEED="${EVAL_SEED:-7}"
CLIENTS_PER_SUITE="${CLIENTS_PER_SUITE:-${NUM_GPUS}}"  # 4 suites x this, over NUM_GPUS GPUs -> 4 per GPU
NUM_TASKS=10                         # every suite above has 10 tasks
EO1_BASE="${EO1_BASE:?source /workspace/env.sh first}"
RESULTS="${RESULTS:-results/frac002}"
STOP_POD="${STOP_POD:-0}"
SMOKE="${SMOKE:-0}"
STEPS_FLAG=()
if [[ "${SMOKE}" == "1" ]]; then
  SEEDS=42 TRIALS=1 RESULTS="${RESULTS}_smoke" STOP_POD=0 STEPS_FLAG=(--max-steps 20)
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false

ACCUM=$((GLOBAL_BATCH / (PER_DEVICE * NUM_GPUS)))
if (( ACCUM < 1 || PER_DEVICE * NUM_GPUS * ACCUM != GLOBAL_BATCH )); then
  echo "ERROR: global batch ${GLOBAL_BATCH} is not ${PER_DEVICE} x ${NUM_GPUS} GPUs x an integer accumulation" >&2; exit 1
fi
[[ -f "${LIBERO_CONFIG_PATH:-}/config.yaml" ]] || { echo "ERROR: run pod_setup.sh first (no LIBERO config)" >&2; exit 1; }
ACCEL_ARGS=(--num_machines 1 --machine_rank 0 --num_processes "${NUM_GPUS}")
(( NUM_GPUS > 1 )) && ACCEL_ARGS+=(--multi_gpu)
mkdir -p "${RESULTS}"

for SEED in ${SEEDS}; do
  EXP="frac002_s${SEED}"
  [[ "${SMOKE}" == "1" ]] && EXP="${EXP}_smoke"
  OUT="${RESULTS}/${EXP}"
  TRAIN_DIR="outputs/frac002/${EXP}"
  mkdir -p "${OUT}"
  if [[ -f "${OUT}/EVALUATED" ]]; then echo "[run_all] ${EXP} already evaluated, skipping"; continue; fi

  if [[ ! -f "${OUT}/TRAINED" ]]; then
    echo "[run_all] train ${EXP}: ${EPOCHS} epochs, batch ${PER_DEVICE} x ${NUM_GPUS} GPUs x ${ACCUM} ($(date))"
    rm -rf "${TRAIN_DIR}"  # never resume a half-finished run (train.py resumes from any checkpoint-* it finds)
    accelerate launch "${ACCEL_ARGS[@]}" scripts/train.py \
        --model-name-or-path "${EO1_BASE}" \
        --data-path experiments/2_libero/frac002/data-frac002.yaml \
        --lerobot-data-video-backend "${VIDEO_BACKEND}" \
        --output-dir "${TRAIN_DIR}" --run-name "${EXP}" --seed "${SEED}" \
        --chunk-size 8 \
        --dataloader-num-workers "${NUM_WORKERS}" \
        --freeze-vision-tower False --freeze-llm False --freeze-merger False \
        --bf16 True --tf32 True --fp16 False \
        --num-train-epochs "${EPOCHS}" "${STEPS_FLAG[@]}" \
        --per-device-train-batch-size "${PER_DEVICE}" --gradient-accumulation-steps "${ACCUM}" \
        --learning-rate 1e-4 --merger-lr 1e-4 --vision-lr 2e-5 \
        --weight-decay 0.1 --warmup-ratio 0.03 --lr-scheduler-type cosine \
        --gradient-checkpointing True \
        --save-strategy no --logging-steps 10 --report-to none \
        --attn-implementation "${ATTN}" > "${OUT}/train.log" 2>&1
    ls -d "${TRAIN_DIR}"/checkpoint-final-* > /dev/null || { echo "ERROR: no final checkpoint, see ${OUT}/train.log" >&2; exit 1; }
    cp "${TRAIN_DIR}/trainer_state.json" "${OUT}/" 2>/dev/null || true
    touch "${OUT}/TRAINED"
  fi
  CKPT=$(ls -d "${TRAIN_DIR}"/checkpoint-final-* | sort -V | tail -1)
  echo "${CKPT}" > "${OUT}/CHECKPOINT"
  if [[ "${TRAIN_ONLY:-0}" == "1" ]]; then continue; fi

  echo "[run_all] eval ${EXP} from ${CKPT}, ${TRIALS} trials/task, ${CLIENTS_PER_SUITE} clients/suite ($(date))"
  per_client=$(( (NUM_TASKS + CLIENTS_PER_SUITE - 1) / CLIENTS_PER_SUITE ))
  rm -rf "${OUT}"/libero_*_[0-9]*  # a rerun starts the eval over (the summary must not mix two client layouts)
  pids=() i=0
  for SUITE in ${SUITES}; do
    for ((k = 0; k < CLIENTS_PER_SUITE; k++)); do
      start=$((k * per_client)); end=$((start + per_client))
      (( start >= NUM_TASKS )) && continue
      CUDA_VISIBLE_DEVICES=$((i % NUM_GPUS)) python experiments/2_libero/eval_libero.py \
          --args.pretrained-path "${CKPT}" --args.task-suite-name "${SUITE}" \
          --args.job-name "${SUITE}_${k}" --args.out-dir "${OUT}" \
          --args.task-start "${start}" --args.task-end "${end}" \
          --args.num-trials-per-task "${TRIALS}" --args.replan-steps "${REPLAN}" --args.seed "${EVAL_SEED}" \
          > "${OUT}/${SUITE}_${k}.stdout" 2>&1 &
      pids+=($!); i=$((i + 1))
    done
  done
  status=0
  for pid in "${pids[@]}"; do wait "${pid}" || status=1; done

  # Per suite: successes / episodes summed over its clients' results.json; every client must have finished its range.
  python - "${OUT}" "${EXP}" "${SUITES}" "${TRIALS}" "${NUM_TASKS}" <<'PY' | tee -a "${RESULTS}/summary.tsv" || status=1
import json, pathlib, sys
out, exp, suites, trials, n_tasks = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(), int(sys.argv[4]), int(sys.argv[5])
ok = True
for suite in suites:
    parts = [json.loads(p.read_text()) for p in sorted(out.glob(f"{suite}_*/results.json"))]
    succ, eps = sum(p["successes"] for p in parts), sum(p["episodes"] for p in parts)
    if not all(p["done"] for p in parts) or eps != trials * n_tasks:
        ok = False
        print(f"[run_all] WARNING: {suite} incomplete ({eps} of {trials * n_tasks} episodes)", file=sys.stderr)
    print(f"{exp}\t{suite}\t{succ}\t{eps}\t{100 * succ / eps:.1f}" if eps else f"{exp}\t{suite}\t0\t0\tNA")
sys.exit(0 if ok else 1)
PY

  if [[ ${status} -eq 0 ]]; then touch "${OUT}/EVALUATED"
  else echo "[run_all] WARNING: eval of ${EXP} incomplete, see ${OUT}/*.stdout and */eval.log (rerun to retry)" >&2; fi
done

echo "[run_all] done ($(date)); summary in ${RESULTS}/summary.tsv (exp, suite, successes, episodes, %)"
if [[ "${STOP_POD}" == "1" && -n "${RUNPOD_POD_ID:-}" ]]; then
  runpodctl stop pod "${RUNPOD_POD_ID}"
fi
