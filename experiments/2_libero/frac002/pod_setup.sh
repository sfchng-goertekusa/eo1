#!/usr/bin/env bash
# One-time setup of a RunPod pod for the EO-1 frac002 run (one demo per LIBERO task, the 40 demos of the flex-pi /
# pi0.5 / FOCA one-demo runs). Everything lands on the network volume (/workspace), so a restarted pod only needs this
# script again for the apt packages (the rest is skipped when present).
#
#   git clone -b frac002 https://<token>@github.com/sfchng-goertekusa/eo1.git /workspace/EO1
#   cd /workspace/EO1 && bash experiments/2_libero/frac002/pod_setup.sh
#   source /workspace/env.sh && tmux new -s run
#   SMOKE=1 bash experiments/2_libero/frac002/run_all.sh        # smoke test
#   STOP_POD=1 bash experiments/2_libero/frac002/run_all.sh
#
# Pod: 4x or 8x H100 80GB (global batch 512 = 64 per GPU x GPUs x accumulation), >= 32 vCPUs, network volume >= 100 GB
# at /workspace, env var NVIDIA_DRIVER_CAPABILITIES=all (EGL rendering for LIBERO). Optional env: HF_TOKEN.
set -euo pipefail

WS="${WS:-/workspace}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# EO-1-3B: the only revision left on the Hub (2026-01-14 upload, code identical to SHAILAB-IPEC/EO1 7ab1501).
EO1_REVISION=fe0a015d9349bdf886d898809213b3574eedf1f9
LIBERO_REVISION=8f1084e3132a39270c3a13ebe37270a43ece2a01
FLASH_ATTN_WHEEL_BASE=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3

apt-get update -qq
apt-get install -y -qq libegl1 libgl1 libglib2.0-0 libosmesa6 ffmpeg tmux git > /dev/null

cat > "${WS}/env.sh" <<ENV
export UV_CACHE_DIR=${WS}/.cache/uv
export HF_HOME=${WS}/.cache/huggingface
export HF_LEROBOT_HOME=${WS}/.cache/lerobot
export LIBERO_CONFIG_PATH=${REPO_DIR}/.libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONNOUSERSITE=1
export EO1_BASE=${WS}/models/EO-1-3B
export PATH=${REPO_DIR}/.venv/bin:\$HOME/.local/bin:\$PATH
eval "\$(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_POD_ID|RUNPOD_API_KEY|HF_TOKEN)=' | sed 's/^/export /' || true)"
ENV
source "${WS}/env.sh"

command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"
cd "${REPO_DIR}"
PY=.venv/bin/python

if [[ ! -x ${PY} ]]; then
  uv venv --python 3.10 --python-preference only-managed .venv
  # torchcodec 0.4 is the release built against torch 2.7 (pyproject pins torch==2.7.0 but leaves torchcodec open).
  uv pip install --python ${PY} -e . "torchcodec==0.4.0"
  # flash-attn 2.8.3 prebuilt for torch 2.7 / CUDA 12 / cp310; the ABI flag must match the torch build.
  abi=$(${PY} -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
  uv pip install --python ${PY} --no-deps \
      "${FLASH_ATTN_WHEEL_BASE}/flash_attn-2.8.3+cu12torch2.7cxx11abi${abi}-cp310-cp310-linux_x86_64.whl"
  # LIBERO simulator + the eval script's own imports (same versions as the FOCA frac002 env).
  uv pip install --python ${PY} robosuite==1.4.0 mujoco==3.3.3 bddl==3.6.0 easydict==1.13 gym==0.26.2 future==1.0.0 \
      cloudpickle matplotlib pyopengl termcolor "imageio[ffmpeg]" tyro
fi

# LIBERO: own copy on the path (its setup.py packages nothing usable), init states loadable with torch >= 2.6.
if [[ ! -d third_party/LIBERO ]]; then
  git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
  git -C third_party/LIBERO checkout -q "${LIBERO_REVISION}"
  sed -i "s/torch.load(init_states_path)$/torch.load(init_states_path, weights_only=False)/" \
      third_party/LIBERO/libero/libero/benchmark/__init__.py
fi
echo "${REPO_DIR}/third_party/LIBERO" > .venv/lib/python3.10/site-packages/libero_src.pth
root="${REPO_DIR}/third_party/LIBERO/libero/libero"
mkdir -p .libero
printf 'assets: %s/assets\nbddl_files: %s/bddl_files\nbenchmark_root: %s\ndatasets: %s/../datasets\ninit_states: %s/init_files\n' \
    "${root}" "${root}" "${root}" "${root}" "${root}" > .libero/config.yaml

# Only the 40 frac002 episodes (parquet + both camera videos) + meta of each suite, and the EO-1-3B weights.
${PY} - <<PY
import json, os
from huggingface_hub import snapshot_download
split = json.load(open("experiments/2_libero/frac002/frac002_episodes.json"))
for suite, eps in split["episodes"].items():
    repo = split["eo_repos"][suite]
    files = [f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet" for e in eps]
    files += [f"videos/chunk-{e // 1000:03d}/{k}/episode_{e:06d}.mp4" for e in eps
              for k in ("observation.images.image", "observation.images.wrist_image")]
    snapshot_download(repo["repo"], repo_type="dataset", revision=repo["revision"], allow_patterns=["meta/*"] + files,
                      local_dir=os.path.join(os.environ["HF_LEROBOT_HOME"], repo["repo"].split("/")[1]))
snapshot_download("IPEC-COMMUNITY/EO-1-3B", revision="${EO1_REVISION}", local_dir=os.environ["EO1_BASE"])
PY

# Checks: GPU + flash-attn, EGL + LIBERO, and the training data: 4/4 suites, 40 episodes, 6,758 frames, subset stats,
# one decoded sample with an 8-step action chunk (the loader skips a dataset that fails to load, so assert all of it).
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
${PY} - <<'PY'
import pathlib, torch, flash_attn
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "flash_attn", flash_attn.__version__)
b = benchmark.get_benchmark_dict()["libero_spatial"](); t = b.get_task(0)
env = OffScreenRenderEnv(bddl_file_name=str(pathlib.Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file),
                         camera_heights=64, camera_widths=64)
env.seed(0); env.reset(); obs = env.set_init_state(b.get_task_init_states(0)[0])
print("LIBERO render ok, mean pixel", float(obs["agentview_image"].mean())); env.close()
PY
${PY} - <<'PY'
import json, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from eo.data.lerobot_dataset import MultiLeRobotDataset
from eo.data.schema import DataConfig
cfg = DataConfig.from_yaml("experiments/2_libero/frac002/data-frac002.yaml")
ds = MultiLeRobotDataset(cfg.lerobot_datasets, video_backend="torchcodec", chunk_size=8)
assert len(ds._datasets) == 4, ds.repo_ids
n_eps = sum(d.num_episodes for d in ds._datasets); n_frames = sum(len(d) for d in ds._datasets)
print(f"datasets {len(ds._datasets)}, episodes {n_eps}, frames {n_frames}")
assert (n_eps, n_frames) == (40, 6758)
for d in ds._datasets:
    full = LeRobotDatasetMetadata(d.repo_id, d.root).stats["action"]["mean"]
    sub = d.meta.stats["action"]["mean"]
    assert not np.allclose(full, sub), f"{d.repo_id}: still the full-dataset stats"
    item = d[len(d) // 2]
    print(d.repo_id, "action chunk", tuple(item["action"].shape), "image", tuple(item["observation.images.image"].shape),
          "subset action mean", np.round(np.asarray(sub), 3).tolist())
PY

echo "setup done. Next: source ${WS}/env.sh && tmux new -s run, then"
echo "  SMOKE=1 bash experiments/2_libero/frac002/run_all.sh"
echo "  STOP_POD=1 bash experiments/2_libero/frac002/run_all.sh"
