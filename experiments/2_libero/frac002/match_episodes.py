"""Map the flex-pi frac002 split onto episode indices of EO-1's LIBERO copies (IPEC-COMMUNITY/libero_*_lerobot).

The split is taken from FOCA's frac002_episodes.json, which carries openpi's map of each flex-pi demo to its exact
twin in physical-intelligence/libero (one swap, recorded there) plus the FOCA twin; this script finds the same demo
in EO-1's per-suite copies so all baselines train on the same 40 demonstrations. A demo is identified by its task
string plus its arm actions (first 6 dims; the gripper sign convention differs between copies): an exact float32
match, searched among episodes of the same task and length. Only the action column is read, straight from the Hub,
so the datasets do not have to be downloaded.

    python experiments/2_libero/frac002/match_episodes.py \
        --source-split /data/shinfang_chng/FOCA/myutils/frac002/frac002_episodes.json \
        --openpi       $HF_LEROBOT_HOME/physical-intelligence/libero \
        --out          experiments/2_libero/frac002/frac002_episodes.json
"""

import argparse
import concurrent.futures
import hashlib
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

EO_REPOS = {
    "libero_spatial": "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object": "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
    "libero_goal": "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
    "libero_10": "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
}


def arm_actions(table, column: str) -> np.ndarray:
    return np.asarray(table.column(column).to_pylist(), dtype=np.float32)[:, :6]


def digest(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-split", type=pathlib.Path, required=True)
    ap.add_argument("--openpi", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    api, fs = HfApi(), HfFileSystem()
    repos = {}
    for suite, repo in EO_REPOS.items():
        rev = api.dataset_info(repo).sha
        root = f"datasets/{repo}@{rev}"
        info = json.loads(fs.read_text(f"{root}/meta/info.json"))
        eps = [json.loads(line) for line in fs.read_text(f"{root}/meta/episodes.jsonl").splitlines()]
        repos[suite] = dict(repo=repo, revision=rev, root=root, info=info, eps=eps)

    def eo_actions(suite: str, idx: int) -> np.ndarray:
        r = repos[suite]
        rel = r["info"]["data_path"].format(episode_chunk=idx // r["info"]["chunks_size"], episode_index=idx)
        with fs.open(f'{r["root"]}/{rel}', block_size=64 * 1024, cache_type="bytes") as f:
            return arm_actions(pq.read_table(f, columns=["action"]), "action")

    op_info = json.loads((args.openpi / "meta/info.json").read_text())
    rows = json.loads(args.source_split.read_text())["rows"]
    for row in rows:
        suite, idx = row["suite"], row["openpi_episode"]
        rel = op_info["data_path"].format(episode_chunk=idx // op_info["chunks_size"], episode_index=idx)
        a = arm_actions(pq.read_table(args.openpi / rel, columns=["actions"]), "actions")
        cands = [e["episode_index"] for e in repos[suite]["eps"] if e["tasks"][0] == row["task"] and e["length"] == len(a)]
        with concurrent.futures.ThreadPoolExecutor(16) as pool:
            cand_actions = dict(zip(cands, pool.map(lambda i: eo_actions(suite, i), cands)))
        hits = [i for i, c in cand_actions.items() if digest(c) == digest(a)]
        if hits:
            row.update(eo_episode=hits[0], eo_match="exact" if len(hits) == 1 else f"exact x{len(hits)}")
        else:
            diffs = {i: float(np.abs(c - a).max()) for i, c in cand_actions.items()}
            best = min(diffs, key=diffs.get) if diffs else None
            row.update(eo_episode=None, eo_match="none", eo_nearest=best,
                       eo_nearest_max_abs_diff=diffs.get(best), eo_same_length_candidates=len(cands))
        print(f'{suite:15s} op {idx:5d} len {len(a):4d} cands {len(cands):2d} -> EO {row.get("eo_episode")} '
              f'{row["eo_match"]}  {row["task"][:60]}')

    exact = [r for r in rows if r["eo_match"].startswith("exact")]
    print(f"{len(exact)}/{len(rows)} exact EO matches, {sum(r['length'] for r in exact)} frames")
    args.out.write_text(json.dumps({
        "source_split": str(args.source_split),
        "eo_repos": {s: {"repo": r["repo"], "revision": r["revision"]} for s, r in repos.items()},
        "episodes": {s: sorted(r["eo_episode"] for r in exact if r["suite"] == s) for s in EO_REPOS},
        "rows": rows,
    }, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
