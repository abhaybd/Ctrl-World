"""
Web interface to manually label world model rollouts (logged by rollout_interact_molmobot_pi0.py) as success or failure.

Shows a random rollout from the wandb project that hasn't been labeled yet (no success/success_rate in its summary),
with its task and the cameras side by side in one video, so they play in sync. The policy and world model are hidden
so labels are blind, they (and the run id) can be revealed for debugging. Labels are written to the run summary like
the real rollouts: success (bool) and success_rate (0/1). The real vs world model video isn't shown since the real
rollout gives away its outcome.

    python scripts/label_wm_rollouts.py --project adeshpande-princeton-university/synthetic-wm-evals-wm

then open http://localhost:8000 (forward the port if running remotely). Keyboard: s = success, f = failure.
"""
import os
import random
import threading
from argparse import ArgumentParser
from pathlib import Path

import mediapy
import numpy as np
import wandb
from flask import Flask, abort, jsonify, request, send_file

UNLABELED = {"$and": [
    {"state": "finished"},
    {"summary_metrics.success": {"$exists": False}},
    {"summary_metrics.success_rate": {"$exists": False}},
]}
HIDDEN_VIDEOS = {"real_vs_wm"}


def world_model_name(ckpt_path):
    if "synthwm" in ckpt_path:
        return "synthwm"
    if "yjguo/Ctrl-World" in ckpt_path:
        return "ctrl-world"
    return "unknown"


class Labeler:
    def __init__(self, project, cache_dir):
        self.project = project
        self.cache_dir = Path(cache_dir)
        self.api = wandb.Api()
        self.lock = threading.Lock()
        self.labeled = set()  # labeled here, in case the wandb filter lags behind the summary update
        self.prefetched = None
        self.videos = {}  # {run id: (cameras, local path of the side by side video)}
        self.video_locks = {}

    def run(self, run_id):
        run = self.api.run(f"{self.project}/{run_id}")
        run.load(force=True)
        return run

    def unlabeled(self):
        runs = self.api.runs(self.project, filters=UNLABELED, per_page=1000)
        return [r.id for r in runs if r.id not in self.labeled]

    def video(self, run_id):
        """Download the run's camera videos and put them side by side in one video, returns (cameras, local path)."""
        with self.video_locks.setdefault(run_id, threading.Lock()):
            if run_id not in self.videos:
                run = self.run(run_id)
                root = self.cache_dir / run_id
                cam_paths = {}
                for key, val in run.summary_metrics.items():
                    cam = key.removeprefix("video/")
                    if not key.startswith("video/") or cam in HIDDEN_VIDEOS or not isinstance(val, dict) or "path" not in val:
                        continue
                    if not (root / val["path"]).exists():
                        run.file(val["path"]).download(root=str(root), replace=True)
                    cam_paths[cam] = root / val["path"]
                cameras = sorted(cam_paths)
                path = root / "cameras.mp4"
                if not path.exists():
                    videos = [mediapy.read_video(cam_paths[cam]) for cam in cameras]
                    n = min(len(v) for v in videos)
                    tmp = root / "cameras.tmp.mp4"
                    mediapy.write_video(tmp, np.concatenate([v[:n] for v in videos], axis=2), fps=videos[0].metadata.fps)
                    tmp.rename(path)
                self.videos[run_id] = (cameras, path)
            return self.videos[run_id]

    def next(self):
        ids = self.unlabeled()
        if not ids:
            return None, 0
        run_id = self.prefetched if self.prefetched in ids else random.choice(ids)
        run = self.run(run_id)
        self.video(run_id)
        self.prefetch([i for i in ids if i != run_id])
        return run, len(ids)

    def prefetch(self, ids):
        """Download the next rollout's videos in the background so it loads right away."""
        self.prefetched = random.choice(ids) if ids else None
        if self.prefetched:
            run_id = self.prefetched
            threading.Thread(target=self.video, args=(run_id,), daemon=True).start()

    def label(self, run_id, success):
        run = self.run(run_id)
        run.summary.update({"success": success, "success_rate": int(success)})
        self.labeled.add(run_id)


def create_app(labeler):
    app = Flask(__name__)

    @app.get("/")
    def index():
        return PAGE

    @app.get("/api/next")
    def next_rollout():
        with labeler.lock:
            run, remaining = labeler.next()
        if run is None:
            return jsonify(done=True)
        # nothing that identifies the policy or world model, that's behind /api/reveal
        return jsonify(
            done=False,
            id=run.id,
            task=run.summary.get("task") or run.config.get("task"),
            remaining=remaining,
            cameras=labeler.video(run.id)[0],
            video=f"/video/{run.id}",
        )

    @app.get("/api/reveal/<run_id>")
    def reveal(run_id):
        run = labeler.run(run_id)
        policy = run.config.get("policy_metadata", {})
        ckpt_path = run.config.get("world_model", {}).get("ckpt_path") or ""
        return jsonify(
            id=run.id,
            policy=policy.get("model_name") or run.name,
            policy_path=policy.get("policy"),
            world_model=world_model_name(ckpt_path),
            world_model_ckpt=ckpt_path,
            url=run.url,
        )

    @app.post("/api/label/<run_id>")
    def label(run_id):
        success = request.get_json().get("success")
        if not isinstance(success, bool):
            abort(400, "success must be true or false")
        labeler.label(run_id, success)
        return jsonify(ok=True)

    @app.get("/video/<run_id>")
    def video(run_id):
        return send_file(labeler.video(run_id)[1], mimetype="video/mp4", conditional=True)

    return app


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Label world model rollouts</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 20px; background: #f6f6f6; color: #222; }
  .hidden { display: none !important; }
  #task { font-size: 1.5em; font-weight: 600; margin: 8px 0 16px; }
  #meta { color: #666; font-size: 0.9em; }
  #cameras { display: flex; font-size: 0.85em; color: #666; margin-bottom: 4px; }
  #cameras span { flex: 1; }
  video { width: 100%; background: #000; border-radius: 4px; }
  #controls, #labels { display: flex; gap: 12px; align-items: center; margin: 16px 0; flex-wrap: wrap; }
  button { font-size: 1em; padding: 8px 16px; border-radius: 6px; border: 1px solid #aaa; background: white; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
  #labels button { font-size: 1.3em; padding: 12px 32px; color: white; border: none; }
  #success { background: #2e7d32; }
  #failure { background: #c62828; }
  #reveal-info { font-family: monospace; }
  #status { color: #666; }
  #done { text-align: center; margin-top: 20vh; font-size: 2em; }
</style>
</head>
<body>
<div id="loading">Loading...</div>
<div id="done" class="hidden">All rollouts are labeled 🎉</div>
<div id="trial" class="hidden">
  <div id="meta"><span id="remaining"></span></div>
  <div id="task"></div>
  <div id="cameras"></div>
  <video id="video" controls muted autoplay loop playsinline></video>
  <div id="controls">
    <button id="restart">Restart video</button>
    <label>Speed
      <select id="speed">
        <option>0.5</option><option selected>1</option><option>2</option><option>4</option>
      </select>
    </label>
    <button id="reveal">Reveal policy and world model</button>
    <span id="reveal-info" class="hidden"></span>
  </div>
  <div id="labels">
    <button id="success">Success (s)</button>
    <button id="failure">Failure (f)</button>
    <span id="status"></span>
  </div>
</div>
<script>
let current = null;
let busy = false;
const $ = (id) => document.getElementById(id);

function show(id) {
  for (const s of ["loading", "done", "trial"]) $(s).classList.toggle("hidden", s !== id);
}

async function loadNext() {
  show("loading");
  const resp = await fetch("/api/next");
  if (!resp.ok) { $("loading").textContent = "Error loading rollout: " + await resp.text(); return; }
  const trial = await resp.json();
  if (trial.done) { current = null; show("done"); return; }
  current = trial;
  $("task").textContent = trial.task;
  $("remaining").textContent = trial.remaining + " unlabeled";
  $("reveal-info").classList.add("hidden");
  $("reveal-info").textContent = "";
  $("reveal").classList.remove("hidden");
  $("status").textContent = "";
  $("cameras").innerHTML = "";
  for (const cam of trial.cameras) {
    const span = document.createElement("span");
    span.textContent = cam;
    $("cameras").appendChild(span);
  }
  $("video").src = trial.video;
  show("trial");
}

async function label(success) {
  if (!current || busy) return;
  busy = true;
  for (const b of document.querySelectorAll("#labels button")) b.disabled = true;
  $("status").textContent = "Saving...";
  try {
    const resp = await fetch(`/api/label/${current.id}`, {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({success}),
    });
    if (!resp.ok) throw new Error(await resp.text());
    await loadNext();
  } catch (e) {
    $("status").textContent = "Error saving label: " + e.message;
  } finally {
    busy = false;
    for (const b of document.querySelectorAll("#labels button")) b.disabled = false;
  }
}

$("success").onclick = () => label(true);
$("failure").onclick = () => label(false);
$("restart").onclick = () => { $("video").currentTime = 0; $("video").play(); };
$("speed").onchange = () => { $("video").playbackRate = parseFloat($("speed").value); };
// the playback rate resets when a new video loads
$("video").onloadedmetadata = () => { $("video").playbackRate = parseFloat($("speed").value); };
$("reveal").onclick = async () => {
  const info = await (await fetch(`/api/reveal/${current.id}`)).json();
  $("reveal-info").innerHTML = "";
  const a = document.createElement("a");
  a.href = info.url; a.target = "_blank"; a.textContent = info.id;
  $("reveal-info").append(`policy: ${info.policy} (${info.policy_path}) | world model: ${info.world_model} (${info.world_model_ckpt}) | run: `, a);
  $("reveal-info").classList.remove("hidden");
  $("reveal").classList.add("hidden");
};
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT" || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "s") label(true);
  if (e.key === "f") label(false);
});
loadNext();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--project", type=str, default="adeshpande-princeton-university/synthetic-wm-evals-wm", help="wandb entity/project of the world model rollouts")
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.cache/ctrl_world/label_videos"))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(Labeler(args.project, args.cache_dir))
    app.run(host=args.host, port=args.port, threaded=True)
