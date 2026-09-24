"""
Policy-in-the-loop rollouts inside Ctrl-World of joint velocity policies served by the MolmoBot-Pi0 codebase, e.g. the
official openpi droid models pi0_droid, pi05_droid and pi0_fast_droid. Joint position policies (MolmoBot-Pi0 checkpoints)
aren't supported yet, see docs/joint_position_policies.md.

Initial conditions (a frame from every camera, the robot joint state and the instruction) are taken from
real robot evals logged to wandb by RoboRollout (scripts/droid/run_policy.py). The real rollout is shown
next to the world model rollout in the saved videos.

Runs are passed as a run path (entity/project/run_id, or its wandb url), a comma separated list of them, or a json
file with a list, where each entry is either a run path or a dict:
[
    "entity/project/run_id",
    {"run": "entity/project/run_id", "start_idx": 0, "instruction": "put the banana on the plate"}
]
start_idx (default 0) is the policy step of the real rollout to start from, and instruction defaults to the task of the run.

Each world model rollout is logged to wandb (and saved under save_dir) in the same format as the real rollouts, with
the real run it started from in its config (source_run).
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # openpi imports jax, don't let it grab the gpu
os.environ["JAX_ENABLE_X64"] = "0"  # 64-bit jax messes with openpi inference

import sys
import json
import pickle
import traceback
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import mediapy
import wandb
from accelerate import Accelerator
from decord import VideoReader, cpu
from filelock import FileLock
from huggingface_hub import hf_hub_download, snapshot_download
from scipy.spatial.transform import Rotation as R

import einops
from openpi.training import config as config_pi
from molmobot_pi0.dataset_openpi import MlSpacesDatasetConfigFactory
from molmobot_pi0.eval.policies.pi import PiJointVelPolicy, EXO_CAM_INPUT_KEY, WRIST_CAM_INPUT_KEY

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.utils import get_fk_solution


# RoboRollout logs the gripper as MolmoSpaces finger joint positions, 0 is open and this is fully closed
GRIPPER_QPOS_RANGE = 0.824033
# droid joint velocity is the joint delta from the current position divided by this
DROID_MAX_JOINT_DELTA = 0.2


def normalize_run_path(run):
    # https://wandb.ai/<entity>/<project>/runs/<run_id>[/overview][?...] -> <entity>/<project>/<run_id>
    run = run.strip().split("?")[0].removeprefix("https://").removeprefix("wandb.ai/").strip("/")
    parts = run.split("/")
    if len(parts) >= 4 and parts[2] == "runs":
        parts = [parts[0], parts[1], parts[3]]
    if len(parts) != 3 or "runs" in parts:
        raise ValueError(f"Expected a wandb run path entity/project/run_id or its url, got {run}")
    return "/".join(parts)


def load_run_specs(runs):
    # a json file, or a comma separated list of run paths
    if os.path.isfile(runs):
        with open(runs) as f:
            specs = json.load(f)
    else:
        specs = [r for r in runs.split(",") if r.strip()]
    specs = [{"run": s} if isinstance(s, str) else dict(s) for s in specs]
    for s in specs:
        s["run"] = normalize_run_path(s["run"])
        s.setdefault("start_idx", 0)
        s.setdefault("instruction", None)
    return specs


class WandbEpisode:
    """A real robot rollout logged to wandb by RoboRollout, downloaded to (and cached in) cache_dir."""

    def __init__(self, run_path, cache_dir, wm_cameras=None, policy_exo_camera=None):
        run = wandb.Api().run(run_path)
        self.run_path = "/".join(run.path)
        self.url = run.url
        self.name = run.name
        self.entity, self.project = run.entity, run.project
        self.tags = list(run.tags)
        self.config = run.config
        self.real_success = run.summary.get("success")
        run_dir = Path(cache_dir) / run.entity / run.project / run.id
        run_dir.mkdir(parents=True, exist_ok=True)

        # videos logged without the _rt suffix have one frame per policy step, aligned with the observations
        video_paths = {key.removeprefix("video/"): val["path"] for key, val in run.summary.items() if key.startswith("video/") and not key.endswith("_rt")}
        with FileLock(run_dir / ".lock"):  # jobs sharing the cache can download the same run concurrently
            for path in ["observations.npz", *video_paths.values()]:
                run.file(path).download(root=run_dir, exist_ok=True)
        self.videos = {cam: run_dir / path for cam, path in video_paths.items()}

        obs = np.load(run_dir / "observations.npz")
        self.joints = obs["qpos/arm"]  # (T, 7)
        self.gripper = np.clip(obs["qpos/gripper"][:, 0] / GRIPPER_QPOS_RANGE, 0, 1)  # (T,) 0 is open, 1 is closed as in droid

        # policy_cameras maps policy inputs to robot cameras, e.g. {exo_camera_1: exo_camera_2, wrist_camera: wrist_camera}
        policy_cameras = self.config.get("policy_cameras")
        policy_cameras = policy_cameras if isinstance(policy_cameras, dict) else {}
        self.policy_cameras = policy_cameras
        self.policy_exo_camera = policy_exo_camera or policy_cameras.get("exo_camera_1")
        self.wrist_camera = policy_cameras.get("wrist_camera", "wrist_camera")
        if self.policy_exo_camera is None:
            raise ValueError(f"Run {self.run_path} has no policy_cameras/exo_camera_1 in its config, pass --policy_exo_camera")

        # ctrl-world views are [exterior 1, exterior 2, wrist], put the policy camera in the same slot as rollout_interact_pi.py
        if wm_cameras is None:
            other_exo = sorted(c for c in self.videos if c not in (self.policy_exo_camera, self.wrist_camera))
            assert len(other_exo) == 1, f"Expected 2 exo cameras in {self.run_path}, got {sorted(self.videos)}, pass --wm_cameras"
            wm_cameras = [other_exo[0], self.policy_exo_camera, self.wrist_camera]
        self.wm_cameras = list(wm_cameras)
        for cam in self.wm_cameras:
            assert cam in self.videos, f"Camera {cam} not logged in {self.run_path}, available: {sorted(self.videos)}"
        assert self.policy_exo_camera in self.wm_cameras and self.wrist_camera in self.wm_cameras

        self.length = min(len(self.joints), *(len(VideoReader(str(self.videos[c]), ctx=cpu(0))) for c in self.wm_cameras))
        self.instruction = self.config.get("task")

    def load_frames(self, cam, frame_ids, height, width):
        vr = VideoReader(str(self.videos[cam]), ctx=cpu(0), num_threads=2)
        frame_ids = np.minimum(frame_ids, self.length - 1)
        frames = vr.get_batch(list(frame_ids))
        frames = frames if isinstance(frames, torch.Tensor) else torch.from_numpy(frames.asnumpy())  # molmobot_pi0 sets the decord torch bridge
        frames = frames.permute(0, 3, 1, 2).float()
        frames = F.interpolate(frames, size=(height, width), mode='bilinear', align_corners=False, antialias=True)
        return frames.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()  # (F, H, W, 3)


def load_policy(policy, device_id, compile_mode):
    """
    policy is the name of an openpi config (pi0_droid, pi05_droid, pi0_fast_droid) to load openpi's released
    checkpoint for it, or a local checkpoint dir or hf://<repo_id> of a joint velocity policy trained with MolmoBot-Pi0.
    """
    if policy.startswith("hf://"):
        ckpt_dir = snapshot_download(policy.removeprefix("hf://"))
    elif os.path.isdir(policy):
        ckpt_dir = policy
    else:
        ckpt_dir = None

    if ckpt_dir is None:
        train_config = config_pi.get_config(policy)
        use_torch = False  # the released droid checkpoints are jax
    else:
        with open(Path(ckpt_dir) / "assets" / "train_config.pkl", "rb") as f:
            train_config = pickle.load(f)
        use_torch = (Path(ckpt_dir) / "model.safetensors").exists()

    # MolmoBot-Pi0 predicts joint positions, the official droid models (and pi05_mlspaces_finetune) joint velocities
    if isinstance(train_config.data, MlSpacesDatasetConfigFactory) and train_config.data.joint_pos_actions:
        raise NotImplementedError(f"{policy} predicts joint positions, which aren't supported yet, see docs/joint_position_policies.md")
    print(f"loading {policy} with {'pytorch' if use_torch else 'jax'}")
    # only the policy's model loading and input processing are used, actions go through the action adapter in agent.forward_policy
    policy = PiJointVelPolicy(
        model_name=policy if ckpt_dir is None else None,
        checkpoint_dir=ckpt_dir,
        use_torch=use_torch,
        cameras={"exo_camera_1": EXO_CAM_INPUT_KEY, "wrist_camera": WRIST_CAM_INPUT_KEY},
        device_id=device_id,
        compile_mode=compile_mode,
    )
    policy.prepare_model()
    return policy


def resolve_ckpt(ckpt_path):
    # a local file, or hf://<repo_id>/<filename>
    if ckpt_path.startswith("hf://"):
        repo_id, filename = ckpt_path.removeprefix("hf://").rsplit("/", 1)
        return hf_hub_download(repo_id, filename)
    return ckpt_path


def droid_scale_vel(joint_vel):
    # droid scales joint velocities down to at most 1 per joint, see joint_velocity_to_delta in droid/robot_ik/robot_ik_solver.py
    return joint_vel / np.maximum(np.max(np.abs(joint_vel), axis=-1, keepdims=True), 1.0)


def unsquash_frame(image):
    # world model frames are real 16:9 frames squashed to 192x320, undo that before the policy's own resizing
    image = torch.from_numpy(image).permute(2, 0, 1)[None].float()
    image = F.interpolate(image, size=(180, 320), mode='bilinear', align_corners=False)
    return image[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()


def joints_to_eef(joint_pos, gripper_pos):
    state_fk = []
    for joints, gripper in zip(joint_pos, gripper_pos):
        current_state_fk = get_fk_solution(joints)
        xyz = current_state_fk[:3, 3]
        euler = R.from_matrix(current_state_fk[:3, :3]).as_euler('xyz')
        state_fk.append(np.concatenate([xyz, euler, [gripper]], axis=0))
    return np.array(state_fk)  # (N, 7)


class agent():
    def __init__(self, args):
        args.val_model_path = resolve_ckpt(args.ckpt_path)
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        self.policy = load_policy(args.policy, self.device.index or 0, args.policy_compile_mode)
        print("load policy success")

        # same action adapter as rollout_interact_pi.py, it predicts the joint positions a real droid robot reaches following a joint velocity chunk
        from models.action_adapter.train2 import Dynamics
        action_adapter = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.action_adapter)
        self.dynamics_model = Dynamics(action_dim=7, action_num=15, hidden_size=512).to(self.device)
        self.dynamics_model.device = self.device
        self.dynamics_model.load_state_dict(torch.load(action_adapter, map_location=self.device))

        self.model = CrtlWorld(args)
        self.model.load_state_dict(torch.load(args.val_model_path))
        self.model.to(self.device).to(self.dtype)
        self.model.eval()
        print("load world model success")
        with open(f"{args.data_stat_path}", 'r') as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat['state_01'])[None,:]
            self.state_p99 = np.array(data_stat['state_99'])[None,:]

    def normalize_bound(self, data, data_min, data_max, clip_min=-1, clip_max=1, eps=1e-8):
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def encode_frames(self, frames):
        x = torch.from_numpy(frames).to(self.device).to(self.dtype).permute(0,3,1,2) / 255.0*2-1
        vae = self.model.pipeline.vae
        with torch.no_grad():
            batch_size = 32
            latents = []
            for i in range(0, len(x), batch_size):
                latents.append(vae.encode(x[i:i+batch_size]).latent_dist.sample().mul_(vae.config.scaling_factor))
        return torch.cat(latents, dim=0)  # (F, 4, 24, 40)

    def forward_wm(self, action_cond, video_latent_cond, his_cond=None, text=None):
        # same as rollout_interact_pi.py, minus decoding the ground truth
        args = self.args
        image_cond = video_latent_cond

        # action should be normed
        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (args.num_frames+args.num_history, args.action_dim)

        # predict future frames
        with torch.no_grad():
            if text is not None:
                text_token = self.model.action_encoder(action_cond, text, self.model.tokenizer, self.model.text_encoder)
            else:
                text_token = self.model.action_encoder(action_cond)
            pipeline = self.model.pipeline

            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
                text=text_token,
                width=args.width,
                height=int(args.height*3),
                num_frames=args.num_frames,
                history=his_cond,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
            )
            latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1) # (3, 5, 4, 24, 40)

            # decode predicted video
            decoded_video = []
            bsz,frame_num = latents.shape[:2]
            x = latents.flatten(0,1)
            decode_kwargs = {}
            for i in range(0,x.shape[0],args.decode_chunk_size):
                chunk = x[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
                decode_kwargs["num_frames"] = chunk.shape[0]
                decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video,dim=0)
        videos = videos.reshape(bsz,frame_num,*videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)

        return videos, latents  # np.uint8 (3, 5, 192, 320, 3)

    def forward_policy(self, exo_image, wrist_image, joints, gripper, text):
        args = self.args
        n_steps = int(args.policy_skip_step*(args.pred_step-1))
        obs = {
            "task": text,
            "qpos": {"arm": joints, "gripper": np.full(2, gripper * GRIPPER_QPOS_RANGE)},
            "exo_camera_1": unsquash_frame(exo_image),
            "wrist_camera": unsquash_frame(wrist_image),
        }

        assert n_steps < 15, f"The action adapter covers 14 policy steps, got {n_steps} per interaction"

        self.policy.reset()
        model_input = self.policy.obs_to_model_input(obs)
        actions = self.policy.model.infer(model_input)["actions"]  # (action_horizon, 8), gripper position in [0, 1]
        # the adapter takes 15 steps, repeat the last action of shorter chunks (pi0_droid and pi0_fast_droid predict 10)
        actions = np.concatenate([actions, np.repeat(actions[-1:], max(15 - len(actions), 0), axis=0)])[:15]
        gripper_pos = np.clip(actions[:, 7], 0, args.gripper_max)
        joint_vel = droid_scale_vel(actions[:, :7])
        joint_future = self.dynamics_model(joints[None], joint_vel, None, training=False)  # (15, 7)
        joint_pos = np.concatenate([joints[None], joint_future], axis=0)[:15]  # (15, 7) index k is k policy steps from now
        gripper_pos = np.concatenate([[gripper], gripper_pos], axis=0)[:15]  # (15,)
        state_fk = joints_to_eef(joint_pos, gripper_pos)

        # subsample the policy rate trajectory to the world model rate
        idx = np.arange(args.pred_step) * args.policy_skip_step
        policy_in_out = {
            'joint_pos': joint_pos[:n_steps],  # (12, 7)
            'joint_vel': joint_vel[:n_steps],  # (12, 7)
            'gripper_pos': gripper_pos[:n_steps],  # (12,)
            'gripper_cmd': gripper_pos[1:n_steps+1],  # (12,) the gripper command of each step
            'state_fk': state_fk[:n_steps],  # (12, 7)
        }
        return policy_in_out, joint_pos[idx], gripper_pos[idx], state_fk[idx]


def rollout(Agent, args, spec):
    interact_num = args.interact_num
    pred_step = args.pred_step
    skip = args.policy_skip_step

    episode = WandbEpisode(spec["run"], args.wandb_cache_dir, wm_cameras=args.wm_cameras, policy_exo_camera=args.policy_exo_camera)
    text_i = spec["instruction"] or episode.instruction
    start_idx_i = spec["start_idx"]
    assert 0 <= start_idx_i < episode.length, f"start_idx {start_idx_i} out of range for {episode.run_path} with {episode.length} steps"
    exo_view = episode.wm_cameras.index(episode.policy_exo_camera)
    wrist_view = episode.wm_cameras.index(episode.wrist_camera)
    print(f"run: {episode.run_path}, task: {text_i}, world model views: {episode.wm_cameras}, policy exo camera: {episode.policy_exo_camera}")

    # ground truth frames from the real rollout at the world model rate, repeating the last frame once it ends
    frame_ids = start_idx_i + np.arange(interact_num*(pred_step-1)+1) * skip
    if frame_ids[-1] >= episode.length:
        print(f"Real rollout ends after {episode.length} steps, ground truth is padded with its last frame")
    video_dict = [episode.load_frames(cam, frame_ids, args.height, args.width) for cam in episode.wm_cameras]

    joints0 = episode.joints[start_idx_i].astype(np.float64)
    gripper0 = float(episode.gripper[start_idx_i])
    eef0 = joints_to_eef(joints0[None], [gripper0])
    print("eef pose at t=0", eef0[0], "joint at t=0", joints0, "gripper at t=0", gripper0)

    # initialize all history buffer
    video_to_save, info_to_save, wm_frames = [], [], []
    his_cond, his_joint, his_gripper, his_eef = [], [], [], []
    first_latent = torch.cat([Agent.encode_frames(v[0:1]) for v in video_dict], dim=2)  # (1, 4, 72, 40)
    assert first_latent.shape == (1, 4, 72, 40), f"Expected first_latent shape (1, 4, 72, 40), got {first_latent.shape}"
    for i in range(args.num_history*4):
        his_cond.append(first_latent)  # (1, 4, 72, 40)
        his_joint.append(joints0)  # (7,)
        his_gripper.append(gripper0)
        his_eef.append(eef0)  # (1, 7)
    video_dict_pred = [v[0:1] for v in video_dict]

    # start rollout
    for i in range(interact_num):
        start_id = int(i*(pred_step-1))
        end_id = start_id + pred_step

        print("################ policy forward ####################")
        current_obs = [v[-1] for v in video_dict_pred]
        policy_in_out, joint_pos, gripper_pos, cartesian_pose = Agent.forward_policy(
            current_obs[exo_view], current_obs[wrist_view], his_joint[-1], his_gripper[-1], text=text_i)
        print("cartesian space action", cartesian_pose[0]) # output xyz and gripper for debug
        print("cartesian space action", cartesian_pose[-1]) # output xyz and gripper for debug

        print("################ world model forward ################")
        print(f'task: {text_i}, run: {episode.run_path}, interact step: {i}/{interact_num}')
        history_idx = args.history_idx
        action_cond = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)
        action_cond = np.concatenate([action_cond, cartesian_pose], axis=0) # (num_history+num_frames, 7)
        his_latent = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
        current_latent = his_cond[-1]  # (1, 4, 72, 40)
        video_dict_pred, predict_latents = Agent.forward_wm(action_cond, current_latent, his_cond=his_latent, text=text_i if args.text_cond else None)

        # real rollout on the left, world model on the right, views stacked vertically
        true_videos = np.stack([v[start_id:end_id] for v in video_dict])  # (3, 5, 192, 320, 3)
        videos_cat = np.concatenate([np.concatenate(list(true_videos), axis=-3), np.concatenate(list(video_dict_pred), axis=-3)], axis=-2)  # (5, 576, 640, 3)

        print("################ record information ################")
        his_joint.append(joint_pos[pred_step-1])
        his_gripper.append(gripper_pos[pred_step-1])
        his_eef.append(cartesian_pose[pred_step-1][None,:]) # (1, 7)
        his_cond.append(torch.cat([v[pred_step-1] for v in predict_latents], dim=1).unsqueeze(0))  # (1, 4, 72, 40)
        video_to_save.append(videos_cat[:pred_step-1])
        wm_frames.append(video_dict_pred[:, :pred_step-1])
        info_to_save.append(policy_in_out)
    wm_frames.append(video_dict_pred[:, pred_step-1:pred_step])  # the final frame, so there's one per observation like the real videos

    # stack the per interaction outputs, at the policy rate
    traj = {key: np.concatenate([info[key] for info in info_to_save], axis=0) for key in info_to_save[0].keys()}
    traj['final_joint_pos'], traj['final_gripper_pos'] = his_joint[-1], his_gripper[-1]
    save_rollout(args, episode, text_i, start_idx_i, traj, np.concatenate(wm_frames, axis=1), np.concatenate(video_to_save, axis=0))


def save_rollout(args, episode, text, start_idx, traj, wm_frames, comparison_video):
    """
    Save a world model rollout in the same format as the real rollouts logged by RoboRollout (scripts/droid/run_policy.py),
    locally and to wandb. Joint positions are the action adapter's predictions, videos are at the world model rate (5hz)
    and resolution (192x320), and there's no success label, wall clock episode length, _rt videos or server timing.
    """
    policy_name = Path(args.policy.rstrip('/')).name
    source_id = episode.run_path.split('/')[-1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.save_dir) / args.task_name / f"{policy_name}_{source_id}_{start_idx}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_dt = episode.config.get("policy_dt", 0.066)

    # observations (T+1) and actions (T) at the policy rate, like the real rollouts
    joint_pos = np.concatenate([traj['joint_pos'], traj['final_joint_pos'][None]], axis=0)  # (T+1, 7)
    gripper_pos = np.append(traj['gripper_pos'], traj['final_gripper_pos'])  # (T+1,) 0 is open, 1 is closed
    ee_fk = [get_fk_solution(q) for q in joint_pos]
    observations = {
        "qpos/arm": joint_pos.astype(np.float32),
        "qpos/gripper": np.repeat(gripper_pos[:, None] * GRIPPER_QPOS_RANGE, 2, axis=1),
        "ee_pose/pos": np.array([T[:3, 3] for T in ee_fk], dtype=np.float32),
        "ee_pose/quat": np.array([R.from_matrix(T[:3, :3]).as_quat() for T in ee_fk], dtype=np.float32),  # xyzw, as logged by polymetis
    }
    actions = {
        "arm": (traj['joint_pos'] + traj['joint_vel'] * DROID_MAX_JOINT_DELTA).astype(np.float32),  # the joint target droid commands
        "gripper": (traj['gripper_cmd'][:, None] * 255.0).astype(np.float32),
        "arm_vel": traj['joint_vel'].astype(np.float32),
    }
    np.savez_compressed(out_dir / "observations.npz", **observations)
    np.savez_compressed(out_dir / "actions.npz", **actions)

    # one video per camera like the real rollouts, and the real rollout next to the world model
    for cam, frames in zip(episode.wm_cameras, wm_frames):
        mediapy.write_video(out_dir / f"{cam}.mp4", frames, fps=5)
    mediapy.write_video(out_dir / "real_vs_wm.mp4", comparison_video, fps=5)

    n_steps = len(actions["arm"])
    info = {
        "episode_length_nominal": n_steps * policy_dt,
        "episode_length_steps": n_steps,
        "task": text,
    }
    source_run = {
        "path": episode.run_path,
        "url": episode.url,
        "name": episode.name,
        "start_idx": start_idx,
        "success": episode.real_success,
    }
    config = {
        "task": text,
        "robot": {"cameras": episode.config.get("robot", {}).get("cameras")},
        "policy_dt": policy_dt,
        "policy_cameras": {**episode.policy_cameras, "exo_camera_1": episode.policy_exo_camera},
        "policy_metadata": {"model_name": policy_name, "policy": args.policy},
        "source_run": source_run,
        "world_model": {
            "ckpt_path": args.ckpt_path,
            "data_stat_path": args.data_stat_path,
            "svd_model_path": args.svd_model_path,
            "action_adapter": args.action_adapter,
            "wm_cameras": episode.wm_cameras,
            "interact_num": args.interact_num,
            "pred_step": args.pred_step,
            "policy_skip_step": args.policy_skip_step,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "fps": 5,
        },
    }
    (out_dir / "info.json").write_text(json.dumps({**info, "config": config}, indent=2, default=str))
    print(f"Saved rollout to {out_dir}")

    if args.no_wandb:
        return
    with wandb.init(
        entity=args.wandb_entity or episode.entity,
        project=args.wandb_project or f"{episode.project}-wm",
        name=f"wm_{policy_name}",
        dir=str(out_dir),
        tags=["world_model", *episode.tags],
        config=config,
        notes=text,
    ) as run:
        videos = {f"video/{cam}": wandb.Video(str(out_dir / f"{cam}.mp4"), caption=cam, format="mp4") for cam in episode.wm_cameras}
        videos["video/real_vs_wm"] = wandb.Video(str(out_dir / "real_vs_wm.mp4"), caption="real (left) vs world model (right)", format="mp4")
        run.summary.update({**info, **videos})
        for name in ["observations.npz", "actions.npz", "info.json"]:
            run.save(str(out_dir / name), base_path=str(out_dir), policy="now")
        print(f"Logged rollout to {run.url}")

if __name__ == "__main__":
    from config import wm_args
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--runs', type=str, required=True, help='wandb runs to take initial conditions from: a run path or url, a comma separated list of them, or a json file')
    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default='hf://yjguo/Ctrl-World/checkpoint-10000.pt', help='ctrl-world checkpoint, a local .pt or hf://<repo_id>/<filename>, defaults to the official one')
    parser.add_argument('--data_stat_path', type=str, default=None, help='state normalization stats the checkpoint was trained with, defaults to droid\'s')
    parser.add_argument('--policy', type=str, default='pi05_droid', help='pi0_droid, pi05_droid, pi0_fast_droid, or a joint velocity checkpoint as a local dir or hf://<repo_id>')
    parser.add_argument('--policy_compile_mode', type=str, default=None, help='torch.compile mode for the policy, not compiled by default')
    parser.add_argument('--policy_exo_camera', type=str, default=None, help='override policy_cameras/exo_camera_1 from the run config')
    parser.add_argument('--wm_cameras', type=str, nargs=3, default=None, help='robot cameras for the 3 world model views, [exo, exo, wrist]')
    parser.add_argument('--gripper_max', type=float, default=None)
    parser.add_argument('--interact_num', type=int, default=None)
    parser.add_argument('--policy_skip_step', type=int, default=None)
    parser.add_argument('--wandb_cache_dir', type=str, default=os.path.expanduser('~/.cache/ctrl_world/wandb_runs'))
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None, help='entity to log rollouts to, defaults to the source run\'s')
    parser.add_argument('--wandb_project', type=str, default=None, help='project to log rollouts to, defaults to <source run project>-wm')
    parser.add_argument('--no_wandb', action='store_true', help='only save rollouts locally')
    args_new = parser.parse_args()

    args = wm_args(task_type='molmobot_pi0')

    def merge_args(cfg, cli_args):
        for k, v in vars(cli_args).items():
            if v is not None or not hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    args = merge_args(args, args_new)

    # create agent
    Agent = agent(args)

    failed = []
    for spec in load_run_specs(args.runs):
        try:
            rollout(Agent, args, spec)
        except Exception:
            traceback.print_exc()
            failed.append(spec["run"])
    if failed:
        raise RuntimeError(f"Rollouts failed for runs: {failed}")

# CUDA_VISIBLE_DEVICES=0 python scripts/rollout_interact_molmobot_pi0.py --runs runs.json --svd_model_path ${svd} --clip_model_path ${clip} --policy pi05_droid
