#!/usr/bin/env python3
import os
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import cv2  # pip install opencv-python
import gymnasium as gym
import robo_gym  # ensure your fork is installed
from transformers import AutoModelForCausalLM
from PIL import Image

# ------- CrossFormer deps -------
import jax
import tensorflow as tf
from crossformer.model.crossformer_model import CrossFormerModel  # repo API

# ============ Config ============
RS_ADDRESS = "127.0.0.1:50051"   # real robot server address
ROBOT_MODEL = "locobot_wx250s"
ENV_ID = "EmptyEnvironmentInterbotixRRob-v0"  # Rob (real) or Sim
USE_GUI = True  # 1/0
WITH_CAMERA = True

# CrossFormer config (override these with your finetune specifics)
CROSSFORMER_CKPT = "hf://rail-berkeley/crossformer"
CROSSFORMER_DATASET = "aloha_pen_uncap_diverse_dataset"  # <-- set to YOUR finetune dataset name
HEAD_NAV = "locobot_nav"     # expected to output 2D (vx, wz)
HEAD_ARM = "locobot_arm"     # expected to output 6D arm joints
IMG_KEY = "image_primary"    # match what you used in finetune
IMG_SIZE = (224, 224)

# Visual target overlay
DOT_RADIUS_PIX = 6  # Y in your spec
DIST_THRESH_M = 0.35  # Z in meters

# Control timing
CTRL_HZ = 10.0
SLEEP_SEC = 1.0 / CTRL_HZ

# Arm defaults (hold current joints during nav)
DEFAULT_ARM_HOLD = np.array([0.0757, 0.0074, 0.0122, -0.00011, 0.0058, -0.00076], dtype=np.float32)

# ============ Stubs / Adapters for your models ============
class MoondreamClient:
    """Moonbeam/Moondream vision client using vikhyatk/moondream2."""
    def __init__(self, model_name: str = "vikhyatk/moondream2"):
        # Load the model once, on CUDA if available
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision='2025-01-09',
            trust_remote_code=True,
            device_map={'': 'cuda'}
        )

    def locate_object(self, rgb_img: np.ndarray, text_query: str) -> Optional[Tuple[int, int]]:
        """
        Return (u, v) pixel coordinates of the target in the given RGB image.
        If not found, return None.
        """
        # Convert np.ndarray (BGR or RGB) to PIL.Image in RGB
        if rgb_img.shape[2] == 3:
            # Assume input is BGR (OpenCV), convert to RGB
            img_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = rgb_img
        pil_img = Image.fromarray(img_rgb)

        # Query the model
        try:
            points = self.model.point(pil_img, text_query)['points']
        except Exception as e:
            print(f"[WARN] Moondream model failed: {e}")
            return None

        if not points:
            return None

        # Use the first point (x, y in [0,1] relative coords)
        point = points[0]
        h, w = img_rgb.shape[:2]
        x = point['x'] * w
        y = point['y'] * h
        return (int(x), int(y))


@dataclass
class CrossFormerOutputs:
    """Container for CrossFormer outputs."""
    arm_joints_6: np.ndarray  # shape (6,)
    base_vx: float
    base_wz: float


# ---------------- CrossFormer: real implementation ----------------
def _tf_resize_uint8(img_uint8_hwc: np.ndarray, size=(224, 224)) -> np.ndarray:
    """Replicates CrossFormer server's Lanczos3 + rounding to uint8."""
    out = tf.image.resize(img_uint8_hwc, size=size, method="lanczos3", antialias=True)
    out = tf.cast(tf.clip_by_value(tf.round(out), 0, 255), tf.uint8).numpy()
    return out

class CrossFormerPolicy:
    """
    Wraps CrossFormer with two heads:
      - 'base' head: navigation (should output [vx, wz])
      - 'arm'  head: 6-DoF joints for the Interbotix arm (waist..wrist_rotate)
    Uses text tasks (language) like the official server.
    """
    def __init__(self, ckpt_path: str, dataset_name: str, head_nav: str, head_arm: str,
                 img_key: str = "image_primary", img_size=(224, 224)):
        # load pretrained/finetuned model
        # (same API as scripts/server.py)
        self.model = CrossFormerModel.load_pretrained(ckpt_path, step=None)
        self.dataset_name = dataset_name
        self.head_nav = head_nav
        self.head_arm = head_arm
        self.img_key = img_key
        self.img_size = img_size
        self._last_text = None
        self._task = None
        self._rng = jax.random.PRNGKey(0)

        # cache action un-normalization stats (dataset-specific)
        self._action_stats = self.model.dataset_statistics[self.dataset_name]["action"]

    # ----- task handling (language goals) -----
    def _ensure_task(self, instruction_text: str):
        if self._task is None or instruction_text != self._last_text:
            # Same API as server: create_tasks(texts=[...])
            self._task = self.model.create_tasks(texts=[instruction_text])
            self._last_text = instruction_text

    # ----- observation packing for CrossFormer -----
    def _pack_obs(self, image_bgr_uint8: np.ndarray) -> Dict[str, Any]:
        """
        Build a single-timestep obs dict expected by CrossFormer:
          - resized RGB image under self.img_key
          - timestep_pad_mask
          - add batch dim (B=1) and time dim (T=len(history)), here T=1
        """
        # CrossFormer expects RGB uint8; convert BGR→RGB, then resize like server.py
        rgb = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2RGB)
        rgb_resized = _tf_resize_uint8(rgb, size=self.img_size)

        # (T=1, H, W, 3)
        obs_t = {
            self.img_key: rgb_resized,  # HxWx3 uint8
        }
        # stack/window handling: we run single-frame (history T=1) with a pad mask of ones
        timestep_pad_mask = np.ones((1,), dtype=np.float32)

        # Add time dimension (T=1)
        obs_t = {k: v[None, ...] for k, v in obs_t.items()}
        obs_t["timestep_pad_mask"] = timestep_pad_mask[None, ...]

        # Add batch dimension (B=1)
        obs_bt = {k: v[None, ...] for k, v in obs_t.items()}
        return obs_bt

    # ----- heads -----
    def set_head(self, head: str):
        assert head in {"base", "arm"}
        self._active_head = head

    def act_nav(self, image_bgr: np.ndarray, instruction: str) -> Tuple[float, float]:
        self._ensure_task(instruction)
        obs = self._pack_obs(image_bgr)
        # same signature the server uses: sample_actions(obs, task, action_stats, head_name=..., rng=key)
        self._rng, key = jax.random.split(self._rng)
        acts = self.model.sample_actions(
            obs, self._task, self._action_stats, head_name=self.head_nav, rng=key
        )
        # acts: (B, T, D). We used B=1, T=1 → take [0,0]
        a0 = np.array(acts)[0, 0]
        if a0.shape[0] < 2:
            raise ValueError(
                f"Navigation head '{self.head_nav}' returned dim {a0.shape[0]}, "
                f"expected at least 2 for (vx,wz)."
            )
        vx, wz = float(a0[0]), float(a0[1])  # assume 2-D nav head
        return vx, wz

    def act_arm(self, image_bgr: np.ndarray, instruction: str) -> np.ndarray:
        self._ensure_task(instruction)
        obs = self._pack_obs(image_bgr)
        self._rng, key = jax.random.split(self._rng)
        acts = self.model.sample_actions(
            obs, self._task, self._action_stats, head_name=self.head_arm, rng=key
        )
        a0 = np.array(acts)[0, 0]
        if a0.shape[0] < 6:
            raise ValueError(
                f"Arm head '{self.head_arm}' returned dim {a0.shape[0]}, expected ≥6."
            )
        return a0[:6].astype(np.float32)

    def act(self, image_bgr: np.ndarray, instruction: str) -> CrossFormerOutputs:
        if getattr(self, "_active_head", "base") == "base":
            vx, wz = self.act_nav(image_bgr, instruction)
            return CrossFormerOutputs(arm_joints_6=DEFAULT_ARM_HOLD.copy(), base_vx=vx, base_wz=wz)
        else:
            arm = self.act_arm(image_bgr, instruction)
            return CrossFormerOutputs(arm_joints_6=arm, base_vx=0.0, base_wz=0.0)
# -------------------------------------------------------------------

# ============ Helpers ============
def to_rgb_from_obs_camera(cam_arr: np.ndarray) -> np.ndarray:
    """
    Your robo-gym Locobot camera obs is (H, W, 3) RGB uint8.
    We convert to BGR for OpenCV drawing.
    """
    assert cam_arr.ndim == 3 and cam_arr.shape[2] == 3, "Expected (H,W,3) camera obs"
    bgr = cv2.cvtColor(cam_arr, cv2.COLOR_RGB2BGR)
    return bgr

def draw_red_dot(rgb_or_bgr_img: np.ndarray, uv: Tuple[int, int], radius: int) -> np.ndarray:
    out = rgb_or_bgr_img.copy()
    cv2.circle(out, center=uv, radius=radius, color=(0, 0, 255), thickness=-1)  # OpenCV uses BGR
    return out

def depth_at_pixel(depth_map: np.ndarray, uv: Tuple[int, int]) -> Optional[float]:
    if depth_map is None:
        return None
    u, v = int(uv[0]), int(uv[1])
    if v < 0 or v >= depth_map.shape[0] or u < 0 or u >= depth_map.shape[1]:
        return None
    d = float(depth_map[v, u])
    if np.isnan(d) or d <= 0:
        return None
    return d

def make_action_vector(arm6: np.ndarray, vx: float, wz: float) -> np.ndarray:
    """
    robo-gym Locobot action layout:
      [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, base_linear_x, base_angular_z]
    """
    a = np.zeros(8, dtype=np.float32)
    a[:6] = arm6
    a[6] = vx
    a[7] = wz
    return a

# ============ Main routine ============
def main(object_query: str = "the target object", basket_query: str = "the basket"):
    # disable TF GPU for the small resize op (optional, mirrors server.py)
    tf.config.set_visible_devices([], "GPU")

    # 1) Instantiate environment (real robot server by default)
    env = gym.make(
        ENV_ID,
        rs_address=RS_ADDRESS,
        gui=USE_GUI,
        robot_model=ROBOT_MODEL,
        with_camera=WITH_CAMERA
    )
    obs, _ = env.reset(seed=int(time.time()))

    # 2) Load Moondream/Moonbeam
    finder = MoondreamClient()

    # 3) Load CrossFormer (real)
    policy = CrossFormerPolicy(
        ckpt_path=CROSSFORMER_CKPT,
        dataset_name=CROSSFORMER_DATASET,
        head_nav=HEAD_NAV,
        head_arm=HEAD_ARM,
        img_key=IMG_KEY,
        img_size=IMG_SIZE,
    )

    # Get initial camera
    assert isinstance(obs, dict) and "camera" in obs, "Environment must return dict with 'camera'"
    bgr, depth = to_rgb_from_obs_camera(np.asarray(obs["camera"]))

    # 4) Ask Moondream to locate the object
    uv = finder.locate_object(bgr, object_query)
    # If object not found, rotate in place until found
    while uv is None:
        print("[WARN] Moondream did not find the object; rotating to search...")
        # Rotate left in place (wz > 0, vx = 0)
        policy.set_head("base")
        search_action = make_action_vector(DEFAULT_ARM_HOLD, 0.0, 0.5)
        obs, reward, terminated, truncated, info = env.step(search_action)
        bgr, depth = to_rgb_from_obs_camera(np.asarray(obs["camera"]))
        uv = finder.locate_object(bgr, object_query)
        time.sleep(SLEEP_SEC)
        if terminated or truncated:
            print("[WARN] Episode ended during object search.")
            return

    # 5) Overlay a red dot on the image
    bgr_dot = draw_red_dot(bgr, uv, DOT_RADIUS_PIX)

    print("[INFO] Starting navigate-then-pick-then-place routine")
    phase = "navigate"
    t_last = time.time()
    basket_uv = None
    basket_bgr_dot = None

    while True:
        # 6) Distance to the target pixel
        dist_m = depth_at_pixel(depth, uv) if depth is not None else None
        if dist_m is not None:
            print(f"[DBG] Distance to target @ {uv} = {dist_m:.3f} m")

        if phase == "navigate":
            # 7) Navigate until we’re within threshold
            if (dist_m is None) or (dist_m > DIST_THRESH_M):
                policy.set_head("base")
                nav_instruction = "Go to the red dot."
                out = policy.act(bgr_dot, nav_instruction)  # returns (vx,wz) + holding arm
                action = make_action_vector(out.arm_joints_6, out.base_vx, out.base_wz)
            else:
                print("[INFO] Reached approach distance. Switching to PICK phase.")
                phase = "pick"
                t_last = time.time()
                continue

        elif phase == "pick":
            policy.set_head("arm")
            pick_instruction = "Align end effector with the red dot and grasp the object."
            out = policy.act(bgr_dot, pick_instruction)
            action = make_action_vector(out.arm_joints_6, out.base_vx, out.base_wz)

            # Check if object has been picked using Moondream
            pick_query = f"Is the object '{object_query}' grasped by the robot? Answer with 1 for yes and 0 for no."
            pick_answer = 0
            try:
                pick_answer = int(finder.model.query(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), pick_query)["answer"])
            except Exception as e:
                print(f"[WARN] Moondream pick query failed: {e}")
            if pick_answer == 1:
                print("[INFO] Pick routine done (Moondream confirmed grasp).")
                # Switch to basket navigation
                bgr, depth = to_rgb_from_obs_camera(np.asarray(obs["camera"]))
                basket_uv = finder.locate_object(bgr, basket_query)
                # If basket not found, rotate in place until found
                while basket_uv is None:
                    print("[WARN] Moondream did not find the basket; rotating to search...")
                    policy.set_head("base")
                    search_action = make_action_vector(DEFAULT_ARM_HOLD, 0.0, 0.5)
                    obs, reward, terminated, truncated, info = env.step(search_action)
                    bgr, depth = to_rgb_from_obs_camera(np.asarray(obs["camera"]))
                    basket_uv = finder.locate_object(bgr, basket_query)
                    time.sleep(SLEEP_SEC)
                    if terminated or truncated:
                        print("[WARN] Episode ended during basket search.")
                        return
                basket_bgr_dot = draw_red_dot(bgr, basket_uv, DOT_RADIUS_PIX)
                uv = basket_uv
                bgr_dot = basket_bgr_dot
                phase = "navigate_basket"
                t_last = time.time()
                continue

        elif phase == "navigate_basket":
            # Navigate to basket
            dist_m = depth_at_pixel(depth, uv) if depth is not None else None
            if dist_m is not None:
                print(f"[DBG] Distance to basket @ {uv} = {dist_m:.3f} m")
            if (dist_m is None) or (dist_m > DIST_THRESH_M):
                policy.set_head("base")
                nav_instruction = "Go to the red dot."
                out = policy.act(bgr_dot, nav_instruction)
                action = make_action_vector(out.arm_joints_6, out.base_vx, out.base_wz)
            else:
                print("[INFO] Reached basket. Switching to PLACE phase.")
                phase = "place"
                t_last = time.time()
                continue

        elif phase == "place":
            policy.set_head("arm")
            place_instruction = "Align end effector with the red dot and place the object in the basket."
            out = policy.act(bgr_dot, place_instruction)
            action = make_action_vector(out.arm_joints_6, out.base_vx, out.base_wz)

            # Check if object has been placed using Moondream
            place_query = f"Is the object '{object_query}' placed in the basket? Answer with 1 for yes and 0 for no."
            place_answer = 0
            try:
                place_answer = int(finder.model.query(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), place_query)["answer"])
            except Exception as e:
                print(f"[WARN] Moondream place query failed: {e}")
            if place_answer == 1:
                print("[INFO] Place routine done (Moondream confirmed place).")
                break

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Refresh image/depth, optionally re-detect target to handle drift/occlusion
        bgr, depth = to_rgb_from_obs_camera(np.asarray(obs["camera"]))
        if phase in ["navigate", "pick"]:
            uv_new = finder.locate_object(bgr, object_query)
            if uv_new is not None:
                uv = uv_new
            bgr_dot = draw_red_dot(bgr, uv, DOT_RADIUS_PIX)
        elif phase in ["navigate_basket", "place"]:
            uv_new = finder.locate_object(bgr, basket_query)
            if uv_new is not None:
                uv = uv_new
            bgr_dot = draw_red_dot(bgr, uv, DOT_RADIUS_PIX)

        # Exit conditions
        if terminated or truncated:
            if phase == "place":
                print("[INFO] Episode finished during PLACE phase.")
                break
            else:
                print(f"[WARN] Episode finished during {phase.upper()} phase. continuing.")
                # Do not reset the environment, just continue with head switching and phase logic
                continue

        time.sleep(SLEEP_SEC)

    env.close()


if __name__ == "__main__":
    q = "the brown croissant"
    b = "the beige basket"
    main(object_query=q, basket_query=b)