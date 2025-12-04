import gymnasium as gym
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import GymWrapper
import robocasa


class RoboCasaWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = self.env.observation_space
        new_obs_space = {}

        for k, v in obs_space.items():
            if "agentview_left_image" in k:
                new_obs_space["video.left_view"] = v
            if "agentview_right_image" in k:
                new_obs_space["video.right_view"] = v
            if "eye_in_hand_image" in k:
                new_obs_space["video.wrist_view"] = v
        
        robosuite_env = self.env.env
        robot = robosuite_env.robots[0]
        gripper_dim = 0
        for arm in robot.arms:
            if robot.has_gripper[arm]:
                gripper_dim += robot.gripper[arm].dof
        if gripper_dim == 0:
            gripper_dim = 2

        new_obs_space["state.base_position"] = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64)
        new_obs_space["state.base_rotation"] = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64)
        new_obs_space["state.end_effector_position_relative"] = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64)
        new_obs_space["state.end_effector_rotation_relative"] = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64)
        new_obs_space["state.gripper_qpos"] = gym.spaces.Box(-np.inf, np.inf, shape=(gripper_dim,), dtype=np.float64)
                
        self.observation_space = gym.spaces.Dict(new_obs_space)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def step(self, action):
        flat_action = self._flatten_action(action)
        obs, reward, terminated, truncated, info = self.env.step(flat_action)
        info["is_success"] = self.env.env._check_success()
        return self._process_obs(obs), reward, terminated, truncated, info

    def _flatten_action(self, action):
        robot = self.env.env.robots[0]
        split_indexes = robot._action_split_indexes
        target_dim = robot.action_dim
        vec = np.zeros(target_dim)
        
        eef_pos = np.asanyarray(action.get("action.end_effector_position", np.zeros(3))).flatten()
        eef_rot = np.asanyarray(action.get("action.end_effector_rotation", np.zeros(3))).flatten()
        gripper = np.asanyarray(action.get("action.gripper_close", np.zeros(1))).flatten()
        base_motion = np.asanyarray(action.get("action.base_motion", np.zeros(4))).flatten()
        control_mode = np.asanyarray(action.get("action.control_mode", np.zeros(1))).flatten()
        aux_values = np.concatenate([base_motion, control_mode]) # Total 5 values
        aux_idx = 0
        sorted_indexes = sorted(split_indexes.items(), key=lambda x: x[1][0])
        
        for part, (start, end) in sorted_indexes:
            dim = end - start
            if dim == 0: continue
            
            if part in ["right", "left", "arm"]:
                if dim == 6:
                    vec[start:end] = np.concatenate([eef_pos, eef_rot])
            elif "gripper" in part:
                if dim == 1:
                    vec[start:end] = gripper
            elif part in ["base", "torso", "head"]:
                if aux_idx + dim <= len(aux_values):
                    vec[start:end] = aux_values[aux_idx : aux_idx + dim]
                    aux_idx += dim
        return vec

    def _process_obs(self, obs):
        new_obs = {}
        if "robot0_agentview_left_image" in obs:
            new_obs["video.left_view"] = obs["robot0_agentview_left_image"]
        if "robot0_agentview_right_image" in obs:
            new_obs["video.right_view"] = obs["robot0_agentview_right_image"]
        if "robot0_eye_in_hand_image" in obs:
            new_obs["video.wrist_view"] = obs["robot0_eye_in_hand_image"]

        robosuite_env = self.env.env
        robot = robosuite_env.robots[0]
        sim = robosuite_env.sim
        prefix = robot.robot_model.naming_prefix

        base_body_name = f"{prefix}base"
        base_id = sim.model.body_name2id(base_body_name)
        base_pos = np.array(sim.data.body_xpos[base_id])
        base_quat = T.convert_quat(sim.data.body_xquat[base_id], to="xyzw")
        new_obs["state.base_position"] = base_pos
        new_obs["state.base_rotation"] = base_quat
        eef_site_id = robot.eef_site_id["right"]
        eef_pos = np.array(sim.data.site_xpos[eef_site_id])
        eef_mat = np.array(sim.data.site_xmat[eef_site_id]).reshape(3, 3)
        eef_quat = T.mat2quat(eef_mat)
        
        T_wb = T.pose2mat((base_pos, base_quat))
        T_we = T.pose2mat((eef_pos, eef_quat))
        T_be = np.linalg.inv(T_wb) @ T_we
        
        rel_pos = T_be[:3, 3]
        rel_mat = T_be[:3, :3]
        rel_quat = T.mat2quat(rel_mat) # xyzw
        
        new_obs["state.end_effector_position_relative"] = rel_pos
        new_obs["state.end_effector_rotation_relative"] = rel_quat
        
        vals = []
        indices = robot._ref_gripper_joint_pos_indexes["right"]
        vals.append(sim.data.qpos[indices])
        new_obs["state.gripper_qpos"] = np.concatenate(vals)

        return new_obs

    def render(self, mode="rgb_array", width=None, height=None, camera_name=None):
        if mode == "rgb_array":
            env = self.env.env
            sim = env.sim
            
            if camera_name is None:
                if hasattr(env, "camera_names") and env.camera_names:
                    for cam in env.camera_names:
                        if "agentview" in cam and "center" in cam:
                            camera_name = cam
                            break
                    if camera_name is None:
                        camera_name = env.camera_names[0]
                else:
                    camera_name = "frontview"

            if width is None:
                if hasattr(env, "camera_widths") and env.camera_widths:
                     width = env.camera_widths[0]
                else:
                     width = 256
            
            if height is None:
                if hasattr(env, "camera_heights") and env.camera_heights:
                     height = env.camera_heights[0]
                else:
                     height = 256

            im = sim.render(width=width, height=height, camera_name=camera_name, depth=False)
            return np.flipud(im)
        else:
            return self.env.render()

def load_robocasa_gym_env(
    env_name,
    seed=None,
    robots=None,
    camera_widths=256,
    camera_heights=256,
    render_onscreen=False,
    camera_names=None,
    obj_instance_split="A",
    generative_textures=None,
    randomize_cameras=False,
    layout_ids=None,
    style_ids=None,
    collect_data=False,
    **kwargs,
):    
    args = {
        "env_name": env_name,
        "robots": robots,
        "camera_widths": camera_widths,
        "camera_heights": camera_heights,
        "has_renderer": render_onscreen,
        "has_offscreen_renderer": not render_onscreen,
        "use_camera_obs": True,
        "use_object_obs": True,
        "control_freq": 20,
    }
    
    if camera_names is not None:
        args["camera_names"] = camera_names
        
    args.update(kwargs)
    extra_kwargs = {}
    if layout_ids is not None:
        extra_kwargs["layout_ids"] = layout_ids
    if style_ids is not None:
        extra_kwargs["style_ids"] = style_ids
    if generative_textures is not None:
        extra_kwargs["generative_textures"] = generative_textures
    if obj_instance_split is not None:
        extra_kwargs["obj_instance_split"] = obj_instance_split

    env = robosuite.make(**args, **extra_kwargs)
    env = GymWrapper(env, keys=None, flatten_obs=False)
    
    return env
