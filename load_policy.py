from dm_control.locomotion.tasks.reference_pose import types
from mocapact.envs import tracking
from mocapact import observables
from mocapact.sb3 import utils
import numpy as np
from dm_control.viewer import application
import mujoco
from dm_control import mjcf
from functools import partial
import re
import copy
import json

model_name = "CMU_049_08"
start_step = 0
end_step = 156
expert_path = f"data/experts/{model_name}-{start_step}-{end_step}/eval_rsi/model"
expert = utils.load_policy(expert_path, observables.TIME_INDEX_OBSERVABLES)

dataset = types.ClipCollection(
    ids=[model_name], start_steps=[start_step], end_steps=[end_step]
)
env = tracking.MocapTrackingGymEnv(
    dataset, task_kwargs={"always_init_at_clip_start": True}
)
obs, done = env.reset(), False

state = None


def policy_fn(time_step):
    global state
    if time_step.step_type == 0:  # first time step
        state = None
    # action, state = expert.predict(
    #     env.get_observation(time_step), state, deterministic=True
    # )
    return np.zeros((56,))
    return action


tmodel = None
xipos = []
ximat = []
write_xipos = False
write_ximat = False

printed_once = False


def callback(model, data, scene, env):
    global tmodel
    global xipos
    global ximat
    global printed_once
    if tmodel is None:
        tmodel = model
    if not printed_once:
        print(env.physics.named.model.body_pos)
        print(env.physics.named.model.body_ipos)
        print(env.physics.named.model.body_quat)
        print(env.physics.named.model.body_inertia)
        print(model.nmocap)
        printed_once = True
    if write_xipos:
        xipos.append(copy.deepcopy(env.physics.data.xipos))
    if write_ximat:
        ximat.append(copy.deepcopy(env.physics.data.ximat))


scene_callback = partial(callback, env=env)

viewer_app = application.Application(
    title="Output", width=1024, height=768, scene_callback=scene_callback
)
viewer_app.launch(
    environment_loader=env.dm_env,
    policy=policy_fn,
)

# xipos = re.sub(r"array\(([\[\]\s\d\.e\+,\n\-]*)\)", r"\1", str(xipos))
# ximat = re.sub(r"array\(([\[\]\s\d\.e\+,\n\-]*)\)", r"\1", str(ximat))

#  https://stackoverflow.com/a/47626762/10168590
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if write_xipos:
    with open("xipos.json", "w") as f:
        json.dump({"xipos": xipos}, f, indent=4, cls=NumpyEncoder)
if write_ximat:
    with open("ximat.json", "w") as f:
        json.dump({"ximat": ximat}, f, indent=4, cls=NumpyEncoder)
# print(xipos)
