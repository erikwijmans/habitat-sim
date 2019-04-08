#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat_sim.bindings as hsim
import habitat_sim
import habitat_sim.utils
import habitat_sim.errors
import numpy as np
import quaternion
import pytest
import os.path as osp
import itertools
from examples.settings import make_cfg
import json


_test_scenes = [
    osp.abspath(
        osp.join(
            osp.dirname(__file__),
            "../data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",
        )
    ),
    osp.abspath(
        osp.join(
            osp.dirname(__file__),
            "../data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        )
    ),
    osp.abspath(
        osp.join(
            osp.dirname(__file__),
            "../data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
        )
    ),
]


@pytest.mark.gfxtest
@pytest.mark.parametrize(
    "sensor_type,scene,has_sem",
    list(
        itertools.product(
            ["color_sensor", "depth_sensor", "semantic_sensor"],
            _test_scenes[0:1],
            [True],
        )
    )
    + list(
        itertools.product(["color_sensor", "depth_sensor"], _test_scenes[1:], [False])
    ),
)
def test_sensors(sensor_type, scene, has_sem, sim, make_cfg_settings):
    if not osp.exists(scene):
        pytest.skip("Skipping {}".format(scene))

    make_cfg_settings = {k: v for k, v in make_cfg_settings.items()}
    make_cfg_settings["semantic_sensor"] = has_sem
    make_cfg_settings["scene"] = scene
    sim.reconfigure(make_cfg(make_cfg_settings))
    with open(
        osp.abspath(
            osp.join(
                osp.dirname(__file__),
                "gt_data",
                "{}-state.json".format(osp.basename(osp.splitext(scene)[0])),
            )
        ),
        "r",
    ) as f:
        render_state = json.load(f)
        state = habitat_sim.AgentState()
        state.position = render_state["pos"]
        state.rotation = habitat_sim.utils.quat_from_coeffs(render_state["rot"])

    sim.initialize_agent(0, state)
    obs = sim.step("move_forward")

    #  sim.seed(5)
    #  np.random.seed(5)
    #  sim.initialize_agent(0)
    #  with open(
    #  osp.abspath(
    #  osp.join(
    #  osp.dirname(__file__),
    #  "gt_data",
    #  "{}-state.json".format(osp.basename(osp.splitext(scene)[0])),
    #  )
    #  ),
    #  "w",
    #  ) as f:
    #  json.dump(
    #  dict(
    #  pos=sim.get_agent(0).state.position.tolist(),
    #  rot=habitat_sim.utils.quat_to_coeffs(
    #  sim.get_agent(0).state.rotation
    #  ).tolist(),
    #  ),
    #  f,
    #  )

    #  np.save(
    #  osp.abspath(
    #  osp.join(
    #  osp.dirname(__file__),
    #  "gt_data",
    #  "{}-{}.npy".format(osp.basename(osp.splitext(scene)[0]), sensor_type),
    #  )
    #  ),
    #  obs[sensor_type],
    #  )
    #  if sensor_type == "color_sensor":
    #  import cv2

    #  cv2.imwrite(
    #  "{}-{}.png".format(osp.basename(osp.splitext(scene)[0]), sensor_type),
    #  obs[sensor_type],
    #  )

    assert sensor_type in obs, f"{sensor_type} not in obs"

    gt = np.load(
        osp.abspath(
            osp.join(
                osp.dirname(__file__),
                "gt_data",
                "{}-{}.npy".format(osp.basename(osp.splitext(scene)[0]), sensor_type),
            )
        )
    )

    if sensor_type == "semantic_sensor":
        assert np.sum(np.abs((obs[sensor_type] - gt))) < 1e-4 * np.prod(
            gt.shape[0:2]
        ), f"Incorrect {sensor_type} output"
    else:
        # Different GPUs and different driver version will produce slightly different
        # images for RGB and depth, so use a big tolerance
        assert np.linalg.norm(
            obs[sensor_type].astype(np.float) - gt.astype(np.float)
        ) < 5e-3 * np.linalg.norm(
            gt.astype(np.float)
        ), f"Incorrect {sensor_type} output"
