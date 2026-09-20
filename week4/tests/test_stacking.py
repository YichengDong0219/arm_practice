import unittest

from week4.common.motion import DryRunSerial, run_sequence
from week4.task10.stacking import (
    BASE_PLACE_POSE,
    HORIZONTAL_TOOL_ANGLE_DEG,
    PICKUP_OUTWARD_OFFSET_MM,
    PICKUP_POSE,
    STACK_INWARD_OFFSET_MM,
    build_cycle_steps,
    forward_tool_pose,
    generate_stacking_plan,
    preflight,
    validate_plan,
)


class StackingTests(unittest.TestCase):
    def setUp(self):
        self.calibration = generate_stacking_plan(30.0, 3, 40.0)

    def test_three_layers_keep_xy_and_add_30_mm(self):
        models = [forward_tool_pose(pose) for pose in self.calibration["stack_release_poses"]]
        for layer, model in enumerate(models):
            self.assertAlmostEqual(model["x_mm"], models[0]["x_mm"], places=6)
            self.assertAlmostEqual(model["y_mm"], models[0]["y_mm"], places=6)
            self.assertAlmostEqual(model["z_mm"], models[0]["z_mm"] + 30.0 * layer, places=6)

    def test_first_layer_is_derived_from_main_place_coordinate(self):
        source = forward_tool_pose(BASE_PLACE_POSE)
        first = forward_tool_pose(self.calibration["stack_release_poses"][0])
        self.assertAlmostEqual(first["radial_mm"], source["radial_mm"] - STACK_INWARD_OFFSET_MM)
        self.assertAlmostEqual(first["z_mm"], source["z_mm"])

    def test_place_and_hover_poses_keep_gripper_horizontal(self):
        poses = self.calibration["stack_release_poses"] + self.calibration["stack_hover_poses"]
        for pose in poses:
            model = forward_tool_pose(pose)
            self.assertAlmostEqual(model["tool_angle_deg"], HORIZONTAL_TOOL_ANGLE_DEG, places=6)
            self.assertAlmostEqual(sum(pose[1:4]), -90.0, places=6)

    def test_pickup_descent_is_vertical_and_horizontal(self):
        source = forward_tool_pose(PICKUP_POSE)
        pickup = forward_tool_pose(self.calibration["pickup_pose"])
        hover = forward_tool_pose(self.calibration["pickup_hover_pose"])
        self.assertAlmostEqual(pickup["radial_mm"], source["radial_mm"] + PICKUP_OUTWARD_OFFSET_MM)
        self.assertAlmostEqual(pickup["x_mm"], hover["x_mm"], places=6)
        self.assertAlmostEqual(pickup["y_mm"], hover["y_mm"], places=6)
        self.assertAlmostEqual(hover["z_mm"] - pickup["z_mm"], 40.0, places=6)
        self.assertAlmostEqual(sum(self.calibration["pickup_pose"][1:4]), -90.0, places=6)
        self.assertAlmostEqual(sum(self.calibration["pickup_hover_pose"][1:4]), -90.0, places=6)

    def test_generated_plan_passes_preflight(self):
        preflight(self.calibration, 30.0, 3, 0)

    def test_height_and_count_mismatch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "高度"):
            validate_plan(self.calibration, 20.0, 3)
        with self.assertRaisesRegex(ValueError, "层数"):
            validate_plan(self.calibration, 30.0, 4)

    def test_invalid_start_layer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "start-layer"):
            preflight(self.calibration, 30.0, 3, 3)

    def test_each_cycle_closes_opens_and_returns_home(self):
        steps = build_cycle_steps(self.calibration, 0)
        self.assertGreater(len(steps), 7)
        gripper_targets = [step.target_deg[5] for step in steps]
        self.assertIn(-40.0, gripper_targets)
        self.assertEqual(gripper_targets[-1], 0.0)
        link = DryRunSerial()
        final_pose = run_sequence(link, steps, realtime=False)
        self.assertEqual(final_pose, (0.0,) * 6)
        self.assertEqual(len(link.frames), sum(step.steps for step in steps))

        vertical_targets = [
            step.target_deg for step in steps if "竖直下降到第1层" in step.name
        ]
        self.assertEqual(len(vertical_targets), 10)
        models = [forward_tool_pose(pose) for pose in vertical_targets]
        for model in models:
            self.assertAlmostEqual(model["x_mm"], models[0]["x_mm"], places=6)
            self.assertAlmostEqual(model["y_mm"], models[0]["y_mm"], places=6)
            self.assertAlmostEqual(model["tool_angle_deg"], 0.0, places=6)
        for pose in vertical_targets:
            self.assertAlmostEqual(sum(pose[1:4]), -90.0, places=6)

if __name__ == "__main__":
    unittest.main()
