"""Camera-free runs: the same yam_arena embodiment as ``inspect-robots`` (same env, reset and settle, the
task's score every step), only built without cameras, so it runs on GPUs without RT cores (A30, A100).
For probes and for checking the scripted demo's score; videos and the Dream-1 run use an A10.

    python headless.py demo task.py OUT [--max-steps 2400]     # the scripted demo, no video
    python headless.py probe task.py                           # probe_start.py's report
"""

from __future__ import annotations

import argparse
import atexit
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def embodiment(task_path: str, control_hz: float = 20.0):
    import yam_arena.inspect_embodiment as ie
    from inspect_robots.types import Observation

    class Headless(ie.YamArenaEmbodiment):
        def _ensure_sim(self, goal) -> None:
            if self._env is not None:
                return
            os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            if ie._APP is None:
                from isaaclab.app import AppLauncher

                ie._APP = AppLauncher(headless=True, device=self.device, enable_cameras=False).app
                atexit.register(ie._close_app_at_exit)
            import torch
            from yam_arena.embodiment import BimanualYamEmbodiment
            from yam_arena.scene import make_env

            emb = BimanualYamEmbodiment(enable_cameras=False, wrist_cam_mount=self.wrist_cam_mount,
                                        contact_sensors=False, rig=self.rig)
            self._free_objects = self.task.free_objects
            self._env = make_env(emb, scene="task", device=self.device, control_hz=self.info.control_hz,
                                 task=self.task)
            self._base = self._env.unwrapped
            self._goal = goal
            self._torch = torch
            self._ready = torch.tensor([[*ie.READY, 1.0, *ie.READY, 1.0]], dtype=torch.float32,
                                       device=self._base.device)
            self._ids = {name: (self._base.scene[name].find_joints(ie.ARM_JOINTS, preserve_order=True)[0],
                                self._base.scene[name].find_joints(ie.FINGER_JOINTS, preserve_order=True)[0])
                         for name in ie.ARMS}

        def _observation(self):
            return Observation(images={}, state={}, instruction=self._instruction, image_times={},
                               state_time=self._t)

    return Headless(task=str(Path(task_path).resolve()), object_jitter=0, control_hz=control_hz)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("demo", "probe"))
    ap.add_argument("task")
    ap.add_argument("out", nargs="?", default="headless_out")
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--log-every", type=int, default=20)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    emb = embodiment(a.task)
    if a.what == "demo":
        import demo

        demo.run(a.task, a.out, a.max_steps, record=False, variant=os.environ.get("DEMO_VARIANT", ""),
                 log_every=a.log_every, emb=emb)
    else:
        import probe_start

        probe_start.main(a.task, emb=emb)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
