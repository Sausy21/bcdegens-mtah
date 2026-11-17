"""Audit of the scenario tests: do they test football, or only mirror the rules? No match data.

    python game_logic/audit_scenes.py

Three checks, each answering one way the scenes could flatter the rules:

1. Mutation. Switch off each rule (rules.ABLATIONS) and rerun the scenes. A rule that no scene catches
   is untested. A scene that fails for every rule tests nothing specific.
2. Threshold sensitivity. Change each parameter by -20% / +20% and rerun the scenes. A scene that flips
   under a small change was written close to the threshold (or the threshold was set from the scene).
3. Noise. The scenes are idealised: exact distances, exact speeds, no dropouts. Each scene is rerun
   with tracking-like noise (fixed seeds): distances +/-0.3 m, player speed +/-0.5 m/s, ball speed
   +/-1.0 m/s, team speed +/-0.15 m/s (1 sd each), and 3% of frames with the ball dropped.
   Pre-registered: every scene passes in at least 80% of noisy runs; a scene below that relies on
   ideal inputs.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules  # noqa: E402
import test_rules as T  # noqa: E402

SCENES = [(fn, gap) for fn, gap in T.SCENES]
NAMES = [fn.__name__.replace("scene_", "") for fn, _ in SCENES]
BASE_RUN = T.run
BASE_PIPELINE = rules.run_pipeline


def results():
    return np.array([bool(fn()[0]) for fn, _ in SCENES])


def with_pipeline(**kwargs):
    """Rerun all scenes with extra keyword arguments passed to run_pipeline."""
    rules.run_pipeline = lambda *a, **k: BASE_PIPELINE(*a, **{**k, **kwargs})
    try:
        return results()
    finally:
        rules.run_pipeline = BASE_PIPELINE


def mutation(base):
    print("1. MUTATION: scenes that fail when one rule is switched off")
    caught = {}
    for abl in rules.ABLATIONS:
        res = with_pipeline(disable=(abl,))
        broke = [NAMES[i] for i in range(len(NAMES)) if base[i] and not res[i]]
        caught[abl] = broke
        print("  {:<20} {:>2} scene(s): {}".format(abl, len(broke), ", ".join(broke) if broke else "-- NONE (untested)"))
    per_scene = {n: sum(n in v for v in caught.values()) for n in NAMES}
    always = [n for n, c in per_scene.items() if c == len(rules.ABLATIONS)]
    print("  scenes failing under every ablation: {}".format(", ".join(always) or "none"))
    return caught


def sensitivity(base):
    print("\n2. THRESHOLD SENSITIVITY: scenes that flip when one parameter moves by -20% / +20%")
    flips = {}
    for key, val in rules.PARAMS.items():
        out = []
        for f in (0.8, 1.2):
            v = val * f
            v = int(round(v)) if isinstance(val, int) else v
            res = with_pipeline(**{key: v})
            out.append([NAMES[i] for i in range(len(NAMES)) if res[i] != base[i]])
        if out[0] or out[1]:
            flips[key] = out
            print("  {:<22} -20%: {:<45} +20%: {}".format(key, ", ".join(out[0]) or "-", ", ".join(out[1]) or "-"))
    print("  parameters with no flip: {}".format(", ".join(k for k in rules.PARAMS if k not in flips)))
    return flips


def noisy_run_factory(rng):
    def noisy_run(t, tag_team=None):
        t = t.copy()
        n = len(t)
        for col, sd, lo in (("d_H", 0.3, 0.0), ("d_A", 0.3, 0.0), ("s_near_H", 0.5, 0.0), ("s_near_A", 0.5, 0.0),
                            ("ball_speed", 1.0, 0.0), ("mean_speed", 0.15, 0.0)):
            t[col] = np.maximum(lo, t[col].to_numpy(float) + rng.normal(0, sd, n))
        drop = rng.random(n) < 0.03
        t.loc[drop, ["ball_x_m", "ball_y_m", "ball_speed", "d_H", "d_A", "s_near_H", "s_near_A"]] = np.nan
        t.loc[drop, ["ball_present", "ball_out"]] = False
        return BASE_RUN(t, tag_team)
    return noisy_run


def noise(base, draws=20):
    print("\n3. NOISE: pass rate over {} noisy runs per scene (pre-registered: >= 80%)".format(draws))
    rates = {}
    for i, (fn, gap) in enumerate(SCENES):
        if fn.__name__ == "scene_feature_table_roundtrip":
            continue                                   # compares two runs; independent noise would differ
        ok = 0
        for seed in range(draws):
            T.run = noisy_run_factory(np.random.default_rng(1000 + seed))
            try:
                ok += bool(fn()[0])
            finally:
                T.run = BASE_RUN
        rates[NAMES[i]] = ok / draws
    for name, r in sorted(rates.items(), key=lambda kv: kv[1]):
        flag = "" if r >= 0.8 else "   <-- below 80%"
        print("  {:<36} {:>4.0f}%{}".format(name, 100 * r, flag))
    return rates


if __name__ == "__main__":
    base = results()
    print("baseline: {} of {} scenes pass\n".format(int(base.sum()), len(base)))
    mutation(base)
    sensitivity(base)
    noise(base)
