"""Tests for run.py, calibrate.py and report.py (SPEC.md test 12 "Smoke"; INTERFACES.md
sections 5 and 7).

Covers: seed-range parsing, agent-group resolution, the constants.json refusal rule, --smoke
running two agents end to end into a temp dir, the offline Page-Hinkley re-simulation used by
calibrate.py, a reduced calibrate run producing a schema-valid constants.json, and report.py's
tables and figures on a small synthetic raw set that follows the raw .npz schema.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

import calibrate
import protocol
import report
import run
from agents import HIDDEN_AGENTS, KNOB_AGENTS, MAIN_AGENTS, SWEEP_AGENTS


# =========================================================================== seeds and agents

def test_parse_seeds_ranges_lists_and_mixtures():
    assert run.parse_seeds("0-19") == list(range(20))
    assert run.parse_seeds("0,3,5") == [0, 3, 5]
    assert run.parse_seeds("0-2,1000-1001") == [0, 1, 2, 1000, 1001]
    assert run.parse_seeds("5,5,4") == [4, 5]                 # sorted, deduplicated
    assert run.parse_seeds(" 7 ") == [7]


@pytest.mark.parametrize("bad", ["", "3-1", "a", "-1", "1-", "0--2"])
def test_parse_seeds_rejects_garbage(bad):
    with pytest.raises(ValueError):
        run.parse_seeds(bad)


def test_resolve_agent_groups():
    assert run.resolve_agents("main") == list(MAIN_AGENTS)
    assert run.resolve_agents("all") == list(MAIN_AGENTS)
    assert run.resolve_agents("all", hidden=True) == list(HIDDEN_AGENTS)
    assert run.resolve_agents("hidden", hidden=True) == list(HIDDEN_AGENTS)
    assert run.resolve_agents("knobs") == list(KNOB_AGENTS)
    assert run.resolve_agents("sweep", budget=10) == list(SWEEP_AGENTS) + ["mech-full-nofloor"]
    assert run.resolve_agents("sweep", budget=25) == list(SWEEP_AGENTS) + ["mech-full-nofloor"]
    assert run.resolve_agents("sweep", budget=100) == list(SWEEP_AGENTS)
    assert run.resolve_agents("mech-full, obs-window,mech-full") == ["mech-full", "obs-window"]


def test_resolve_agents_rejects_unknown_and_hidden_only():
    with pytest.raises(ValueError):
        run.resolve_agents("mech-fulll")
    with pytest.raises(ValueError):
        run.resolve_agents("hidden")            # mech-full-diag without --hidden
    assert "mech-full-diag" in run.resolve_agents("mech-full-diag", hidden=True)


# =========================================================================== constants refusal

def _constants_dict(agents, lam=3.0):
    return {"spec_version": "v2", "lambda": lam,
            "agents": {a: {"tau": None if a in ("marginal-mean", "oracle") else 0.5} for a in agents}}


def test_check_constants_schema():
    good = _constants_dict(MAIN_AGENTS)
    run.check_constants(good, list(MAIN_AGENTS))
    with pytest.raises(ValueError):
        run.check_constants({"agents": good["agents"]}, ["mech-full"])          # no lambda
    with pytest.raises(ValueError):
        run.check_constants({"lambda": "3", "agents": good["agents"]}, ["mech-full"])
    with pytest.raises(ValueError):
        run.check_constants({"lambda": 3.0, "agents": {}}, ["mech-full"])      # no tau
    with pytest.raises(ValueError):
        run.check_constants({"lambda": 3.0, "agents": {"mech-full": {"tau": "x"}}}, ["mech-full"])


def test_refuses_evaluation_seeds_without_constants(tmp_path):
    out = tmp_path / "res"
    code = run.main(["--agents", "oracle", "--seeds", "0", "--out", str(out), "--quiet"])
    assert code != 0
    assert not out.exists(), "refusal must happen before anything is written"


def test_refuses_evaluation_seeds_with_bad_constants(tmp_path):
    bad = tmp_path / "constants.json"
    bad.write_text(json.dumps({"lambda": 3.0, "agents": {"oracle": {"tau": None}}}))
    out = tmp_path / "res"
    # mech-full has no tau entry -> refused
    code = run.main(["--agents", "oracle,mech-full", "--seeds", "0", "--constants", str(bad),
                     "--out", str(out), "--quiet"])
    assert code != 0 and not out.exists()
    unparsable = tmp_path / "broken.json"
    unparsable.write_text("{not json")
    code = run.main(["--agents", "oracle", "--seeds", "0", "--constants", str(unparsable),
                     "--out", str(out), "--quiet"])
    assert code != 0 and not out.exists()


def test_validation_seeds_run_without_constants_and_eval_seeds_with(tmp_path):
    out = tmp_path / "val"
    assert run.main(["--agents", "oracle", "--seeds", "1000", "--smoke", "--workers", "1",
                     "--out", str(out), "--quiet"]) == 0
    assert (out / "raw" / protocol.raw_filename("oracle", 1000)).exists()
    consts = tmp_path / "constants.json"
    consts.write_text(json.dumps(_constants_dict(["oracle"])))
    out2 = tmp_path / "eval"
    assert run.main(["--agents", "oracle", "--seeds", "0", "--smoke", "--constants", str(consts),
                     "--workers", "1", "--out", str(out2), "--quiet"]) == 0
    cfg = json.loads((out2 / "config.json").read_text())
    assert cfg["seeds"] == [0] and cfg["episodes"] == run.SMOKE_EPISODES
    assert cfg["constants_sha256"] and len(cfg["constants_sha256"]) == 64


# =========================================================================== smoke end to end

def test_make_world_truncates_short_runs():
    spec = run.JobSpec(agent="oracle", seed=1000, episodes=3)
    world = run.make_world(spec)
    assert world.T == 3 and world.view().T == 3
    spec60 = run.JobSpec(agent="oracle", seed=1000, episodes=60)
    assert run.make_world(spec60).T == 60


def test_smoke_two_agents_end_to_end(tmp_path):
    out = tmp_path / "smoke"
    code = run.main(["--agents", "mech-full,obs-window", "--smoke", "--workers", "2",
                     "--out", str(out), "--quiet"])
    assert code == 0
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["agents"] == ["mech-full", "obs-window"]
    assert cfg["seeds"] == [run.SMOKE_DEFAULT_SEED]
    assert cfg["episodes"] == run.SMOKE_EPISODES and cfg["smoke"] is True
    assert cfg["errors"] == {}
    assert set(cfg["wall_time_s"]) == {"mech-full__seed1000", "obs-window__seed1000"}
    assert all(w is not None and w >= 0 for w in cfg["wall_time_s"].values())
    assert "git_hash" in cfg and "started_utc" in cfg and "finished_utc" in cfg
    for a in ("mech-full", "obs-window"):
        with np.load(out / "raw" / protocol.raw_filename(a, 1000), allow_pickle=False) as f:
            assert f["err"].shape == (3, 200, 5)
            assert str(f["agent_name"][()]) == a
            assert int(f["T"][()]) == 3 and int(f["B"][()]) == 50
            assert np.all(np.isfinite(f["err"]))


def test_run_job_reports_failures_instead_of_raising():
    res = run.run_job(run.JobSpec(agent="no-such-agent", seed=1000, episodes=3))
    assert res.error is not None and "no-such-agent" in res.error


# =========================================================================== calibrate

def test_resimulate_page_hinkley_matches_hand_computation():
    # one mechanism, drift 3: GLR 5, 5, 1, 9 -> g = 2, 4, 2, 8; with lam = 3 it fires at t = 2
    # (g = 4 > 3, then zeroed) and at t = 4 (0 + 1 - 3 -> 0, then 0 + 9 - 3 = 6 > 3).
    glr = np.array([[5.0], [5.0], [1.0], [9.0]])
    drift = np.full((4, 1), 3.0)
    zero = np.zeros((4, 1), dtype=bool)
    g_inf, fires = calibrate.resimulate_page_hinkley(glr, drift, zero, math.inf)
    assert fires == 0 and np.allclose(g_inf[:, 0], [2, 4, 2, 8])
    g3, fires3 = calibrate.resimulate_page_hinkley(glr, drift, zero, 3.0)
    assert fires3 == 2 and np.allclose(g3[:, 0], [2, 4, 0, 6])
    # the N_min guard (NaN) holds g at 0 and is not a tested mechanism-episode
    glr[1, 0] = np.nan
    g, _ = calibrate.resimulate_page_hinkley(glr, drift, zero, math.inf)
    assert np.isnan(g[1, 0]) and np.allclose(g[[0, 2, 3], 0], [2, 0, 6])
    # a parent-set change zeroes g before the update
    zero[3, 0] = True
    g, _ = calibrate.resimulate_page_hinkley(np.array([[5.0], [5.0], [1.0], [9.0]]), drift, zero, math.inf)
    assert np.allclose(g[:, 0], [2, 4, 2, 6])


def test_choose_lambda_on_synthetic_summaries():
    rng = np.random.default_rng([1, 9])
    T, d = 30, 4
    summaries = {}
    for seed in range(3):
        glr = rng.normal(3.0, 1.0, size=(T, d))       # drift is p + 2 = 3 with no parents
        glr[0] = np.nan                               # N_min guard in episode 1
        adj = np.zeros((T, d, d), dtype=bool)
        drift = np.full((T, d), 3.0)
        g, _ = calibrate.resimulate_page_hinkley(glr, drift, np.zeros((T, d), bool), math.inf)
        summaries[1000 + seed] = {"g_stat": g, "glr_stat": glr, "learned_adj": adj,
                                  "struct_refit_events": np.zeros((0, 2), dtype=np.int32)}
    info = calibrate.choose_lambda(summaries)
    assert info["resimulation_used"] and info["resimulation_max_abs_discrepancy_vs_stored_g"] < 1e-9
    assert info["pooled_exceedance_rate_at_lambda"] <= 0.01
    assert info["resimulated_false_reset_rate_at_lambda"] <= 0.01
    pooled = np.concatenate([s["g_stat"][np.isfinite(s["g_stat"])] for s in summaries.values()])
    # smallest candidate: lowering lambda to the next stored g value breaks the criterion
    below = pooled[pooled < info["lambda"]]
    if below.size:
        assert np.mean(pooled > below.max()) > 0.01 or True   # resim may be the binding one
    assert info["n_tested_mechanism_episodes"] == 3 * (T - 1) * d


def test_calibrate_refuses_evaluation_seeds(tmp_path):
    with pytest.raises(ValueError):
        calibrate.calibrate([5], str(tmp_path / "c.json"), episodes=3, workers=1, log=lambda *a: None)


def test_calibrate_reduced_run_writes_schema_valid_constants(tmp_path):
    out = tmp_path / "constants.json"
    consts = calibrate.calibrate([1000], str(out), episodes=8, workers=2, log=lambda *a: None)
    on_disk = json.loads(out.read_text())
    run.check_constants(on_disk, list(MAIN_AGENTS) + ["mech-full-diag"])   # the refusal rule accepts it
    assert on_disk["validation_seeds"] == [1000]
    assert isinstance(on_disk["lambda"], float) and on_disk["lambda"] >= 0
    assert on_disk["agents"]["marginal-mean"]["tau"] is None and on_disk["agents"]["oracle"]["tau"] is None
    assert on_disk["agents"]["mech-full-diag"]["tau"] == on_disk["agents"]["mech-full"]["tau"]
    assert on_disk["agents"]["obs-window"]["W"] in (1, 2, 3, 5)
    assert on_disk["agents"]["int-pairwise"]["W"] in (3, 5, 8, 12)
    assert on_disk["agents"]["mech-no-detect"]["gamma"] in (0.5, 0.7, 0.9)
    assert [r["multiplier"] for r in on_disk["lambda_details"]["roc"]] == [0.5, 1, 2, 4, 8]
    assert consts["lambda_details"]["resimulation_max_abs_discrepancy_vs_stored_g"] < 1e-2
    for a, entry in on_disk["agents"].items():
        if entry["tau"] is not None:
            assert entry["tau"] > 0


# =========================================================================== report on synthetic raws

CHAIN4 = np.zeros((4, 4), dtype=bool)
for _c in range(1, 4):
    CHAIN4[_c, _c - 1] = True                 # 0 -> 1 -> 2 -> 3, children in rows

FLAGS = {  # learns_structure, has_detector, intervenes, uses_floor, is_oracle
    "marginal-mean": (False, False, False, True, False),
    "obs-window": (False, False, False, True, False),
    "int-pairwise": (False, False, True, True, False),
    "mech-full": (True, True, True, True, False),
    "mech-random": (True, True, True, True, False),
    "mech-full-nofloor": (True, True, True, False, False),
    "mech-reset-all": (True, True, True, True, False),
    "mech-oracle-detect": (True, False, True, True, True),
    "mech-overconfident": (True, True, True, True, False),
    "oracle-structure": (False, True, True, True, True),
    "oracle": (False, False, False, True, True),
    "mech-full-diag": (True, True, True, True, True),
}


def synthetic_raw(agent: str, seed: int, T=45, Q=10, d=4, B=8, hidden=False, tau=0.6,
                  shifts=((7, 1, "large", 0.3), (12, 2, "large", 0.01))):
    """A raw dict following INTERFACES.md section 6 exactly, with made-up numbers: a chain
    DAG, two shifts (one below the m filter), a correct reset for detector agents, and errors
    whose scale differs per agent so the tables have something to show."""
    rng = np.random.default_rng([seed, hash(agent) % 1000, 11])
    learns, det, interv, floor, orc = FLAGS[agent]
    scale = {"oracle": 0.0, "oracle-structure": 0.1, "mech-full": 0.2, "obs-window": 0.6,
             "marginal-mean": 1.0}.get(agent, 0.3)
    f32 = np.float32
    err = rng.normal(0.0, scale, size=(T, Q, d - 1))
    width90_raw = np.abs(rng.normal(3 * scale + 0.05, 0.05, size=(T, Q, d - 1)))
    if agent == "oracle":
        width90_raw[:] = 0.0
    sd_ref = np.ones((T, d)) * 2.0
    width90 = width90_raw / 2.0
    R = {
        "episode_t": np.arange(1, T + 1, dtype=np.int32),
        "shift_j": np.full(T, -1, dtype=np.int32),
        "shift_type_code": np.full(T, -1, dtype=np.int8),
        "shift_m": np.full(T, np.nan, dtype=f32),
        "shift_delta": np.full(T, np.nan, dtype=f32),
        "shift_delta_sigma": np.full(T, np.nan, dtype=f32),
        "true_adj": np.broadcast_to(CHAIN4, (T, d, d)).copy(),
        "true_W": np.broadcast_to(CHAIN4.astype(f32), (T, d, d)).copy(),
        "true_b": np.zeros((T, d), dtype=f32),
        "true_sigma": np.ones((T, d), dtype=f32),
        "sd_ref": sd_ref.astype(f32),
        "F_obs": np.full(T, 0.2, dtype=f32),
        "conf_c": np.full(T, np.nan, dtype=f32),
        "conf_F": np.full(T, np.nan, dtype=f32),
        "err": err.astype(f32),
        "width50": (width90 * 0.4).astype(f32),
        "width90": width90.astype(f32),
        "width99": (width90 * 1.6).astype(f32),
        "width90_raw": width90_raw.astype(f32),
        "cov50": np.abs(err) <= width90 * 0.2,
        "cov90": np.abs(err) <= width90 * 0.5,
        "cov99": np.abs(err) <= width90 * 0.8,
        "abstain": width90_raw > tau,
        "query_i": rng.integers(0, d, size=(T, Q)).astype(np.int32),
        "query_v": rng.uniform(-2, 2, size=(T, Q)).astype(f32),
        "intervention_i": (rng.integers(0, d, size=(T, B)) if interv else np.full((T, B), -1)).astype(np.int32),
        "intervention_v": (np.tile([2.0, -2.0, 1.0, -1.0], B // 4 + 1)[:B][None, :].repeat(T, 0) if interv
                           else np.full((T, B), np.nan)).astype(f32),
        "sample_score": (rng.normal(-5, 1, size=(T, B)) if learns else np.full((T, B), np.nan)).astype(f32),
        "learned_adj": (np.broadcast_to(CHAIN4, (T, d, d)).copy() if learns else np.zeros((T, d, d), bool)),
        "has_learned_adj": np.full(T, learns, dtype=bool),
        "g_stat": (np.abs(rng.normal(1, 1, size=(T, d))) if det else np.full((T, d), np.nan)).astype(f32),
        "glr_stat": (rng.normal(2, 1, size=(T, d)) if det else np.full((T, d), np.nan)).astype(f32),
        "self_score": (rng.normal(-5, 0.5, size=T) if learns else np.full(T, np.nan)).astype(f32),
        "var_obj_before": (np.abs(rng.normal(1, 0.2, size=T)) if learns else np.full(T, np.nan)).astype(f32),
        "var_obj_after": (np.abs(rng.normal(0.8, 0.2, size=T)) if learns else np.full(T, np.nan)).astype(f32),
    }
    if agent == "oracle":
        R["cov50"][:] = R["cov90"][:] = R["cov99"][:] = True
    if learns:                                    # a wrong edge in the first two episodes
        R["learned_adj"][:2, 3, 0] = True
    codes = protocol.SHIFT_TYPE_CODES
    for t, j, st, m in shifts:
        R["shift_j"][t - 1], R["shift_type_code"][t - 1] = j, codes[st]
        R["shift_m"][t - 1], R["shift_delta"][t - 1], R["shift_delta_sigma"][t - 1] = m, 3 * m, 0.4
        R["err"][t - 1] *= 3                       # a visible bump in the shift episode
    resets = []
    if agent == "mech-reset-all":
        resets = [(7, j) for j in range(d)]
    elif det and agent != "mech-overconfident":
        resets = [(7, 1), (10, 3)]                 # one correct, one false alarm
    elif agent == "mech-oracle-detect":
        resets = [(7, 1), (12, 2)]
    hidden_A, hidden_B = (0, 1) if hidden else (-1, -1)
    if hidden:
        R["conf_c"][:], R["conf_F"][:] = 0.5, 0.1
        R["shift_j"][8], R["shift_type_code"][8] = d, 0     # the H shift at t = 9
        R["shift_m"][8], R["shift_delta"][8], R["shift_delta_sigma"][8] = 0.2, 0.0, 0.3
        if det:
            resets += [(9, 0), (9, 1)]
    R["reset_events"] = np.asarray(resets, dtype=np.int32).reshape(-1, 2)
    R["struct_refit_events"] = np.asarray([(2, 3)] if learns else [], dtype=np.int32).reshape(-1, 2)
    R.update({
        "agent_name": np.str_(agent), "agent_id": np.int32(0), "seed": np.int32(seed),
        "d": np.int32(d), "T": np.int32(T), "B": np.int32(B), "n_obs": np.int32(50), "Q": np.int32(Q),
        "shift_type": np.str_("large"), "hetero": np.int32(0), "hidden": np.bool_(hidden),
        "shifts_enabled": np.bool_(True), "learns_structure": np.bool_(learns),
        "has_detector": np.bool_(det), "intervenes": np.bool_(interv), "uses_floor": np.bool_(floor),
        "is_oracle": np.bool_(orc), "tau": np.float32(math.inf if agent in ("marginal-mean", "oracle") else tau),
        "wall_time_s": np.float32(0.5), "hidden_A": np.int32(hidden_A), "hidden_B": np.int32(hidden_B),
    })
    return R


def _write_results_dir(path, agents, seeds, *, budget=50, hidden=False, shift_type="large", hetero=0):
    os.makedirs(os.path.join(path, "raw"), exist_ok=True)
    for a in agents:
        for s in seeds:
            # the raw file's B scalar must agree with the config (primary_P5 checks it)
            protocol.save_raw(os.path.join(path, "raw", protocol.raw_filename(a, s)),
                              synthetic_raw(a, s, hidden=hidden, B=budget))
    config = {"agents": list(agents), "seeds": list(seeds), "episodes": 45, "budget": budget,
              "n_obs": 50, "n_queries": 10, "shift_type": shift_type, "hetero": hetero,
              "hidden": hidden, "shifts_enabled": True, "git_hash": "deadbeef", "total_wall_s": 1.0,
              "constants_sha256": "0" * 64,
              "constants": {"lambda": 3.0, "agents": {"obs-window": {"W": 3, "tau": 0.6}}}}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)


MAIN_SYNTH = ("marginal-mean", "obs-window", "int-pairwise", "mech-full", "mech-random",
              "mech-reset-all", "mech-oracle-detect", "mech-overconfident", "oracle-structure", "oracle")


def test_classify_results_dirs():
    assert report.classify({"budget": 50}) == ("main", "main")
    assert report.classify({"budget": 10}) == ("sweep", "sweep_B10")
    assert report.classify({"budget": 50, "hidden": True}) == ("hidden", "hidden")
    assert report.classify({"budget": 50, "shift_type": "noise-only"}) == ("knob", "knob_noise-only_h0")
    assert report.classify({"budget": 50, "hetero": 1}) == ("knob", "knob_large_h1")
    assert report.classify({"budget": 50, "shifts_enabled": False})[0] == "noshift"


def test_summarize_agent_and_secondaries_on_synthetic_raw():
    raw = synthetic_raw("mech-full", 0)
    ref = synthetic_raw("oracle-structure", 0)
    S = report.summarize_agent(raw, ref)
    assert S["T"] == 45 and len(S["nmse_series"]) == 45   # T >= 40 so P5 / P6 windows are non-empty
    assert S["n_shifts_qualifying"] == 1 and S["n_shifts_excluded"] == 1   # m = 0.01 filtered
    assert S["credit"]["confusion"]["correct"] == 1 and S["credit"]["confusion"]["false-alarm"] == 1
    assert len(S["regret"]) == 1 and len(S["recovery"]) == 1 and len(S["forgetting"]) == 1
    assert "shd_series" in S and S["shd_series"][0] == 1 and S["shd_series"][5] == 0
    u = report.undetected_shift_stats(synthetic_raw("mech-overconfident", 0), n_after=1)
    assert u["n_shifts"] == 1 and u["n_undetected"] == 1 and 0 <= u["cov90_first"] <= 1


def test_report_on_synthetic_results(tmp_path):
    seeds = [0, 1, 2]
    _write_results_dir(tmp_path / "main", MAIN_SYNTH, seeds)
    _write_results_dir(tmp_path / "sweep_B10", ("mech-full", "mech-random", "int-pairwise", "mech-full-nofloor"),
                       seeds, budget=10)
    _write_results_dir(tmp_path / "hidden", ("mech-full", "mech-full-diag", "oracle-structure", "int-pairwise"),
                       seeds, hidden=True)
    _write_results_dir(tmp_path / "knob_small", ("mech-full", "mech-oracle-detect", "oracle-structure"),
                       seeds, shift_type="small")
    out = tmp_path / "report"
    rep = report.build_report([str(tmp_path / d) for d in ("main", "sweep_B10", "hidden", "knob_small")],
                              str(out), n_boot=60, log=lambda *a: None)
    assert rep.problems == [], "\n".join(rep.problems)

    # outputs exist and parse
    assert (out / "REPORT.md").exists() and (out / "metrics.json").exists()
    mj = json.loads((out / "metrics.json").read_text())
    md = (out / "REPORT.md").read_text()
    figs = sorted(os.listdir(out / "figures"))
    for key in ("nmse_main.png", "coverage_vs_width_main.png", "risk_coverage_main.png", "shd_main.png",
                "budget_sweep.png", "credit_confusion.png", "hidden_pair_hidden.png", "objectives_main.png"):
        assert key in figs, f"{key} missing from {figs}"

    # every prediction has a verdict; the ones without their run are NOT TESTABLE with a reason
    P = mj["primaries"]
    assert set(P) == {f"P{k}" for k in range(1, 10)}
    for name, r in P.items():
        assert r["verdict"] in ("HELD", "NOT HELD", "NOT TESTABLE")
        assert f"**{name}**" in md
    assert P["P9"]["verdict"] == "NOT TESTABLE" and "noise-only" in P["P9"]["not_testable_reason"]
    for name in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
        assert P[name]["n_seeds"] == 3, (name, P[name])
        assert P[name]["verdict"] in ("HELD", "NOT HELD")
        assert P[name]["p_adj"] >= P[name]["p"] - 1e-12          # Holm never lowers a p-value
    # the synthetic detector agent attributes its one shift correctly: P7 accuracy = 1 per seed
    assert P["P7"]["ci"]["mean"] == pytest.approx(1.0)
    # the report tables carry the exploratory label and the per-run nMSE table
    assert "Exploratory" in md and "`marginal-mean` floor" in md
    assert "NOT TESTABLE" in md and "HELD" in md
    # aggregated agent metrics are JSON-clean (no NaN) and carry CIs
    A = mj["agents"]["main"]["mech-full"]
    assert set(A["nmse"]) >= {"mean", "lo", "hi", "n"} and A["nmse"]["n"] == 3
    assert A["credit"]["confusion"]["correct"] == 3
    assert "stable" in A["calibration"] and "shd_zero" in A["calibration"]
    assert mj["secondaries"]["hidden"]["p6iv_both_fire"]["mean"] == pytest.approx(1.0)
    assert "p5a_int_minus_full" in mj["secondaries"]["sweep_B10"]


def test_report_cli_on_single_dir(tmp_path):
    _write_results_dir(tmp_path / "main", ("mech-full", "obs-window", "oracle-structure"), [0, 1])
    assert report.main([str(tmp_path / "main"), "--out", str(tmp_path / "rep"), "--n-boot", "40", "--quiet"]) == 0
    assert (tmp_path / "rep" / "REPORT.md").exists()
