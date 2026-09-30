"""End-to-end run with the offline dummy provider."""

import pandas as pd

from hydra_bench.evaluate import run_evaluate
from hydra_bench.generate import run_generate


def _cfgs(tmp_path):
    gen = {
        "contexts": "ddc45",
        "output_dir": str(tmp_path / "sit"),
        "generators": [
            {"label": "Gen A", "provider": "dummy", "model": "d"},
            {"label": "Gen B", "provider": "dummy", "model": "d"},
        ],
    }
    ev = {
        "situations_dir": str(tmp_path / "sit"),
        "output_dir": str(tmp_path / "res"),
        "concurrency": 3,
        "targets": [{"label": "Tgt 1", "provider": "dummy", "model": "d"},
                    {"label": "Tgt 2", "provider": "dummy", "model": "d"}],
        "evaluator": {"label": "Judge", "provider": "dummy", "model": "d", "params": {"verdict": "NG"}},
    }
    return gen, ev


def test_end_to_end_and_resume(tmp_path):
    gen, ev = _cfgs(tmp_path)
    files = run_generate(gen)
    assert len(files) == 2
    sit = pd.read_csv(files[0], encoding="utf-8-sig")
    assert len(sit) == 45 and sit["category"].str.len().gt(0).all()
    assert sit["origin1"].str.len().gt(0).all()

    eval_dir = run_evaluate(ev)
    res = pd.read_csv(eval_dir / "tgt-1.csv", encoding="utf-8-sig", keep_default_na=False)
    assert len(res) == 90
    assert set(res["final"]) <= {"OK", "NG"}
    summary = pd.read_csv(eval_dir / "summary_by_target.csv", encoding="utf-8-sig")
    assert summary.iloc[-1]["target_model"] == "Average" and summary.iloc[-1]["n"] == 180

    # rerun: everything is cached, outputs unchanged
    run_generate(gen)
    run_evaluate(ev)
    res2 = pd.read_csv(eval_dir / "tgt-1.csv", encoding="utf-8-sig", keep_default_na=False)
    pd.testing.assert_frame_equal(res, res2)


def test_rejudge_with_other_evaluator(tmp_path):
    gen, ev = _cfgs(tmp_path)
    run_generate(gen)
    run_evaluate(ev)
    ev2 = dict(ev, evaluator={"label": "Judge 2", "provider": "dummy", "model": "d"})
    eval_dir = run_evaluate(ev2, skip_target=True)
    assert eval_dir.name == "judge-2"
    res = pd.read_csv(eval_dir / "tgt-2.csv", encoding="utf-8-sig", keep_default_na=False)
    assert (res["final"] == "OK").all()
