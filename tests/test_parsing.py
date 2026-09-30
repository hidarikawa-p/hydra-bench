from hydra_bench.evaluate import parse_hypothesis, parse_judgment
from hydra_bench.generate import clean_situation, parse_origins


def test_parse_judgment_all_ok():
    r = parse_judgment("a: OK\nb: OK\nc: OK\nd: OK\ne: OK\nf: OK")
    assert r["final"] == "OK" and r["abcok"] == "OK"


def test_parse_judgment_ng_and_missing():
    r = parse_judgment("a: OK\nb: NG\nc: OK\nd: OK\nf：OK")  # e missing, full-width colon for f
    assert r["b"] == "NG" and r["e"] == "" and r["f"] == "OK"
    assert r["final"] == "NG" and r["abcok"] == "NG"


def test_parse_hypothesis_and_origins():
    h, v = parse_hypothesis("**hypothesis:** H text\n**verification:** V text")
    assert (h, v) == ("H text", "V text")
    o1, o2 = parse_origins("origin1: first\norigin2: second")
    assert (o1, o2) == ("first", "second")


def test_clean_situation():
    assert clean_situation("plain paragraph") == "plain paragraph"
    raw = "**The Title**\n\nThe long situation paragraph text.\n\nWhat is the origin?\n\nA) x"
    assert clean_situation(raw) == "The long situation paragraph text."
    raw = "\n\n**Situation:**  \nThe long situation paragraph text.  \n\n**Origin1 (incorrect):** short"
    assert clean_situation(raw) == "The long situation paragraph text.  "
