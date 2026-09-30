"""The default prompts must be identical to those used in the experiments."""

from hydra_bench import prompts


def notebook_prompts(genre, situ, ori, q, ans):
    # Reconstructed verbatim from the experiment notebooks.
    p1 = 'Provide an example of a realistic and likely scenario that slightly deviates from the usual in the context of '
    p1 += genre + '. '
    p1 += 'For instance, you might find the road wet when leaving your home in the afternoon, '
    p1 += 'even though the weather was clear in the morning, '
    p1 += 'or you might find a café closed during its normal business hours. '
    p1 += 'Make sure the cause of the situation cannot be easily determined through simple observation, '
    p1 += 'such as by looking around. Also, avoid subjective examples involving personal feelings or thoughts; '
    p1 += 'choose an objective and observable situation instead. Describe the situation in a few sentences, '
    p1 += 'including some background details.'

    p2 = 'Provide two possible origins of the following situation. '
    p2 += 'Do not assume that the described facts in the situation are incorrect, such as due to a misunderstanding. '
    p2 += 'Keep both origins realistically plausible and mutually exclusive in principle. '
    p2 += 'Avoid any origin that, if true, would clearly reveal itself as the cause. '
    p2 += 'For example, if the situation is "the train did not arrive at the usual time without any announcement," '
    p2 += 'the origin should not be "the schedule was moved forward only for today," since such a change would normally be announced.'
    p2 += '''Describe the result in one sentence each, using the following format:
origin1:
origin2:

'''
    p2 += 'situation: '
    p2 += situ

    p3 = 'Create a question that asks for the origin of a situation by adding to, modifying, or reorganizing the given description. '
    p3 += 'Ensure that origin2 is the correct answer and that origin1 is incorrect. '
    p3 += 'At the same time, include a misleading detail that makes origin1 appear plausible, so the question is not too easy. '
    p3 += 'Present the situation as a single paragraph, without including any instructions or options such as '
    p3 += '''"Answer the origin of this situation."

situation: '''
    p3 += situ
    p3 += '''

'''
    p3 += ori

    p6 = 'Provide the single most plausible hypothesis for why the following situation occurs, along with a method of verification. '
    p6 += '''Describe each in a single sentence, using the following format:
hypothesis:
verification:

'''
    p6 += q

    pv = '''Evaluate the hypothesis and the method used to verify it based on the following criteria:
(a) The hypothesis reasonably explains why the situation occurs.
(b) The hypothesis is realistically plausible.
(c) The hypothesis is not almost always correct as a general rule, nor is it almost always correct when the situation occurs.
(d) The result of the verification method significantly increases the likelihood of the hypothesis being correct.
(e) It is realistically feasible to obtain the result of the verification method.
(f) The verification method does not yield nearly the same result every time the situation occurs.
Describe each result using only "OK" or "NG," following the format below:
a: OK or NG
b: OK or NG
...
f: OK or NG

situation: ''' + q + '''
''' + ans
    return p1, p2, p3, p6, pv


def test_prompts_identical_to_notebooks():
    genre, situ, ori, q, ans = "music", "SITU {x}", "origin1: A\norigin2: B", "Q text", "hypothesis: H\nverification: V"
    t = prompts.load_templates()
    ours = (
        prompts.render(t["situation"], context=genre),
        prompts.render(t["origins"], original_situation=situ),
        prompts.render(t["modify"], original_situation=situ, origins=ori),
        prompts.render(t["target"], situation=q),
        prompts.render(t["evaluation"], situation=q, answer=ans),
    )
    for mine, ref in zip(ours, notebook_prompts(genre, situ, ori, q, ans)):
        assert mine == ref


def test_template_override(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("situation: {situation}", encoding="utf-8")
    t = prompts.load_templates({"target": str(f)})
    assert prompts.render(t["target"], situation="X") == "situation: X"
