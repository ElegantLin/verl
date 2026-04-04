import logging

from verl.utils.reward_score import math_verify as math_verify_module


def test_compute_score_silences_math_verify_warning(monkeypatch, caplog):
    warning_message = "We did not manage to extract a prediction in the correct format."

    def fake_math_metric(*args, **kwargs):
        def fake_verify(golds, predictions):
            logging.getLogger("math_verify.metric").warning(
                "%s Gold: %s, Pred: %s",
                warning_message,
                golds,
                predictions,
            )
            return 0.0, None

        return fake_verify

    monkeypatch.setattr(math_verify_module, "math_metric", fake_math_metric)
    caplog.set_level(logging.WARNING, logger="math_verify.metric")

    score = math_verify_module.compute_score("beneficial의 정类似于", "42")

    assert score == 0.0
    assert not [
        record
        for record in caplog.records
        if record.name == "math_verify.metric" and warning_message in record.message
    ]
