from src.critic import Critic


def test_critic_classification_ok():
    critic = Critic()
    result = critic.review(
        "This is a sufficiently long answer for validation.",
        task_type="classification"
    )
    assert result == "Answer OK"


def test_critic_classification_short():
    critic = Critic()
    result = critic.review("short", task_type="classification")
    assert result == "Answer too short"


def test_critic_regression():
    critic = Critic()
    result = critic.review(
        "This is a reasonably long answer for confidence scoring.",
        task_type="regression"
    )

    assert "confidence_score" in result
    assert 0.0 <= result["confidence_score"] <= 1.0


def test_critic_invalid_task():
    critic = Critic()
    try:
        critic.review("answer", task_type="invalid")
        assert False, "Expected ValueError for invalid task_type"
    except ValueError as e:
        assert "Unsupported task_type" in str(e)