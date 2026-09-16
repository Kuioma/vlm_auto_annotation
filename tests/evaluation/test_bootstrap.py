from auto_annotation.evaluation.bootstrap import episode_bootstrap, quantile
from auto_annotation.evaluation.models import EpisodeEvaluation


def episode(index, score):
    return EpisodeEvaluation(
        episode_index=index,
        video_id=str(index),
        status="evaluated",
        provenance_status="verified",
        boundaries={},
        segments={},
        mean_tiou=score,
    )


def test_linear_interpolation_quantile():
    assert quantile([0, 10], 25) == 2.5


def test_episode_bootstrap_is_deterministic():
    episodes = [episode(1, 0.2), episode(2, 0.8)]
    statistic = lambda sample: sum(item.mean_tiou for item in sample) / len(sample)
    first = episode_bootstrap(episodes, statistic, confidence_level=0.95, resamples=100, seed=42)
    second = episode_bootstrap(episodes, statistic, confidence_level=0.95, resamples=100, seed=42)
    assert first == second
    assert first.valid_resamples == 100

