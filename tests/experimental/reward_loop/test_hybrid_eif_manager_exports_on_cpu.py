import importlib


def test_hybrid_eif_offline_manager_is_not_exported():
    module = importlib.import_module("verl.experimental.reward_loop.reward_manager")

    assert not hasattr(module, "HybridEIFRewardManager")
    assert "HybridEIFRewardManager" not in module.__all__
    assert hasattr(module, "HybridEIFOnlineRewardManager")
    assert "HybridEIFOnlineRewardManager" in module.__all__
