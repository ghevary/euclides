import pytest

import synth


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    """A synthetic data bundle, its config and the true labels."""
    root = tmp_path_factory.mktemp("bundle")
    cfg = synth.small_config()
    truth = synth.write_bundle(root, cfg)
    return root, cfg, truth
