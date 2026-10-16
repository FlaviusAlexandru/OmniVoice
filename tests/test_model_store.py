import os

from omnivoice.utils.model_store import MARKER, resolve_model_path


def _fake_hub(calls, cached=None):
    def snapshot_download(repo_id, local_files_only=False, local_dir=None):
        calls.append((repo_id, local_files_only, local_dir))
        if local_files_only:
            if cached is None:
                raise FileNotFoundError("not cached")
            return cached
        os.makedirs(local_dir, exist_ok=True)
        with open(os.path.join(local_dir, "model.safetensors"), "w") as f:
            f.write("weights")
        return local_dir

    return snapshot_download


def test_local_dir_is_returned_as_is(tmp_path):
    assert resolve_model_path(str(tmp_path), snapshot_download=_fake_hub([])) == str(tmp_path)


def test_downloads_once_then_reuses(tmp_path):
    calls = []
    hub = _fake_hub(calls)
    first = resolve_model_path("k2-fsa/OmniVoice", snapshot_download=hub, root=str(tmp_path))
    assert os.path.isfile(os.path.join(first, MARKER))
    n = len(calls)
    second = resolve_model_path("k2-fsa/OmniVoice", snapshot_download=hub, root=str(tmp_path))
    assert second == first
    assert len(calls) == n  # no hub access the second time


def test_copies_from_the_hf_cache_instead_of_downloading(tmp_path):
    cache = tmp_path / "hf_cache_snapshot"
    cache.mkdir()
    (cache / "config.json").write_text("{}")
    calls = []
    path = resolve_model_path("k2-fsa/OmniVoice", snapshot_download=_fake_hub(calls, str(cache)), root=str(tmp_path / "models"))
    assert os.path.isfile(os.path.join(path, "config.json"))
    assert all(local_files_only for _, local_files_only, _ in calls)
