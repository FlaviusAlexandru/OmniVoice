# Copyright    2026  (fork addition)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Keep a permanent local copy of every Hugging Face model OmniVoice uses.

``resolve_model_path("k2-fsa/OmniVoice")`` returns a local folder, fetching the
model at most once:

1. A local directory path is returned as-is.
2. ``<models dir>/<org>--<name>/`` is used if it holds a complete copy (marked
   by ``.omnivoice_complete``). No network access at all.
3. Otherwise the model is copied there from the Hugging Face cache if it is
   already cached, or downloaded straight into it, then marked complete.

The models dir is ``$OMNIVOICE_MODELS_DIR`` if set, else ``models/`` in the
repository checkout (editable install). For a plain pip install with no
checkout and no env var, behaviour falls back to the normal Hugging Face cache.
"""

import os
import shutil
from pathlib import Path
from typing import Callable, Optional

MARKER = ".omnivoice_complete"


def models_dir() -> Optional[str]:
    env = os.environ.get("OMNIVOICE_MODELS_DIR")
    if env:
        return env
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").is_file():
        return str(repo_root / "models")
    return None


def local_copy_path(repo_id: str, root: Optional[str] = None) -> Optional[str]:
    root = root if root is not None else models_dir()
    if root is None:
        return None
    return os.path.join(root, repo_id.replace("/", "--"))


def resolve_model_path(
    name_or_path: str,
    snapshot_download: Optional[Callable[..., str]] = None,
    root: Optional[str] = None,
) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path

    if snapshot_download is None:
        from huggingface_hub import snapshot_download

    local = local_copy_path(name_or_path, root)
    if local is None:
        return snapshot_download(name_or_path)

    if os.path.isfile(os.path.join(local, MARKER)):
        return local

    os.makedirs(local, exist_ok=True)
    try:
        # Already in the Hugging Face cache? Copy it over instead of downloading again.
        cached = snapshot_download(name_or_path, local_files_only=True)
        shutil.copytree(cached, local, dirs_exist_ok=True)
    except Exception:
        snapshot_download(name_or_path, local_dir=local)

    with open(os.path.join(local, MARKER), "w", encoding="utf-8") as f:
        f.write(name_or_path + "\n")
    return local
