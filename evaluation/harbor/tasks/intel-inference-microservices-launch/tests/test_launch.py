"""Parse the docker run command; each assertion is one way this container fails to serve."""
import json
import re
import shlex
from pathlib import Path

import pytest

SCRIPT = Path("/app/launch.sh")
CATALOG = json.loads(Path("/app/catalog.json").read_text())
REQUEST = Path("/app/request.txt").read_text()
EXPECTED = next(row for row in CATALOG["models"] if row["model_id"] in REQUEST)


@pytest.fixture(scope="module")
def argv():
    assert SCRIPT.is_file(), f"{SCRIPT} was not written"
    text = SCRIPT.read_text().replace("\\\n", " ")
    runs = [line for line in text.splitlines() if re.search(r"\bdocker\s+run\b", line)]
    assert len(runs) == 1, "exactly one docker run command"
    return shlex.split(runs[0][runs[0].index("docker"):], comments=True)


def values(argv, name):
    found = []
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            found.append(argv[i + 1])
        elif arg.startswith(name + "="):
            found.append(arg.split("=", 1)[1])
    return found


def env(argv):
    pairs = {}
    for raw in values(argv, "-e") + values(argv, "--env"):
        key, _, value = raw.partition("=")
        pairs[key] = value
    return pairs


def test_catalog_image(argv):
    assert EXPECTED["image"] in argv, "a listed model uses its catalog image, not the base image"
    assert CATALOG["base_image"] not in argv


def test_socket_and_shared_memory(argv):
    assert "SYS_NICE" in values(argv, "--cap-add"), "without SYS_NICE a multi-socket host cannot bind memory"
    shm = values(argv, "--shm-size")
    assert shm, "Docker's default shared memory is too small for the engine to start"
    assert shm[0].lower() not in {"64m", "64mb", "67108864"}


def test_published_port(argv):
    published = values(argv, "-p") + values(argv, "--publish")
    assert any(item.endswith("8000:8000") or item == "8000:8000" for item in published)
    port = env(argv).get("INFERENCE_PORT")
    assert port in (None, "8000")


def test_engine_left_to_the_image(argv):
    assert "INFERENCE_ENGINE" not in env(argv)
    assert "INFERENCE_MODEL_ID" not in env(argv)
    image_at = argv.index(EXPECTED["image"])
    command = argv[image_at + 1:]
    assert not any(token in {"dry-run", "list-profiles", "detect-hardware", "download-to-cache"} for token in command)


def test_not_a_gpu_launch(argv):
    assert "/dev/dri" not in " ".join(argv)
    assert "--gpus" not in argv
