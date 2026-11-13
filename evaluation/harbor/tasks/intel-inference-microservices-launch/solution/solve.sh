#!/usr/bin/env bash
set -euo pipefail
# The oracle: the catalog image for the requested model, socket binding, and enough
# shared memory for the engine. The base image is the wrong choice when the catalog
# already lists the model.
python3 - <<'PY'
import json
from pathlib import Path

catalog = json.loads(Path("/app/catalog.json").read_text())
request = Path("/app/request.txt").read_text()
match = next(row for row in catalog["models"] if row["model_id"] in request)
Path("/app/launch.sh").write_text(
    "#!/usr/bin/env bash\n"
    "docker run --rm -p 8000:8000 \\\n"
    "  --cap-add SYS_NICE \\\n"
    "  --shm-size=2g \\\n"
    f"  {match['image']}\n"
)
PY
echo "launch script written to /app/launch.sh"
