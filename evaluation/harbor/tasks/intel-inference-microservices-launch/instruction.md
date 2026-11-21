# Serve the requested model

`/app/request.txt` is what to serve. `/app/catalog.json` lists the validated images for
this Intel Xeon host, plus the base image used when a model is not listed.

Write `/app/launch.sh`: one `docker run` that serves that model so a client can call an
OpenAI-compatible API on port 8000.

Only the command is graded. It is parsed, not executed. Do not start a container.
