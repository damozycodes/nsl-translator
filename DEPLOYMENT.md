# Three demonstration options

## 1. Full app on localhost (works without internet once built)

Start Docker Desktop and run from the project directory:

```sh
docker compose up -d nsl
```

Open http://localhost:8502. This mounts your full local dataset read-only.
Keep Docker running. No Render account or public link is required.

## 2. Temporary public link to the full local app

```sh
docker compose --profile share up -d
docker compose logs tunnel
```

Find the HTTPS URL ending in trycloudflare.com in the logs and share it with your
lecturer. Your laptop must stay awake, connected to the internet, and running
Docker. The URL is public and temporary; it may change when the tunnel restarts.
Quick Tunnels provide no uptime guarantee. Keep localhost as your defence backup.
Stop public sharing without stopping the local app:

```sh
docker compose stop tunnel
```

## 3. Render Free with the bundled demonstration dataset

The Docker image includes `demo-data/`: 60 selected vocabulary labels, one clip
and landmark sequence per label, and 20 recorded alphabet letters. A–Z illustrated
cards remain in `assets/`. This is a smaller vocabulary than the full local app.
Unavailable words use the existing matching/fingerspelling behavior. Use Browse
vocabulary to choose examples, including HELLO, WATER, SCHOOL, and THANK YOU.
Try short inputs on the free instance, especially for stickman rendering.

No persistent disk, SSH upload, external data download, or laptop is required.
The data is part of the image and is restored on every restart. Generated output
videos are temporary. A free service may sleep after inactivity and take time to
wake up; open it before your demonstration. Capacity is limited and concurrent
renders can exceed available memory.

For your already deployed service, commit and push these changes:

```sh
git add .
git diff --cached --stat
git commit -m "Add free Render demo and local sharing"
git push
```

Make sure `demo-data/`, `requirements-runtime.txt`, the Dockerfile, and Compose
configuration appear in the commit. Keep the original `dataset/` ignored.
In Render, open the existing service and use **Manual Deploy > Deploy latest
commit** if auto-deploy does not trigger. Retain the Free instance and leave Docker
Command blank. The bundled data fixes the missing-landmarks error.

For a new deployment, use **New > Blueprint** and connect the repository.
`render.yaml` now requests a free Docker web service with no disk. Do not create a
second service if the existing one already works.

## Updating the demonstration selection

Edit WORDS in scripts/prepare_demo.py and run:

```sh
python3 scripts/prepare_demo.py
```

The script copies from the full dataset without modifying it. It does not delete
previously generated files; review the demo directory if removing a selection.
Manifest entries retain original provenance and review status. The subset is a
project demonstration, not a new independently validated dataset.

## Build and validation

```sh
docker compose config --quiet
docker compose up --build -d nsl
```

The Docker image installs web-only requirements-runtime.txt and the spaCy model.
The original requirements.txt retains training dependencies for local research.
Local Compose targets linux/amd64 to match Render; Apple Silicon uses emulation.

Before presenting, test original playback, stickman, illustrated fingerspelling,
and downloading a video. Health checks only confirm that Streamlit is running.

References:
- https://render.com/docs/free
- https://render.com/docs/blueprint-spec
- https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/
