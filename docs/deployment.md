# Deploying on a Linux server

Tested on Ubuntu Server 24.04 in a homelab VM with Docker and the Compose plugin.

## Requirements

| Resource | Minimum | Why |
|---|---|---|
| Disk | ~15 GB free | App image with CPU PyTorch (~3 GB), Ollama image, 2 GB model, Postgres data |
| RAM | 6–8 GB | Postgres, embedding + reranker models, Llama 3.2 3B |
| CPU | 4+ cores | Everything runs on CPU; more cores mean faster answers |

## First deploy

```bash
# on your PC
scp -P <ssh-port> rag-manuals.zip <user>@<server>:~
ssh -p <ssh-port> <user>@<server>

# on the server
unzip rag-manuals.zip && cd rag-manuals
./deploy.sh        # creates .env from .env.example, then stops so you can review it
nano .env          # optional: switch LLM, port, bind address
./deploy.sh        # builds, starts db + app + ollama, pulls the model, ingests data/corpus
```

Copy the manuals into `data/corpus/` (e.g. `scp -P <ssh-port> -r manuals <user>@<server>:~` and move the PDFs), then run:

```bash
docker compose exec app python -m rag ingest data/corpus
```

## Opening the UI

The app listens on `127.0.0.1` only by default. Reach it through an SSH tunnel from your PC:

```bash
ssh -p <ssh-port> -L 8000:localhost:8000 <user>@<server>
# then open http://localhost:8000
```

To reach it from your LAN or a VPN such as Tailscale, set `APP_BIND=0.0.0.0` in `.env`. The app has no login: **never port-forward it on your router.**

## Settings you may need

| `.env` setting | Default | Change it when |
|---|---|---|
| `APP_PORT` | 8000 | Port 8000 is taken (e.g. by Portainer) |
| `APP_BIND` | 127.0.0.1 | You want LAN/VPN access |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | Ollama, `llama3.2:3b` | Using Gemini/Groq free tiers or a bigger local model |
| `EMBEDDING_*` | local `bge-small` | Using OpenAI embeddings (needs `python -m rag reset` and a re-ingest) |
| `RERANKER` | local | `none` to skip reranking (faster, less accurate) |

## Updating

```bash
cd ~/rag-manuals
# replace changed files (or git pull), then
docker compose up -d --build app
docker image prune -f
```

The Dockerfile installs dependencies in their own cached layer, so code-only changes rebuild in seconds.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Bind for 0.0.0.0:8000 failed: port is already allocated` | Another service uses the port. Set `APP_PORT` in `.env` |
| `no space left on device` during build | Run `docker builder prune -af` and `docker image prune -f`. Check `df -h /`: on Ubuntu with LVM, the filesystem may be smaller than its volume (`sudo vgs`, `lsblk`); grow it with `sudo resize2fs /dev/mapper/ubuntu--vg-ubuntu--lv` |
| App shows `(unhealthy)` | `docker compose logs app --tail 30` |
| `Database was built with embeddings ...` | You changed the embedding model. Run `docker compose exec app python -m rag reset`, then ingest again |
| `has no extractable text` while ingesting | The PDF is a scan; it needs OCR first |
| Answers take over a minute | Expected for a 3B model on CPU; use a GPU, a hosted model, or `TOP_K=3` |

## Security checklist

- SSH: key-only login (`PasswordAuthentication no`), no default ports exposed without need
- Install OS updates and reboot when `*** System restart required ***` shows
- Keep API keys only in `.env` (git-ignored, `chmod 600`)
- Don't run this on the same VM as unrelated public services
