# agentgateway-stack

Qwen3.5-0.8B auf vLLM hinter [agentgateway](https://agentgateway.dev), Codebox als MCP-Server,
Open WebUI als Frontend, Postgres für agentgateway, Prometheus und Grafana fürs Monitoring.

```
Open WebUI ──LLM──▶ agentgateway :4000 ──▶ vLLM (Qwen/Qwen3.5-0.8B)
     │                    │
     └──MCP──▶ agentgateway :3000/mcp ──▶ codebox (OWUI-Codebox-MCP, microVMs)
                          │
                          ├─▶ Postgres   (Request-Logs, UI-Config-Overlay)
                          └─▶ Prometheus ─▶ Grafana
```

## Start

```bash
cp .env.example .env    # Secrets eintragen, z. B. mit: openssl rand -hex 32
docker compose up -d
```

Voraussetzungen: NVIDIA-GPU mit Container-Toolkit, `/dev/kvm` für Codebox, das Modell im
Hugging-Face-Cache (`HF_CACHE_DIR`). Der erste vLLM-Start dauert wegen `torch.compile` ein paar
Minuten, danach liegt der Cache im Volume `vllm-cache`.

## Ports

| Dienst | URL | Zugriff |
|---|---|---|
| Open WebUI | http://localhost:8081 | Admin aus `OWUI_ADMIN_EMAIL` / `OWUI_ADMIN_PASSWORD` |
| agentgateway LLM | http://localhost:4000/v1 | `Authorization: Bearer $AGW_LLM_API_KEY` |
| agentgateway MCP | http://localhost:3000/mcp | OpenWebUI-JWT (prüft Codebox) |
| agentgateway Admin-UI | http://127.0.0.1:15000/ui | nur localhost |
| Grafana | http://localhost:3001 | `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD` |
| Prometheus | http://127.0.0.1:9090 | nur localhost |
| Postgres | 127.0.0.1:5432 | nur localhost |

## Dateien

| Datei | Inhalt |
|---|---|
| `compose.yaml` | alle Dienste |
| `agentgateway/config.yaml` | agentgateway-Config, read-only gemountet; `${VAR}` kommen aus `.env` |
| `prometheus/prometheus.yml` | scrapt nur agentgateway (`:15020/metrics`, 15 s), wie im agentgateway-Helm-Chart |
| `grafana/provisioning/` | einzige Datasource Prometheus, dazu der Dashboard-Provider |
| `grafana/dashboards/overview.json` | **Overview** (Startseite): Status und Kennzahlen von LLMs und MCP |
| `grafana/dashboards/llm-overview.json` | **LLM Overview**: alle Modelle, Backend-Status, Vergleich je Modell |
| `grafana/dashboards/mcp-overview.json` | **MCP Overview**: alle MCP-Server und Tools |
| `grafana/dashboards/agentgateway.json` | **Gateway Internals**: Dashboard aus dem agentgateway-Repo (v1.7.0) |

## Dashboards

Alle liegen in Grafana im Ordner *agentgateway* und haben oben dieselbe Navigation:

```
Overview ──▶ LLM Overview      alle Modelle
         └─▶ MCP Overview      alle MCP-Server und Tools
             Gateway Internals agentgateway selbst (Routen, Runtime, Speicher)
```

- **Overview** ist die Startseite. Oben steht der Status von agentgateway, von jedem LLM-Backend
  und vom MCP-Endpunkt, darunter je ein Block *LLMs* und *MCP* mit den wichtigsten Zahlen. Ein
  Klick auf ein Panel öffnet die passende Übersicht mit demselben Zeitraum.
- **LLM Overview:** Backend-Status je Modell, Summen, Modelltabelle, Vergleich je Modell (Requests,
  Tokens, Latenz, Ausgabegeschwindigkeit, Fehler nach Ursache, Tokens pro Request, Gateway-Overhead).
- **MCP Overview:** Endpunkt-Status, Tool-Calls je Server und Tool, Requests nach Server, Methode
  und Typ, Latenz und Fehler.

Prometheus und Grafana hängen nur an agentgateway, wie es das agentgateway-Helm-Chart vorsieht:
Prometheus scrapt `agentgateway:15020/metrics` alle 15 s, Grafana liest nur Prometheus. vLLM,
Codebox und Postgres werden von Prometheus und Grafana nicht abgefragt.

- **Einzelne Requests** (User, Tokens, Dauer, Fehler) zeigt die agentgateway-Admin-UI unter
  http://127.0.0.1:15000/ui/llm/logs und `/ui/llm/analytics`. Sie liest das Request-Log, das
  agentgateway selbst in Postgres schreibt.
- **Genauigkeit:** Durchschnitte (Tokens pro Request, Ausgabegeschwindigkeit, Abstand zwischen
  Chunks) sind exakt (Summe ÷ Anzahl). p95-Werte schätzt Prometheus aus den Histogramm-Buckets
  von agentgateway.

- **Backend-Status:** Ergibt sich aus dem Anteil der Requests ohne 5xx in den letzten 5 Minuten.
  OK ab 99 %, DEGRADED ab 90 %, darunter FAILING. NO TRAFFIC heißt, es gab in dieser Zeit keine
  Requests. Ein gestopptes LLM-Backend erscheint als `503 UpstreamFailure`, ein ausgefallener
  MCP-Server als `500 MCP`.
- **MCP:** agentgateway meldet MCP-Fehler nur für den gesamten Endpunkt `/mcp`, nicht pro Server.
- **Pro User:** Die Standard-Metriken von agentgateway haben kein User-Label. Auswertungen pro User
  gibt es daher in der Admin-UI, nicht in Grafana.
- Neue Modelle und MCP-Server aus `config.yaml` erscheinen nach ihrem ersten Request von selbst.

## Hinweise

- **agentgateway-Storage `hybrid`:** `config.yaml` ist die Basis, Änderungen in der Admin-UI
  landen in Postgres (`agw_config_resources`). Request-Logs stehen in `request_logs`, der
  OWUI-User (Header `X-OpenWebUI-User-Email`) in `agentgateway_user`.
- **Open WebUI:** `TOOL_SERVER_CONNECTIONS`, `OPENAI_API_*` und der Admin-Account wirken nur beim
  ersten Start mit leerem Volume. Danach gelten die Einstellungen in der OWUI-Datenbank.
- **MCP-Auth:** OWUI schickt das Session-JWT des Users (`auth_type: session`), agentgateway reicht
  es durch, Codebox prüft es mit `WEBUI_SECRET_KEY`. Codebox lädt Ausgabedateien mit diesem Token
  über `http://openwebui:8080` in OWUI hoch.
- **Thinking:** Qwen3.5-0.8B antwortet standardmäßig ohne Thinking (`VLLM_ENABLE_THINKING=false`).
  Pro Request lässt es sich mit `chat_template_kwargs: {"enable_thinking": true}` einschalten.
- **Codebox-Sandbox:** `codebox-init` zieht `CODEBOX_SANDBOX_IMAGE` einmalig ins Volume.
  Ein Update geht mit
  `docker compose run --rm codebox-init uv run --no-sync msb pull --force <image>`.
