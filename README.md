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
| `postgres/init/` | legt die read-only Rolle für Grafana an (nur beim ersten Start) |
| `prometheus/prometheus.yml` | Scrape nur von agentgateway `:15020` (vLLM und Codebox werden nicht abgefragt) |
| `grafana/provisioning/` | Datasources (Prometheus, Postgres) und Dashboard-Provider |
| `grafana/dashboards/overview.json` | **Overview** (Startseite): Status und Kennzahlen von LLMs und MCP |
| `grafana/dashboards/llm-overview.json` | **LLM Overview**: alle Modelle, Backend-Status, Nutzer, Requests |
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
  Tokens, Latenz, Fehler nach Ursache, Tokens pro Request), Nutzer und die letzten 100 Requests.
- **MCP Overview:** Endpunkt-Status, Tool-Calls je Server und Tool, Requests nach Server, Methode
  und Typ, Latenz und Fehler.

Alle Daten kommen aus agentgateway: Metriken (`agentgateway_*` über Prometheus) und das
Request-Log, das agentgateway in Postgres schreibt (`request_logs`). vLLM und Codebox werden nicht
abgefragt.

- **Backend-Status:** Ergibt sich aus dem Anteil der Requests ohne 5xx in den letzten 5 Minuten.
  OK ab 99 %, DEGRADED ab 90 %, darunter FAILING. NO TRAFFIC heißt, es gab in dieser Zeit keine
  Requests. Ein gestopptes LLM-Backend erscheint als `503 UpstreamFailure`, ein ausgefallener
  MCP-Server als `500 MCP`.
- **MCP:** agentgateway meldet MCP-Fehler nur für den gesamten Endpunkt `/mcp`, nicht pro Server.
  Ins Postgres-Log schreibt agentgateway nur LLM-Requests, deshalb gibt es für MCP keine
  Auswertung pro User.
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
