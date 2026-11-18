# Load-Balancing für 2× Qwen3.8-Flash-Next

Open WebUI → host1:8000 + host2:8000 (vLLM) · agentgateway oder Caddy · Stand: 16.11.2026, geprüft am 18.11.2026

> **Kernpunkt:** Alle Anfragen kommen von der IP des Open-WebUI-Servers, IP-Hash bringt also nichts. Verteilt wird nach **Chat-ID**: Jeder Chat bleibt auf einem Host. Dadurch trifft der Prefix-Cache, und lange Verläufe werden nicht bei jedem Turn neu berechnet. Anfragen ohne Chat-ID werden nach Last verteilt.
>
> **Empfehlung:** Variante A, nur agentgateway. Eine Komponente weniger, Token-Budgets und Kosten pro Nutzer bleiben erhalten. Caddy (Variante B) nur, wenn ihr Überlauf bei Überlast braucht oder Ausfälle ganz ohne Fehlermeldung abfangen wollt.

## Prüfergebnis

Die Aussagen zu agentgateway stimmen für **agentgateway 1.7.0** (aktuelles Release, 27.10.2026). Geprüft wurde so:

- Beide Configs gegen das JSON-Schema von v1.7.0 und mit `agentgateway --validate-only`. Zwei Gegenproben mit absichtlich falscher Config wurden abgelehnt, die Prüfung greift also.
- Variante A lief im Container `ghcr.io/agentgateway/agentgateway:v1.7.0` gegen zwei OpenAI-kompatible Mock-Backends: 100 Chats × 3 Turns, danach Ausfall und Rückkehr von host2.
- Die Lastverteilung wurde im Quellcode von v1.7.0 nachgelesen (`crates/agentgateway/src/llm/mod.rs`, `types/loadbalancer.rs`).
- Der Chat-ID-Header wurde im Code von Open WebUI v0.13.0 nachgelesen (`utils/headers.py`).

| Aussage | Ergebnis | Wie geprüft |
|---|---|---|
| Config Variante A ist gültig | ✅ | Schema v1.7.0 + `--validate-only` |
| Chat-Affinity mit zwei Providern in einer Gruppe | ✅ 0 von 100 Chats auf beiden Hosts (Verteilung 51/49) | Test |
| Affinity per Rendezvous-Hash | ✅ `select_by_affinity` = Rendezvous-Hash über die Provider | Quellcode |
| Ohne Chat-ID: Power-of-Two-Choices | ✅ Zwei zufällige Provider, der mit dem besseren Score gewinnt: `health / (1 + latency × (1 + 0,1 × pending))` | Quellcode, Test (31/29) |
| Leere Chat-ID (außerhalb von Chats) → normale Verteilung | ✅ 29/31 | Test |
| Ausfall: 1–2 Anfragen 503, dann Umschalten | ✅ genau 2× 503, danach alle Anfragen auf host1 | Test |
| Host kehrt nach 30 s automatisch zurück | ✅ nach 32 s laufen seine Chats wieder auf host2 | Test |
| Keine `retry`-Policy setzen | ✅ Mit `retry` (attempts 2, codes 502/503) wird host2 **nie** ausgeschlossen: 30 von 30 Anfragen scheitern | Test |
| `hostOverride: "host:port"` nutzt plain HTTP | ✅ | Test |
| `backendAuth.key: "$VLLM_API_KEY"` wird aus der Umgebung ersetzt | ✅ Backend sieht `Bearer <key>` | Test |
| Chat-ID-Header wird ans Backend durchgereicht | ✅ | Test |
| `llm:`-Modus hat kein Feld für Affinity | ✅ `models[]` kennt `health`, aber weder `policies` noch `sessionAffinity` | Schema |
| `agentgateway -f config.yaml --validate-only` | ✅ vorhanden (steht in `--help`, nicht in der Doku) | CLI |
| Config Variante B (agentgateway-Teil) ist gültig | ✅ | Schema + `--validate-only` |
| Open WebUI: Header-Platzhalter `{{CHAT_ID}}` | ✅ | Code v0.13.0 |

**Ergänzungen zur PDF:**

1. **`ENABLE_FORWARD_USER_INFO_HEADERS=true` sendet die Chat-ID schon selbst.** Neben `X-OpenWebUI-User-Id`, `-User-Email`, `-User-Name` und `-User-Role` setzt Open WebUI bei Chat-Anfragen auch `X-OpenWebUI-Chat-Id`. Mit dieser Variable ist der eigene Header aus Schritt 1 also nicht nötig. Er schadet aber auch nicht: Gleicher Name und gleicher Wert, der eigene Header überschreibt.
2. **`unhealthyExpression: 'response.code >= 500'` ist der Default.** Ohne den Ausdruck zählen 5xx und Verbindungsfehler ebenfalls als Fehler. Getestet: gleiches Verhalten (2× 503, dann Umschalten). Die Zeile kann bleiben oder entfallen.
3. **Die Sperrzeit wächst bei wiederholten Ausfällen.** `duration` ist die Basis. agentgateway multipliziert sie mit der Zahl der bisherigen Ausschlüsse, bis der Host wieder eine Anfrage erfolgreich beantwortet (`times_ejected` im Quellcode). Ein Host, der mehrfach hintereinander ausfällt, bleibt also länger als 30 s draußen.
4. **Nach der Rückkehr wandern die Chats zurück.** Das ist die Eigenschaft des Rendezvous-Hash. Chats, die während des Ausfalls auf den anderen Host umgezogen sind, verlieren dort beim Zurückwandern ihren Prefix-Cache. Ein neu gestarteter vLLM hat ohnehin noch keinen. Das ist einmalig und unkritisch.
5. **Retry widerspricht der Doku.** Laut agentgateway-Doku helfen Retries, nach einem Ausschluss einen anderen Provider zu wählen. In 1.7.0 mit `sessionAffinity` verhindert eine `retry`-Policy dagegen den Ausschluss. Nach einem Upgrade erneut testen.

**Nicht nachgeprüft:**

- Variante B mit Caddy (kein Caddy vorhanden).
- Die Aussage, dass Affinity bei einem DNS-Namen mit zwei A-Records oder bei zwei `host:`-Backends nicht greift. Getestet wurde nur der empfohlene Weg mit zwei Providern in einer Gruppe.
- Der Betrieb mit echtem vLLM.

## 1. Open WebUI: Chat-ID mitsenden

Admin-Einstellungen → Verbindungen → Verbindung zu agentgateway bearbeiten → Headers (JSON). Der Platzhalter wird pro Anfrage gefüllt:

```json
{ "X-OpenWebUI-Chat-Id": "{{CHAT_ID}}" }
```

Alternativ, oder zusätzlich für die Nutzer-Zuordnung in agentgateway: `ENABLE_FORWARD_USER_INFO_HEADERS=true`. Das sendet `X-OpenWebUI-User-Id`, `-User-Email`, `-User-Name` und `-User-Role` und bei Chat-Anfragen auch `X-OpenWebUI-Chat-Id`. Außerhalb von Chats (z. B. „Verbindung prüfen“) ist die Chat-ID leer. Dann greift die normale Lastverteilung.

## 2. Variante A: agentgateway allein (empfohlen)

```yaml
# yaml-language-server: $schema=https://agentgateway.dev/schema/config
gateways:
  default:
    port: 3000
routes:
- name: qwen38-flash-next
  backends:
  - ai:
      groups:
      - providers:
        - name: host1
          provider:
            openAI:
              model: qwen3.8-flash-next
          hostOverride: "host1:8000"
        - name: host2
          provider:
            openAI:
              model: qwen3.8-flash-next
          hostOverride: "host2:8000"
    policies:
      sessionAffinity:
        source: request.headers["x-openwebui-chat-id"]
      health:
        unhealthyExpression: 'response.code >= 500'  # Default, darf entfallen
        eviction:
          consecutiveFailures: 2
          duration: 30s
  policies:
    backendAuth:
      key: "$VLLM_API_KEY"
```

`policies` mit `sessionAffinity` und `health` steht neben `ai`, nicht darin. Innerhalb von `ai` ist neben `groups` kein weiteres Feld erlaubt.

- **Wichtig: zwei Provider in einer Gruppe.** Nur so greift `sessionAffinity`. Im Test der PDF wurden Anfragen bei einem DNS-Namen mit zwei A-Records oder bei zwei `host:`-Backends abwechselnd verteilt, die Affinity wirkte nicht. Im vereinfachten `llm:`-Modus gibt es kein Feld dafür.
- **Ohne Chat-ID:** Power-of-Two-Choices. Von zwei zufällig gewählten Hosts gewinnt der mit besserer Gesundheit, Latenz und weniger laufenden Anfragen.
- **Ausfall:** Nach 2 Fehlern (≥ 500 oder Verbindungsfehler) wird der Host 30 s ausgeschlossen, bei wiederholtem Ausfall länger. Seine Chats wandern auf den anderen Host. Die ersten 1–2 Anfragen an den toten Host scheitern mit 503. Danach kehrt er automatisch zurück, und seine Chats mit ihm.
- **Keine `retry`-Policy setzen:** Sie verhindert, dass der ausgefallene Host ausgeschlossen wird (dauerhaft 503, am 18.11. nachgetestet).
- Bestehende JWT-, API-Key- und Budget-Policies kommen wie bisher unter `routes[].policies`.
- Prüfen: `agentgateway -f config.yaml --validate-only` gibt „Configuration is valid!“ aus.

## 3. Variante A oder B?

| | A: agentgateway | B: agentgateway + Caddy |
|---|---|---|
| Chat-Affinity | ja (Rendezvous-Hash) | ja (Header-Hash) |
| Ohne Chat-ID | Last + Latenz (P2C) | wenigste Verbindungen |
| Host fällt aus | 1–2 Anfragen 503, dann Umschalten | kein Fehler (Retry + Health-Check) |
| Überlast eines Hosts | Chat bleibt gebunden | Überlauf ab 32 Anfragen |
| Komponenten | 1 | 2 |
| Getestet mit | agentgateway 1.7.0 | Caddy 2.10.2 |

## 4. Variante B: agentgateway → Caddy → vLLM

agentgateway behält Auth und Budgets und zeigt auf Caddy. agentgateway reicht den Chat-ID-Header unverändert durch (im Test der PDF bestätigt; der Nachtest vom 18.11. bestätigt es für die Route aus Variante A, nicht für den `llm:`-Modus).

```yaml
# yaml-language-server: $schema=https://agentgateway.dev/schema/config
llm:
  port: 3000
  models:
  - name: qwen3.8-flash-next
    provider: openAI
    params:
      model: qwen3.8-flash-next
      baseUrl: "http://qwen-lb.intern:8080/v1"
      apiKey: "$VLLM_API_KEY"
```

```caddyfile
# Caddy als Load Balancer vor zwei vLLM-Instanzen (host1/host2:8000)
http://:8080 {
  @api path /v1/*
  handle @api {
    reverse_proxy host1:8000 host2:8000 {
      # Gleiche Chat-ID -> gleicher Host (Prefix-Cache-Treffer)
      lb_policy header X-OpenWebUI-Chat-Id {
        fallback least_conn
      }
      lb_try_duration 30s
      lb_try_interval 250ms
      # Aktiver Health-Check gegen vLLM
      health_uri /health
      health_interval 10s
      health_timeout 5s
      # Passiver Health-Check + Überlauf
      fail_duration 30s
      max_fails 3
      unhealthy_status 5xx
      unhealthy_request_count 32
      # Token-Streaming sofort durchreichen
      flush_interval -1
      # POST-Bodies puffern, damit bei Ausfall auf den anderen Host wiederholt werden kann
      request_buffers 16MB
      transport http {
        dial_timeout 3s
        keepalive 5m
        keepalive_idle_conns_per_host 128
      }
    }
  }
  handle {
    respond 404
  }
}
```

- **`lb_policy header … fallback least_conn`:** Chat-Affinity. Ohne Header geht die Anfrage an den Host mit den wenigsten laufenden Anfragen.
- **`request_buffers 16MB`:** Ohne Pufferung kann Caddy POST-Anfragen bei einem Ausfall nicht auf den anderen Host wiederholen (im Test 1× 502). Mit Puffer gab es keinen Fehler. Prompts über 16 MB werden trotzdem weitergeleitet, nur nicht wiederholt.
- **`unhealthy_request_count 32`:** Überlauf. Hat ein Host 32 laufende Anfragen, gehen neue Anfragen an den anderen, auch die von gebundenen Chats. Richtwert: 2 × `max-num-seqs` von vLLM. Sind beide voll, wartet Caddy bis zu 30 s (`lb_try_duration`).
- **Keine Timeouts gesetzt:** Caddy hat standardmäßig keinen Antwort-Timeout. Das ist wichtig für lange Prefills mit 100K+ Token. `flush_interval -1` reicht gestreamte Token sofort durch. Nur `/v1/*` wird weitergeleitet.

## Tests und Betrieb

- **PDF, 16.11.2026:** Beide Varianten wurden mit zwei Test-Backends geprüft, nicht mit echtem vLLM. 200 Chats mit zufälligen IDs ergaben 100/100 (agentgateway) bzw. 97/103 (Caddy), kein Chat lief auf beiden Hosts.
- **Nachtest, 18.11.2026:** Variante A mit agentgateway 1.7.0 und zwei Mock-Backends. 100 Chats × 3 Turns ergaben 51/49, kein Chat lief auf beiden Hosts. Ausfall, Rückkehr und Retry wie oben in der Tabelle.

**Im Betrieb prüfen:** In den vLLM-Logs beider Hosts sollte eine Chat-ID nur auf einem Host auftauchen. Die Prefix-Cache-Trefferquote in `/metrics` sollte deutlich steigen.

**Quellen:**

- agentgateway.dev/docs (Backends, Load Balancing, Backend Health, Virtual Models, Retries)
- JSON-Schema v1.7.0: `raw.githubusercontent.com/agentgateway/agentgateway/refs/tags/v1.7.0/schema/config.json`
- Quellcode agentgateway v1.7.0
- caddyserver.com/docs/caddyfile/directives/reverse_proxy
- docs.openwebui.com (Custom Headers, `ENABLE_FORWARD_USER_INFO_HEADERS`)
- Quellcode Open WebUI v0.13.0
