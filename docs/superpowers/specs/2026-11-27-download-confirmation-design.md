# Early spoken confirmation for download requests

**Date:** 27.11.2026
**Status:** approach 1 approved by the user («ладно делай по первому»)

## Intent and why

The user said «скачай восьмой сезон Симпсонов» in the living room at
13:10:31 MSK. Hermes added the torrent at 13:13:56 MSK, but the gateway's
SSE socket died at 13:11:32 MSK and the room said «Простите, что-то пошло не
так». Fixing the socket timeout let the turn wait, but the user then said
plainly: nobody waits several minutes, do not say «скачано» at completion —
say **«Скачивается <название>»** at the moment the torrent is found and added
to the download queue. The phrase itself says what it was about, so a late
confirmation still makes sense; a completion announcement does not.

## Goals

- A voice download request with a parseable title and season goes straight
  from the router to `torrent-updater`'s MCP tool `search_and_download`
  (`:8020/mcp`), the service's documented agent interface.
- The turn emits one sentence, «Скачивается <название>.», at the moment that
  call returns success (which is exactly when the torrent was added), then
  ends with `done`. No Hermes call, no «скачано», no later announcement.
- A request whose shape we cannot parse keeps today's Hermes fallback with
  the silent SSE keepalive from the previous fix. A request we CAN parse
  stays deterministic all the way: success says one fact («Скачивается …»),
  a definite tool refusal says one fact («Не нашла подходящего релиза.»),
  and neither hands the turn to Hermes. The tool-agent path guessed a show
  from earlier turns and narrated every step (27.11.2026 field case).

## Non-goals / anti-goals

- No change to `torrent-updater`, Transmission or any camera setting.
- No notification when the download finishes, and no polling of Jellyfin.
- No proactive speech outside the turn (that would bypass the echo ring and
  the follow-up window).

## Constraints

- `torrent-updater` runs on the host; its MCP server is at `:8020/mcp`
  (streamable HTTP) and wraps the `:6050` API. The tool takes 1–3 min
  (one browser session) and answers HTTP 200 even on failure — `success`
  in the tool body is the verdict. The client must do `initialize` →
  `notifications/initialized` → `tools/call`, carry `Mcp-Session-Id`, and
  read the result from the SSE `data:` line (`result.content[0].text` is a
  JSON string). No credentials.
- The gateway reads `/route` with `sock_read=60 s` (silent `keepalive`
  events from the router already keep that alive) and `total=200 s`.
  A direct browser call can exceed 200 s, so the gateway's total budget must
  rise to 300 s (`CASCADE total_timeout`, overridable in code).
- The router's parse must stay narrow: a question, a device command and an
  ordinary utterance must never become a download.

## Approaches

1. **Chosen — router calls `search-and-download` directly.** The router
   already owns deterministic integrations (`scale.py`, HA REST, Kodi); a
   new sibling module parses the Russian phrase and speaks the response.
   Cost: one narrow parser; unusual phrasings fall back to Hermes. Benefit:
   the confirmation comes from the service that actually added the torrent,
   with no LLM in the path and no race against the gateway's timeout.
2. *Hermes searches, router polls a new «last added» endpoint.* Keeps natural
   language understanding but needs a second service change, keeps the
   ~3.5 min Hermes overhead and still needs the gateway budget raised.
3. *Hermes calls `voice_mcp.speak` mid-tool-loop.* Speech would leave the
   turn's echo/window guards and depends on the model choosing to call it —
   rejected as unreliable.

## Shape of the implementation

- New `services/jev-router/download.py`: `parse_request(text)` →
  `DownloadRequest | None` (query, kind, season); `search_and_download(...)`
  → result dict via injected `post` (no aiohttp needed in unit tests);
  `confirmation_sentence(result, request)` → Russian sentence. Follows
  `scale.py`'s injected-`post` pattern.
- `services/jev-router/app.py`: the `download_request` branch calls the new
  client instead of `chat_proxy.stream_chat`; on success emits one `sentence`
  and lets the normal `done` close the turn; on failure falls back to Hermes
  (same route event, no double acknowledgement).
- `config.py` gains `TORRENT_UPDATER_MCP_URL` (host fallback
  `http://192.168.22.102:8020/mcp`) and `DOWNLOAD_TIMEOUT_S = 240`;
  no new secrets. `download.py` speaks MCP JSON-RPC over `aiohttp`
  (no new dependency in the router image).
- `backends.py`: `CascadeBackend.total_timeout` default 200 → 300.

The user's own words name the content («Симпсонов»), so the confirmation can
use the requested phrase rather than the tracker's long title — e.g.
«Скачивается восьмой сезон Симпсонов.». The tracker title stays in the log.

## Post-review corrections (27.11.2026)

- The direct browser call also emits the existing textless `keepalive` SSE
  event every `EXPERT_KEEPALIVE_S` while waiting. Without it the gateway's
  `sock_read=60 s` would apologize at +60 s, long before the add at +205 s.
- A timeout or an unreadable response from `search-and-download` is reported
  as `download.UNKNOWN`: the add may already have happened, so the router
  says «Не удалось подтвердить добавление в закачки.» and does **not** send
  the text to Hermes (which would add the torrent twice). Only a definite
  refusal (`success:false`) falls back to Hermes.
- `confirmation_sentence(req)` — the spec's earlier two-argument wording was
  a typo; the plan and code use the single `DownloadRequest`.
- Transport moved from the raw REST endpoint to the MCP tool on 27.11.2026
  after `torrent_updater/AGENT_GUIDE.md` named MCP the preferred agent
  interface. A failure BEFORE `tools/call` is a definite failure; a timeout
  or unreadable answer AFTER it is `UNKNOWN` (the add may have happened —
  never retry, never claim success).
- The working mechanism was restored by putting the repository's own
  `search_and_download.py` back into the container: Hermes had copied a stale
  copy over it, removing `download_url_and_add_to_transmission`. From then on
  the MCP tool follows the same path that succeeded in the field: Rutracker
  tracker search → best release → Transmission add → topic URL stored as
  the torrent's tracking label/comment.
- User's follow-up decision: no step-by-step narration. A parseable request
  produces exactly one factual spoken sentence; a definite refusal is a
  short refusal, not a task for Hermes to guess at. `CascadeBackend` also
  suppresses the generic «Вопрос услышан. Секунду, думаю.» ack for
  `reason=download_request` — otherwise the turn still spoke two phrases.
