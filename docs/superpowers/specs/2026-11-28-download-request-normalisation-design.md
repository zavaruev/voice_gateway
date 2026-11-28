# Download requests: general normalisation and an early acknowledgement (v2.92)

## Intent

A voice request to download a series must reach the deterministic client whatever declension or verb form Whisper writes, and the room must not sit silent while the browser searches RuTracker. The 28.11.2026 field case: «скачает третий сезон Симпсонов» (future tense) was classified as suspect STT and produced only «Похоже, я неверно расслышала… повторите»; after a corrected repeat, the search needed 2 min 24 s (12:29:47 → 12:32:11 MSK) before anything was spoken.

## Approaches considered

1. **Per-word alias tables** (`Симпсонов → Симпсоны`, one entry per declined title). Rejected: every new title needs another edit; the same applies to conjugating verbs.
2. **Search both the raw and normalised query in torrent-updater.** Rejected for now: a second browser navigation per request is expensive and Cloudflare-sensitive; the single-session discipline in `torrent_updater/AGENTS.md` forbids overlapping work.
3. **Lemmatise once at the router's MCP boundary with `pymorphy3`.** Chosen: one place serves cameras, ESP32 and Telegram; the search service keeps its single navigation; the spoken confirmation continues to use the user's own words.

## Design

- `services/jev-router/query_norm.py:normal_query()` lemmatises each outbound query token. A plural-inflected word is sent as nominative plural (`Симпсонов → Симпсоны`), not as the singular lemma, and an adjective is inflected to agree with the following noun (`Черного зеркала → Чёрное зеркало`) because the title filter matches whole words. Capitalisation is preserved.
- `query_norm.py:normalise_download_verb()` brings the first download verb among the first three tokens to its infinitive. The set is finite (`скачать`, `скачивать`, `загрузить`, `загружать`, `качать`) and is shared by `resolver.is_download_request()` and `download.parse_request()`.
- `download.py` emits the acknowledgement before creating the shielded search task; the factual sentence, refusal or “cannot confirm” still waits for the result. `dobbi_style` maps the acknowledgement to `Домовой пытается скачать{h}.`

## Guards (and why)

- **Past tense** (`tag.tense == "past"`) is skipped: «скачал серию…» is a report, not an order.
- **Negation** immediately before the verb (`не`/`нет`) refuses: «не скачай серию». Narrow on purpose — «не забудь скачать…» stays a request.
- **Proper names**: if a capitalised token would become a strict prefix of itself (`Влада → Влад`), keep the original spelling; a dictionary cannot be trusted with names.
- **Unknown forms** and a missing `pymorphy3` dependency pass through unchanged. A normalisation failure must never break the download path.

## Non-goals / limits

- Foreign names the analyser does not inflect (`Интерстеллар`) are sent as heard.
- The user's words are never rewritten in the spoken confirmation; only the outbound query changes.
- The browser search latency is not removed; only the silence during it is.

## Verification

Tests were written first and watched failing for each rule: declined nouns, plural form, adjective agreement, conjugated verbs, past tense, negation, proper names, and the acknowledgement arriving while the tool is blocked. Full in-image suite at release: **1163 passed, 4 skipped, 1 pre-existing failure** (`test_ring_buffer_scores_identically`). The router was rebuilt and clone-recreated behind an exit-code image probe and a `/health` gate; a live probe with a blocked tool showed `Пытаюсь скачать.` on the stream before the result, then `Скачивается третий сезон Симпсонов.`

Release: v2.92, 28.11.2026.
