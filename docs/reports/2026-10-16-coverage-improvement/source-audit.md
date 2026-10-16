# Public-source eligibility audit

Checked 2026-10-16. Public/free sources only. No newly downloaded tracking has been added to training.

| Source | Verified evidence | Decision |
|---|---|---|
| Existing BDB 2023/2025 material | Local prepared manifest already contains 2021 W1–8 and 2022 W1–9 | Re-downloading overlapping seasons is not new independent evidence |
| nflverse 2018 participation | 47,875 rows; 18,297 labels in the seven-class vocabulary, plus 97 Prevent labels | Candidate labels only; no verified tracking join |
| nflverse 2023 participation | 46,168 rows; 22,916 non-null labels, including COMBO, COVER_9 and BLOWN | New source/taxonomy requires explicit mapping; no tracking coordinates |
| nflverse 2025 participation | 45,184 rows; 22,055 non-null labels, including out-of-vocabulary labels | Same restriction |
| BDB 2021 tracking | Official file-list API returns HTTP 401; no local Kaggle credential file or environment credentials | Cannot verify downloadable sample, pre-snap duration or player completeness |
| BDB 2027 tracking | Official file-list API returns HTTP 401 | Cannot verify season, schema, joins or reuse terms |

The [nflreadr participation documentation](https://nflreadr.nflverse.com/reference/load_participation.html) identifies NGS as the source through 2022 and FTN from 2023, and says participation data is released after the postseason. Attribution and CC-BY-SA terms apply. Postseason publication does not make eventual coverage, pressure, pass-rusher count, route or time-to-throw valid pre-snap inputs.

[sources.json](sources.json) records URLs, downloaded byte sizes, SHA-256 hashes, schemas, coverage counts and authentication failures. The three participation files were inspected in memory; no bulk tracking was downloaded. Authentication must use an authorized Kaggle account and applicable competition terms before proceeding; third-party mirrors are not a substitute for verified access.

A future source may enter training only after checking stable game/play joins, overlap against existing keys, pre-snap frames at the required cutoffs, player completeness, label compatibility and documented reuse rights. The label counts above are not counts of eligible new training plays.
