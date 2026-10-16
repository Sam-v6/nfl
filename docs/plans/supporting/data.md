# Free NFL data for improving pre-snap coverage prediction

Research date: October 11, 2026. Scope: public/free sources only. No vendor contact, restricted download, bulk tracking download, or training was performed. Three small public participation Parquet files were read in memory to verify actual contents. Counts below are label rows, not verified new tracking examples.

## Which free sources can add new labeled plays?

### Takeaway

The strongest concrete lead is **2018 Big Data Bowl tracking joined to nflverse's full-season 2018 coverage labels**. This is a conditional opportunity, not a verified 18,000-example addition: pre-snap duration, missing linemen, join coverage, and source-label agreement must pass an audit. Also audit the brand-new 2027 competition immediately; it has regular-season tracking, but its suitability remains unverified.

### Cited Findings

- **BDB 2021 means the 2018 season, not 2021.** The competition's data description embedded in a contestant's Kaggle input page identifies passing plays throughout 2018 and explicitly excludes offensive AND defensive linemen. It lists weeks 1–17. This requires a masked/common-player representation, not pretending we have complete all-22 tracking. — [Kaggle competition input description](https://www.kaggle.com/zachdrapkin/target-agnostic-evaluation-of-cbs-wrs/data); independently corroborated by the dataset section of [original research](https://link.springer.com/article/10.1186/s40537-026-01416-6).
- The old BDB 2021 bonus labels covered only 2018 week 1; that historical limitation should not be mistaken for the current label availability. — [Open Source Football author's worked example](https://opensourcefootball.com/posts/2021-05-31-computer-vision-in-r-using-torch/).
- Current **nflverse participation release** has downloadable Parquet assets for 2016–2025. Live GitHub API inventory verified this. Public 2018 file contains **47,875 unique game/play keys and 18,394 non-null coverage labels**. Of those, **18,297** are in the seven target classes, before joining tracking or checking exclusions. Counts: Cover0 701, Cover1 5,292, Cover2 2,302, Cover3 6,287, Cover4 2,055, 2-Man 463, Cover6 1,197, Prevent 97. Computed directly from the source file, not estimated. — [Official release](https://github.com/nflverse/nflverse-data/releases/tag/pbp_participation), [2018 asset](https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_2018.parquet).
- Live 2023 participation file has **46,168 unique game/play keys; 22,916 labeled rows**. Live 2025 has **45,184 unique keys; 22,055 labeled rows**. These newer labels are real free data, but do not themselves include tracking coordinates. — [2023 asset](https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_2023.parquet), [2025 asset](https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_2025.parquet).
- Participation through 2022 is from NFL NextGenStats; 2023 onward comes from FTN after the postseason. Attribution is required under CC-BY-SA 4.0. This is a label-source change that must be measured, not silently pooled. — [Official loader and attribution](https://nflreadr.nflverse.com/reference/load_participation.html).
- **BDB 2027 launched October 6, 2026**, and the official page says Combine sensor tracking is supplied alongside regular-season NFL tracking. The publicly parsed data page did not expose the schema, seasons, player completeness, or snap timing. — [Official timeline/description](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/overview/timeline), [data page](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/data).

### Inferences

- Prioritize a bounded 2018 sample audit: inspect actual frames strictly before ball_snap, available player positions, time spans, IDs, duplicate counts, and label joins. If sufficient, add 2018 as older training data and test whether it improves 2021/2022 forward validation. Keep a common-player ablation so additional data is not confounded with a different representation. Full 2018 regular season can supply at most 18,297 seven-class label candidates from this file; the actual retained count is unknown.
- Inspect BDB 2027 metadata and one small lawful sample before any bulk acquisition. Qualify only records with pre-snap frames, suitable defensive/offensive player coverage, stable snap timestamps and joinable eventual labels. If it qualifies on newer seasons, reserve a chronologically later subset as untouched test data **before scoring or model-directed case inspection**.
- Since 2018 is older, a held-out 2018 test would provide a previously unseen cohort but would not demonstrate future-season deployment generalization. Use precise wording.

### Gaps

- No tracking sample from BDB 2021 or 2027 was downloaded in this research. Exact usable pre-snap horizons and new unique retained plays remain unknown.
- BDB 2021/2027 full competition rules were not retrievable through the text browser. Download through official Kaggle access, preserve license/rules snapshot, and respect account/rules acceptance; do not substitute arbitrary mirrors.
- Exact NGS/PFF/FTN label agreements are unmeasured. Cover4→Quarters is an explicit taxonomy mapping to validate; PREVENT, COVER_9, COMBO, BLOWN must remain separately accounted, not coerced to convenient labels.
- Three verified public-file SHA256s: 2018 `334f03887775e98f818ab7f5b22c7973f050cae7c3594e5d4b6f5c9c9890969d`; 2023 `b157736972248e6f71e0c5ad6c1010b71375a500a63944dc4956c65de263e5a6`; 2025 `3b26b9487267f011fdbafbce133800ce964c963fdb8b27725abf65663cc46e0f`. These are research evidence identities, not acquisition manifests.

## Which apparent new datasets are duplicates or the wrong prediction time?

### Takeaway

Competition edition names are misleading: BDB 2023 and 2025 largely describe the seasons already in this project. BDB 2026 offers newer games but “before the throw” is not “before the snap”; do not treat it as proven pre-snap training data.

### Cited Findings

- BDB 2023 is **2021 weeks 1–8**, dropback passes. The host announcement describes tracking from snap to pass release; local audit should determine whether short leading pre-snap frames exist. This covers the existing 2021 range, not new seasons. — [Competition host announcement](https://www.kaggle.com/competitions/nfl-big-data-bowl-2023/discussion/359079), [CMU workshop](https://www.stat.cmu.edu/cmsac/conference/2022/workshop/slides.html).
- BDB 2025 is **2022 weeks 1–9**, with full pre-snap-to-post-snap tracking. Its dictionary exposes PFF coverage, man/zone, individual coverage responsibilities, and matchup IDs. Reusing these labels or richer features may help, but it does not add new games beyond the project's existing range. — [Original research data section](https://academic.oup.com/jrsssa/article/189/3/1988/8315856), [official Kaggle dictionary](https://www.kaggle.com/c/nfl-big-data-bowl-2025/data).
- BDB 2024 tackling data also span **2022 weeks 1–9**, 136 games; its 12,486 plays are a task-specific sample, not 12,486 independent new examples beyond BDB 2025. — [Original research, Methods](https://www.nature.com/articles/s41598-025-85993-1).
- NFL describes BDB 2026 tracking from **2023 and 2024**, evaluated on 2025 weeks 14–18, for movement while the ball is airborne. The Analytics data dictionary specifically lists input_2023_w01–18 and describes tracking before the pass is thrown; that page and broad NFL overview do not establish that every edition exposes both seasons or any pre-snap window. — [NFL official overview](https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl), [Analytics data dictionary](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data).
- The 2026 Analytics page displays CC-BY-NC 4.0 and rules acceptance; eventual target receiver/ball landing/pass result fields exist and would leak future information if used as pre-snap inputs. — [Official data dictionary](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data).

### Inferences

- Count incremental **unique season/game/play keys**, not files, frames, cutoffs or labels. Reacquiring BDB 2023/2025 might repair missing records but has no demonstrated season expansion.
- BDB 2026 is a secondary sample audit only; until actual timestamps prove otherwise, classify it as post-snap/pre-throw data. It could support a later identification model or auxiliary representation experiment, but that is outside the first acquisition priority.
- Existing BDB 2025 defender responsibility labels offer a training-only auxiliary task without buying more data. Use actual pre-snap inputs only; never feed coverage responsibilities, matchup IDs, actual routes, post-snap alignment changes, or pass outcomes into inference. Test versus same-data single-task models.

### Gaps

- No verified freely downloadable full all-22, multi-second pre-snap tracking release for a completely new season was established here. BDB 2027 is the fresh lead, not a confirmed solution.
- BDB 2026 boundaries should be proven with raw sample events rather than reverse-inferred from competition winners' preprocessing. A winner's ball_snap→pass_forward extraction may refer to their supplementary 2018 data, not definitively every original file.

## What is the highest-value public-only acquisition plan?

### Takeaway

Acquire **richer causal context and verified older labels first**, while testing newer tracking eligibility. More labeled tabular plays can improve a context model and strengthen conclusions even if matching tracking is unavailable; do not claim they directly enlarge the tracking training set.

### Cited Findings

- nflverse participation includes offense formation, personnel, defenders in box, defensive personnel, player IDs and eventual coverage labels. Its documented coverage vocabulary includes Cover0/1/2/3/4/6/9, 2-Man, Combo and Blown. — [Official participation dictionary](https://nflreadr.nflverse.com/articles/dictionary_participation.html).
- The distinct `load_ftn_charting` public subset begins in **2022** and includes QB alignment, starting hash, backfield count, box count and motion-related flags; it is not the same table as participation and its current dictionary does not expose coverage type. — [FTN charting dictionary](https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html), [official FTN loader](https://nflreadr.nflverse.com/reference/load_ftn_charting.html).
- FTN participation is released after the postseason, unlike the regularly updated FTN charting subset. Do not build an in-season lagged-coverage feature assuming current-season labels are available live. — [Official availability schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html).
- BDB2026 explicitly documents BDB gameId/playId to nflverse old_game_id/play_id compatibility. Any join still needs uniqueness and unmatched-rate checks rather than blind numeric assumptions. — [Official BDB dictionary](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data).

### Inferences

1. **Small metadata/eligibility audit, no GPU:** snapshot official source/version/license/hash; inventory each season and coverage class; validate one-to-one keys and compare overlap with current manifest; report distinct plays, games, pre-snap seconds and missing-player rates. Include BDB2021+2018 participation, BDB2027, and a low-priority BDB2026 boundary check.
2. **Causal context enrichment:** join down, distance, field location, clock, score, formations/personnel and player roles where known at the cutoff. For a -1 second forecast, final at-snap formation, box count and motion flags may be future information: reconstruct from the tracking prefix or omit. Reject outcome/route/pressure/coverage fields as predictors. Fit a context-only boosted-tree baseline on free participation+pbp and evaluate on chronological future seasons as a separate task, then context+tracking on common matched plays.
3. **Older tracking expansion if eligible:** use 2018's permitted labeled, pre-snap prefixes for a common-player masked model. Compare current-only versus plus-2018 under identical 2021/2022 forward folds and same compute; measure minority recall and label-source effects. Do not force 2018 into long-horizon experiments if the required frames are missing.
4. **New-season holdout if eligible:** if 2027 files expose a suitable newer season, seal later games before development; keep a separate calibration slice and class-support audit. If no such source qualifies, state that fresh full-tracking generalization is still unproven and use nested game/time splits as development evidence only.
5. **Stop based on evidence:** only expand bulk downloads after the small sample passes; retain provenance and exclusion receipts. If additional data hurts, test label-source weighting, common-player restriction and time-window harmonization before any larger model search. No claimed score target is guaranteed by more data.

### Gaps

- No commercial purchases, quotes, vendor correspondence, or paid datasets are part of this recommendation.
- Exact acquisition sizes/time for 2018/2027 and the count of retained new plays are unknown until official file manifests and sample checks are available. Free access may still require Kaggle authentication and rule acceptance.
- Whether final at-snap charting can be used for a given forecast horizon is a feature-timing decision, not resolved by the fact that the dataset calls a field “pre-snap.”
