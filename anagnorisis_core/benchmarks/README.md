# Benchmarks
To measure the current the benchmark numbers run the following:

```bash
python3 anagnorisis_core/benchmarks/search_benchmark.py \
    --media /path/to/media --index --describe-sample 10 --out results.json
```

Needs `anagnorisis-core` installed. The published figures use the project's internal demo set of files that is not yet public.

## What is measured

There are several important processes each search query goes through to provide the results that are measured separately:

| Phase | Scales with | Notes |
|---|---|---|
| `list` | number of files | walking the filesystem for candidates |
| `query` | constant | embedding the query text |
| `score` | number of files | cache lookups and similarity / per-file cost |

Keeping `query` separate is not fastidiousness. It is one short piece of text through the model on the CPU, and on the measured set it costs ~0.28 s while scoring all 439 files costs ~0.03 s. Fold the two together, divide by the file count and extrapolate, and you attribute that fixed 0.28 s to every file, turning a 7-second estimate into a 70-second one. 

So extrapolation is `per-file score × N + query`, adding the constant once rather than multiplying it.

Reported separately for each mode:

- **cold**: the cache has to come off disk into RAM. What you feel on the first search after a restart.
- **warm**: the average of five repeats with the cache already in RAM. What you feel while working.

Model loading is excluded from both and reported on its own. It is a one-off of tens of seconds that has nothing to do with how long searching takes, and folding it in would drown every other number.

**Only indexed files count.** An unindexed file is skipped rather than scored, so including a pile of them would make search look fast by giving it nothing to do. The `indexed` column says how many of the measured files actually had an embedding to compare against.

## Indexing: two costs, orders of magnitude apart

"How long does it take to index 100,000 files?" has two answers, and quoting the wrong one is off by a factor of hundreds:

| Step | What it does | Order of cost |
|---|---|---|
| `describe` | the descriptor model writes a description of the file | **seconds** per file |
| `index` | embeds the file's content, and embeds its description *text* | fractions of a second per file |

`api.index` never generates a description. It passes `generate_desc_if_not_in_cache=False` and embeds whatever text already exists, from the cache or from a `.meta` sidecar. So the `index` figures are the cost of indexing the descriptions a library *already has*: exactly what a data-server folder costs when it arrives with its sidecars, and nothing like what a fresh library of undescribed files costs.

Both are measured, separately, and the descriptor is sampled per media type because the spread between a text file and a long video is the whole story. A blended figure is given too, but the per-type rows are the ones that travel: the media mix is what differs between libraries.

Two more things are kept separate for the same reason the query is kept out of the score. They are paid once per pass, not once per file, so multiplying them by a file count would be nonsense:

- **Model loading.** Measured once and subtracted from every phase.
- **Re-indexing an up-to-date library.** A second pass where every embedding is already cached. This is what the scheduled background passes actually cost most of the time, so it gets its own row.

## Quality: self-retrieval

Each run also measures **self-retrieval** quality score: take a file's own description as the query, and check whether that file comes back first.

- `recall@1`: how often the file itself is the top hit.
- `recall@5`: how often it is in the top five.
- `MRR`: mean reciprocal rank, which rewards being second over being twentieth.

The main benefit of this metric that it needs no labelled dataset, however it is not a perfect measure as we cannot guarantee the quality of the descriptions in the fist place. It is a floor, not a ceiling. A file matching its *own* description is the easiest possible query. A proper relevance dataset is still to come, and these numbers should not be mistaken for it.

## Results

Each entry records the version, the machine, and how many files were actually measured. Numbers for 100k files are **extrapolated** from the per-file cost unless the file count says otherwise. The demo set is a few hundred files, so scaling is linear-in-theory and should be re-measured on a real massive library before being trusted.

<!-- Append a new block per release. Do not edit old ones: the point is the trend. -->

### 0.4.10 — 2026-08-21

- **Machine:** NVIDIA GeForce RTX 3070 Laptop GPU 8GB, 30GB RAM, media on an external HDD drive
- **Measured on:** 439 indexed files (demo set); 100,000-file figures are extrapolated from the per-file cost
- **Listing the files:** 0.004s for 439 files (9.2 µs/file) through the cache
- **Process startup:** the first `collect_files` in a fresh process took 1.4978s, of which ~1.4938s is one-time imports (pyfilesystem, the media-type taxonomy) rather than walking. A CLI invocation pays this once; the app pays it at boot.

**Finding the files on the drive**, cached per directory by mtime, so three regimes:

| Regime | Time | µs/file | Est. @100,000 |
|---|---|---|---|
| uncached (cache empty) | 0.0237s | 53.96 | 5.4s |
| cold (cache on disk) | 0.0053s | 12.11 | 1.2s |
| warm (cache in RAM) | 0.0045s ±0.0 | 10.29 | 1.0s |

The *uncached* row is measured with our cache empty but the operating system's own directory cache were likely warm, so it is a floor rather than a true first-ever scan. 

**Building the index**: embedding each file's content, and embedding its description text. This does *not* include writing the description in the first place; that is the table below, and it dominates.

| Media type | Files | Content s/file | Descriptions s/file |
|---|---|---|---|
| `audio` | 205 | 1.3986 | 0.367 |
| `images` | 202 | 0.5551 | 0.3692 |
| `text` | 17 | 0.352 | 0.4358 |
| `videos` | 15 | 3.7895 | 0.4031 |

| Phase | Total | s/file | Est. @100,000 |
|---|---|---|---|
| content | 461.67s | 1.0516 | 105,176s (29.2h) |
| descriptions | 163.28s | 0.3719 | 37,205s (10.3h) |
| **both, whole library** | — | — | **142,375s (39.5h)** |
| re-index, everything already cached | 0.022s | 0.05 ms | 5s |

Embedder load, measured once and excluded above: 12.28s. The *re-index* row is what a scheduled background pass costs on a library that is already up to date, which is the case that runs over and over.

**Writing the descriptions**: the descriptor model, sampled per media type (3 files each). This is the real cost of indexing a fresh library, and it is measured in hours:

| Media type | s/file (mean) | median | min–max | n | All 100,000 of this type |
|---|---|---|---|---|---|
| `audio` | 47.38 | 45.84 | 45.02–51.29 | 3 | 1316.2h |
| `images` | 15.63 | 15.75 | 14.43–16.71 | 3 | 434.2h |
| `text` | 5.31 | 0.0 | 0.0–15.94 | 3 | 147.6h |
| `videos` | 55.93 | 56.94 | 52.81–58.03 | 3 | 1553.5h |

At this set's media mix (31.43s per file on average), 100,000 files would take **873.2 hours**. The mix is what varies between libraries, so the per-type rows are the ones that travel.

Descriptor load, once: 13.07s.

**Searching**, once the files are known:

| Mode | Indexed | Query (constant) | Score cold | Score warm (avg 5) | Cold µs/file | Warm µs/file | Est. cold @100,000 | Est. warm @100,000 |
|---|---|---|---|---|---|---|---|---|
| `name` | 439/439 | 0.0s | 0.0102s | 0.0101s ±0.0001 | 23.15 | 23.03 | 2.3s | 2.3s |
| `semantic` | 439/439 | 0.3068s | 0.0225s | 0.0188s ±0.0008 | 51.32 | 42.83 | 5.4s | 4.6s |
| `metadata` | 439/439 | 0.3033s | 0.0214s | 0.0215s ±0.0019 | 48.78 | 48.99 | 5.2s | 5.2s |

| Mode | recall@1 | recall@5 | MRR | sampled |
|---|---|---|---|---|
| `metadata` | 0.6 | 0.667 | 0.649 | 15 |
| `semantic` | 0.467 | 0.8 | 0.574 | 15 |

Model warm-up, excluded from the timings above: name 0.0s, semantic 9.12s, metadata 0.272s.
Queries used: `name`: "winter", `semantic`: "a quiet acoustic recording", `metadata`: "a quiet acoustic recording".

#### What this run says

- **Short text now costs nothing.** The mean is 5.31s and the median is **0.00s**, because text under 4,000 characters is kept verbatim instead of being summarised. Only the long ones reach a model at all. That is the largest proportional saving anywhere in this table: 15.6s → 5.3s per file.
- **Video is a third cheaper**, 82.1s → 55.9s, from sampling three windows instead of five. Audio pays the same three-window shape.
- **A scheduled pass over an up-to-date library is four times cheaper**, 0.205 → 0.05 ms/file, or five seconds for 100,000 files. This is what the background passes do almost every time they run.

The headline for anyone planning a library: **describing still dominates by a factor of twenty**. 39.5 hours to embed 100,000 files, against 873 hours to describe them first, and that blend is 47% audio. A library of video would be 1,553 hours; a library of short text files, close to nothing.
