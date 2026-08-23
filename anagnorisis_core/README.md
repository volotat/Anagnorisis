# anagnorisis-core

The engine underneath Anagnorisis: it describes files, embeds them, searches them, remembers what you thought of them, and learns to predict that.

It has no web server, no database and no browser. Three things use it: the Flask application, the `anagnorisis` command line, and the data server. They all call the same functions, so the command line cannot drift from what the app does.

```bash
pip install -e ./anagnorisis_core
anag --help                  # `anagnorisis` also works; `anag` is the short name
anag help describe           # or `anag describe -h`, or `anag -h describe`
```

**It stands on its own.** Installed from a wheel, with no Anagnorisis checkout on the machine and no config file to point at, it runs: which embedder, which descriptor and the generation settings they were tuned with all ship in `anagnorisis_core/defaults.yaml`. So the smallest possible run is

```bash
anag describe /my/media --sink meta
```

Everything it produces goes in one folder, laid out exactly the way the application lays out its `project_config/`:

```
~/.local/share/anagnorisis/
    project_config/
        cache/       derived and disposable, rebuilt on demand
        memory/      your ratings, the only thing here that cannot be regenerated
        models/      the evaluator trained on them
    models/          the downloaded weights
```

So there are only two paths to configure. Set it once and it sticks:

```bash
anag config set project_config_path /path/to/Anagnorisis/project_config
anag config set embedding_models_path /path/to/Anagnorisis/models
```

Because the layout matches the application's, that makes the command line and the app share one cache, one set of ratings and one evaluator. The downloaded weights sit outside the project folder deliberately: they are gigabytes, identical for everyone, and worth pointing several users or checkouts at one copy instead of duplicating.

---

## How the project is laid out

One folder, holding the package and the few things that belong beside it:

```
anagnorisis_core/          this directory *is* the import package
    api.py                 the whole surface, start here
    cli.py                 argparse over api.py, and nothing else
    config.py              settings, and where a machine keeps things
    progress.py            check() and update(); the seam for pausing work
    defaults.yaml          every setting, and what each one costs

    models/                everything that loads weights
        embedder.py  descriptor.py  universal_evaluator.py
        scoring_models.py  model_identity.py  model_manager.py

    search/                ranking files, and the caches that keep it quick
        content_search.py  metadata_search.py  content_cache.py
        proxy.py  ranking.py  common_filters.py  recommendation_engine.py

    storage/               where things are kept, and how they are addressed
        caching.py  sinks.py  virtual_file_system.py
        file_walker.py  file_paths.py  soft_hash.py

    media/                 what a file is, and how to say it in words
        media_types.py  description.py  models.py
        media_types/       extension lists and the tag vocabularies
        extractors/        one reader per kind of internal metadata

    ratings/               what you thought of a file, and learning it
        memory.py  training.py

    pyproject.toml
    README.md
    tests/                 must pass with no Flask, no database, no browser
    benchmarks/            how fast and how good search is, release by release
```

Flat at the top: the folder named `anagnorisis_core` is the package, named once, with no `src/` wrapper. Below that the modules are grouped, so the five files worth reading first are not lost among the other thirty.

Each data file sits beside the code that reads it: `defaults.yaml` next to `config.py`, the taxonomy next to `media_types.py`. That is what lets `dirname(__file__)` find them identically whether the package is installed or run from the tree.

`tests/` and `benchmarks/` sit here but are not part of the distribution: they are not listed in `pyproject.toml` and have no `__init__.py`, so a built wheel contains the modules and their data and nothing else.

Neither data file is meant to be edited: machine settings go to `~/.config/anagnorisis/config.yaml` through `anag config set`, and the taxonomy is deliberately not configurable.

**Install it** rather than relying on where it happens to sit. That is the supported way in, and it is what the containers do:

```dockerfile
COPY anagnorisis_core/ /app/anagnorisis_core/
RUN pip install --no-deps -e /app/anagnorisis_core
```

Editable, and from `/app` because that is where the repository is bind-mounted at runtime, so `pip show` and `import` behave exactly as they would for any dependency, while the code being run is the tree you are editing. `--no-deps` matters: the package declares plain `torch`, which resolves to a CUDA 13 build on PyPI and would replace the image's cu124 wheels. Dependencies belong to the image; the package contributes only its own code.

## Configuring a machine

Settings live in `~/.config/anagnorisis/config.yaml`, outside the installed package, so `pip install -U` cannot overwrite them.

**Do not edit `defaults.yaml` to configure a machine.** It ships inside the package: an upgrade replaces it, and anything you put there travels to whoever installs the wheel next. It is distribution content, and the command below is what it is there for.

```bash
anag config list                  # every setting, its value, and where it came from
anag config get project_config_path
anag config set project_config_path /media/library
anag config unset project_config_path      # back to the default
anag config path                  # $EDITOR "$(anag config path)"
```

**There are exactly two settings**, and they are the two in `defaults.yaml`:

| | |
|---|---|
| `project_config_path` | the folder holding cache, memory and the evaluator |
| `embedding_models_path` | the downloaded weights, kept outside it and shared |

Everything else follows, and is not listed or settable:

| Where it lands | What it holds |
|---|---|
| `<project_config>/cache/` | derived data, rebuilt on demand |
| `<project_config>/memory/` | your ratings, the one thing here that cannot be regenerated |
| `<project_config>/models/` | the evaluator trained on them |
| inside the installed package | the media-type taxonomy: extension lists and tag vocabularies |

The first three are not settable beside the project folder because that is how you end up with a memory folder belonging to one library and a cache belonging to another, which is silent and hard to notice. The taxonomy is not settable at all: it is found relative to the package, since the tag vocabularies are precisely what the caches and the trained evaluator were built against.

`list` prints the **source** of every value, not just the value:

```
embedding_models_path  /srv/shared-models
                       (user config)
project_config_path    /media/library
                       (user config)
```

Only the two, because a setting you cannot set is noise in the output of a command whose job is to say what you can change. Where the rest end up is the table above, and it does not vary.

That column is the point of the command. "It is not taking effect" is always a question about which layer won, and the answer is invisible unless something prints it. `anag config list --project-config /tmp/x` answers the other useful question (what *this* invocation would use) by showing that row as `(command line)`.

An unknown name is an error rather than a line in the file that nothing reads, including the derived paths above. Model names and generation settings describe what the engine *does* rather than where a machine keeps things; those belong to a project and are passed with `--config`.

### Which layer wins

Lowest first, each merged over the last, key by key:

| | Source | Set by |
|---|---|---|
| 1 | `defaults.yaml` inside the package | us; read-only |
| 2 | the media-type taxonomy | us; not configurable |
| 3 | `~/.config/anagnorisis/config.yaml` | **you, once** |
| 4 | files passed with `--config`, in order | per invocation |
| 5 | `--project-config` / `--models` | per invocation |
| 6 | `overrides=` | library callers |

The application sits at levels 5-6: it computes all of its paths and passes them explicitly, so it never inherits the settings of whoever launched it, which is what you want when it runs in a container where `$HOME` belongs to root.

From a library, `load_config` reads that same file by default, so a script sees the library you configured on the command line:

```python
from anagnorisis_core.config import load_config

cfg = load_config()                       # honours ~/.config/anagnorisis/config.yaml
cfg = load_config(project_config_path='/media/library')   # or say it outright
cfg = load_config(use_user_config=False)  # ignore it entirely
```

Turn it off for an application that carries its own configuration and must not inherit the settings of whoever launched it, which is why the Flask app builds its configuration itself rather than calling this at all.

## Installing it on the machine itself

No virtual environment, no container. The command lands in `~/.local/bin`, which is already on `PATH` on most systems, and the package goes into your user site-packages, so no `sudo` either:

```bash
pip install --user -e /path/to/Anagnorisis/anagnorisis_core
```

`-e` keeps it pointing at the repository, so pulling a new commit updates the command with it. Installing the built wheel instead works the same and needs no checkout at all.

Two things to get right first, both of which will otherwise waste an hour.

### Install torch yourself, from the index matching your driver

This is the one that really matters. `anagnorisis-core` depends on plain `torch`, and plain `torch` on PyPI is now a **CUDA 13** build. On a machine whose driver only speaks CUDA 12 it installs perfectly and then fails the moment it touches the GPU. Check what the driver supports (`nvidia-smi`, top right, "CUDA Version") and install the matching build *before* the package:

```bash
# driver capping at CUDA 12.4, which is what this project is tested against
pip install --user torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

pip install --user -e /path/to/Anagnorisis/anagnorisis_core
```

The second command then finds torch already satisfied and leaves it alone. Confirm it took, before doing anything slow:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# want: 2.6.0+cu124 True
```

### Know what it replaces

A `--user` install upgrades packages already in your user site-packages, and this one needs recent versions of two big ones: `torch` and `transformers>=5`. If something else of yours depends on the versions you have now, it will be looking at different ones afterwards. Check first, so it is a decision rather than a surprise:

```bash
pip list --user | grep -Ei "^(torch|transformers|numpy|accelerate) "
```

If that list matters to you, this is the case where a virtual environment earns its keep: `python3 -m venv`, the same two commands with the venv's pip, then a symlink from `~/.local/bin/anag` to the venv's `anag` to keep the short name. Otherwise a plain `--user` install is simpler and behaves identically.

One smaller trap nearby: PyFilesystem2 imports `pkg_resources`, which **setuptools 84 removed**. If `import fs` starts failing with `ModuleNotFoundError: No module named 'pkg_resources'`, that is why. `pip install --user "setuptools<81"` puts it back.

`ffmpeg` must be on `PATH` for video frame extraction. Nothing else is needed outside Python.

## Pointing it at an existing checkout

With nothing configured it already works, on its own defaults. To share an application checkout's cache, ratings and weights instead of building a second of each, set the two paths once, see [Configuring a machine](#configuring-a-machine):

```bash
anag config set project_config_path /path/to/Anagnorisis/project_config
anag config set embedding_models_path /path/to/Anagnorisis/models
```

For a one-off run against somewhere else, the same two are flags, and they come *after* the subcommand:

```bash
anag describe /my/media \
    --project-config /somewhere/else/project_config \
    --models         /somewhere/else/models
```

One thing to know about sharing a cache with the containers: entries are keyed by the file's URL, and a folder bind-mounted at `/work/media` in Docker is `/media/...` on the machine. The per-file embeddings therefore do not carry over between the two. The tag vocabularies do: they are keyed by vocabulary and model, not by path, and they are the expensive part.

---

### `describe`: write descriptions for files that have none

```bash
anag describe /my/media                    # one pass over the folder, then exit
anag describe .                            # or just the folder you are standing in
```

Descriptions go into the cache, and nothing outside it is touched. Files that already have a description are skipped, so a second run does nothing and costs nothing, and that is also how you resume after stopping one.

```bash
anag describe /my/media --sink meta        # write a <file>.meta sidecar instead
anag describe /my/media --watch            # stay running; describe new files as they land
anag describe /my/media --watch --interval 60
anag describe /my/media --no-stamp         # omit the provenance line from sidecars
anag describe /my/media --batch-size 0     # embed everything, then describe everything
```

`--sink meta` is the sharing form: it writes a `<file>.meta` next to each file, so the descriptions travel with the media when the folder is copied and a peer can read them without any of this installed. **A sidecar that already exists is never overwritten**, so anything you type into one stays. To regenerate a description, delete its `.meta` and run again. It is not the default because it writes into your media folders, which is not something a command should do unasked.

`--batch-size` trades model swaps against how soon sidecars appear: each batch costs one load of each model, so bigger is faster overall while smaller starts producing files sooner. `0` means do not interleave at all.

This is the slow command: seconds per file, and a minute and a half per video. See *What is slow, and why* below before pointing it at a large library.

### `index`: embed files so they can be searched

```bash
anag index /my/media
anag index /my/media --no-content          # descriptions only (metadata search)
anag index /my/media --no-metadata         # file content only (semantic search)
```

Builds the two indexes: one from each file's content, one from its description text. **Safe to interrupt**: every embedding is written as it is produced, so stopping loses only the file in flight and running it again resumes.

Note that `index` never *generates* a description; it embeds whatever text already exists, from the cache or from a `.meta`. Run `describe` first if you want the model's sentences in the metadata index.

### `search`: find files

```bash
anag search "quiet acoustic recording" /my/media
anag search "winter" /my/media --mode name
anag search "a cat on a windowsill" /my/media --mode semantic
anag search "live at the roadhouse" /my/media --mode metadata --limit 5
```

The query comes first, then the folders. Three modes:

| Mode | Matches against | Needs an index |
|---|---|---|
| `name` | the filename and path, fuzzily | no |
| `semantic` | the file's own content | `index` |
| `metadata` | the file's description: tags, the model's sentences, internal metadata, `.meta` | `index` |

`metadata` is the default. Searching never uses the GPU, since the query is embedded on the CPU, so it is safe to run while a `describe` is going.

### `rate`: record what you think of a file

```bash
anag rate "/my/media/Music/Tenshun - Glass Harp.flac" 8.5 \
    --memory ~/Desktop/Github/Anagnorisis/project_config/memory
anag rate "/my/media/photo.jpg" 3 --memory ... --describe
```

The rating is 0 to 10. This writes a memory file holding the rating and the file's description; memory files are what the evaluator trains on, and they outlive the file being renamed, moved or deleted. `--describe` runs the descriptor now so the memory captures it, rather than recording whatever is already cached.

**`--text` rates words instead of a file**, and is the same act: the model's subject is descriptions, not files, so anything you can describe you can rate:

```bash
anag rate --text "slow acoustic recordings with close-mic vocals" 9
anag rate --text - 2 < review.txt
```

Training cannot tell the two apart: it strips the rating from line 1 and embeds the rest, exactly as it does for a rated file. Use it for opinions that have no file: what you are in the mood for, a review you agree with, a genre you cannot stand. A text memory is named by a hash of the text rather than by a soft hash, since there is no file that could be renamed, so rating the same passage twice replaces the earlier record instead of teaching it twice under two names.

### Short text is kept, long text is compressed

Text shorter than `omni.text_verbatim_max_chars` (4,000 by default) is used exactly as it stands. Summarising it would spend a model run to produce something *less* informative, since the words themselves are what a search should match and what the evaluator should learn from. Only once text is too long to be taken in whole does compressing it start to pay.

The rule applies wherever text arrives, and that consistency is the point:

| | short | long |
|---|---|---|
| `rate --text` | stored as it stands | summarised, then stored |
| `score --text` | scored as it stands | summarised, then scored |
| a `.txt` being described | kept as `# Content:` | summarised as `# Automatic description:` |

If rating stored a summary while scoring read the raw text, the number would be answering a question about something the model was never taught. And a short text file now costs no model time at all. It used to spend a full generation on a paragraph, only for the assembled description to discard the result.



### `score`: what would the model make of this?

```bash
anag score /my/media/track.flac                      # score a file
anag score --text "a quiet acoustic recording"       # or just words
anag score --text - < review.txt                     # - reads standard input
# 4.76
```

Prints the rating the trained evaluator predicts. A file is scored from its assembled description, using only what is cached, so no descriptor is loaded and nothing is generated. Text is scored from the words themselves.

Either way the number comes from the same model and the same embedding, so it is comparable with what `sort --predicted` gives. That comparability is the point: it is a way to ask the model what it has learned, in your own words, without hunting for a file that happens to match.

Needs a trained evaluator; without one it tells you to run `anag train`.

### `sort`: list files by rating, best first

```bash
anag sort /my/media --memory ~/.../project_config/memory
anag sort /my/media --memory ~/.../project_config/memory --predicted --limit 20
```

Without `--predicted` you get only the files you have actually rated. With it, the trained evaluator scores the rest as well, and your own ratings always win over its guesses. `--personal-models` says where the trained evaluator lives, if it is not under `--models`.

### `train`: fit the evaluator to your ratings

```bash
anag train --memory ~/Desktop/Github/Anagnorisis/project_config/memory
anag train --memory ... --time-budget 600      # stop after ten minutes
anag train --memory ... --max-steps 2000
```

Takes no paths: it learns from the memory files, not from the library, and needs no database. Rate some files first: with nothing to learn from there is nothing to fit.

---

## How it works

### One file, one description

Everything here turns on a single idea: **a file is represented by a paragraph of English.** Not by a vector, because vectors are not portable: two people running different models produce numbers that cannot be compared. A sentence can be read by anyone, edited by hand, stored in a text file next to the media, and copied between machines. So the description is the project's universal currency.

A description is assembled from up to five parts, in this order:

```
File Name: Second Winter (reprise).mp3
File Path: /mnt/media/music/marrow-lantern/Second Winter (reprise).mp3

# Automatic description:
An instrumental piece with piano and strings building to a driving rhythm...

Tags (audio): indie rock, piano, building intensity, male vocals
Fingerprint: qhkkfmnlqp bqfjdlrmst ...

# Internal metadata:
artist: Marrow Lantern
album: Quiet Ordinance
duration: 288.4

# External metadata from 'Second Winter (reprise).mp3.meta' file:
The song my brother played on repeat the summer he moved out.
```

| Part | Where it comes from | Cost |
|---|---|---|
| Name and path | the filesystem | free |
| Automatic description | the descriptor model reading the file | seconds |
| Tags and fingerprint | derived from the file's content embedding | free once embedded |
| Internal metadata | EXIF, ID3, duration, resolution | milliseconds |
| `.meta` sidecar | a text file you (or a server) wrote | free |

The **fingerprint** is the odd one. It is a rank-quantised projection of the file's embedding rendered as pronounceable nonsense. It carries no meaning to a human, but two files that are similar to the model get similar fingerprints, so when the text is embedded, some of that similarity survives even for a reader using a different model. It is how a vector smuggles itself through a text-only channel.

**There is exactly one description of a file, and every consumer uses it verbatim.** What metadata search embeds, what the app shows as "full search description", and what a memory file stores are the same string. This used to be assembled in several places and they drifted apart, which made search results look random for weeks. `tests/test_description.py` now asserts they agree.

The one sanctioned variation is the data server's: it omits the path (which would publish a server's directory layout) and the `.meta` section (because it is writing that file).

### Two indexes, because there are two questions

- **Content embedding**: the file itself through the model. Answers *"what does this look/sound like?"* Only local files: embedding a remote file would mean downloading it.
- **Description embedding**: the text above through the model. Answers *"what do we know about this?"* Works for remote files too, because a `.meta` sidecar is cheap to fetch where the file is not.

Both live in a cache keyed by the file *and* the model that produced the vector, so swapping models invalidates the old entries instead of silently mixing incomparable numbers.

### Two rules that are never broken

**Searching never uses the GPU.** Your query is embedded on the CPU, in-process. The GPU belongs to background work you can see and stop. A search therefore never waits behind indexing, and never triggers it: a file that has not been indexed is simply absent from the results.

**Reading a file reads only that file.** No network requests, ever. This is not rhetorical: the embedding model *did* fetch URLs it found inside text files, because it guesses whether a string is a media reference by trying to download it. That is disabled at load, and `tests/test_embedder_no_network.py` fails if it comes back.

---

## Searching

```bash
anagnorisis search QUERY PATH... --mode name|semantic|metadata
```

Three modes, answering different questions:

### `--mode name`

Fuzzy match against the filename and path. **No model, no index, no waiting**: it works on a folder you have never touched.

```bash
anagnorisis search "second winter" /mnt/media/music --mode name
#  1.0000  /mnt/media/music/marrow-lantern/Second Winter (reprise).mp3
#  0.8500  /mnt/media/music/second-winter-sessions/ep3.mp3
```

Scoring is deliberately blunt: a substring of the filename scores 1.0, a substring anywhere in the path 0.85, and anything else is fuzzy-matched below that. It matches how people remember files: by name first, folder second.

### `--mode semantic`

Compares your query against the embedding of **the file's own content**. Finds a photo of a beach whatever it is called.

```bash
anagnorisis search "a beach at sunset" /mnt/media/images --mode semantic
```

Requires `anagnorisis index` first. Files with no content embedding are skipped, not ranked low.

### `--mode metadata` (default)

Compares your query against the embedding of **the description**. This is the only mode that can match on things the content cannot show:

```bash
anagnorisis search "the song my brother played" /mnt/media/music --mode metadata
```

No model can see that in an audio file; it is in the `.meta` sidecar somebody wrote.

Metadata search is also the mode that makes *remote* files searchable at all, because a description can be fetched where the file itself cannot. That is the application's job, though: **the command line works on local paths only** and refuses a remote URL rather than quietly downloading it. Pointing the app at a data server is how you search someone else's files.

### Why unindexed files vanish instead of ranking last

An unindexed file is *unknown*, not *irrelevant*. Scoring it zero and sorting it to the bottom would present it as a bad match, which is a lie. They score NaN and are dropped. If a search returns less than you expect, index first.

---

## Indexing

```bash
anagnorisis index PATH... [--no-content] [--no-metadata]
```

Builds both indexes. Content embeddings need the GPU and take from half a second (an image) to several seconds (a video); description embeddings are milliseconds each once the descriptions exist.

### What happens if you stop it halfway

**Nothing is corrupted, and how much is lost depends on how you stop it.**

There is no index file to truncate. The cache holds one entry per file, keyed by the file *and* the model that embedded it, so an entry either exists and is valid or does not exist. Re-running **resumes**: "already done" means "already in the cache", so finished files are skipped.

How much of the recent work survives depends on the kind of stop, because the cache buffers writes in memory and flushes them in batches:

| How you stop it | What happens |
|---|---|
| **Ctrl-C**, or the app shutting down | Clean exit runs the cache's `close()`, which flushes everything pending. Nothing is lost. |
| **`docker stop`** | Sends SIGTERM first, so a clean exit, same as Ctrl-C, provided the process is not still shutting down when the kill timer expires. |
| **SIGKILL, OOM-kill, power loss** | Entries not yet flushed are lost: up to the flush interval's worth, **five minutes by default**. |

Nothing is *corrupted* in the last case either: the lost entries simply were never written, so the next run recomputes them. If you are about to hard-kill a long indexing run and want to keep the work, stop it politely instead.

The batch in flight is otherwise the only work repeated, at most a few files.

The same is true of the app's background indexing, which is why it is safe to pause from the Task Manager mid-run.

Two things worth knowing:

- **Descriptions are separate from embeddings.** `index` embeds descriptions that already exist; it does not write new ones. `anagnorisis describe` does that, and it is much slower. Index is cheap to re-run; describing is not.
- **Order differs between the CLI and the app.** The CLI walks files in sorted order, which makes a run predictable and easy to resume. The application's background pass instead picks each batch *at random*, because it never finishes in one go: random sampling spreads early coverage across your whole collection rather than leaving the last folders untouched for days.

---

## Rating files, and memory

```bash
anagnorisis rate FILE RATING --memory /path/to/memory [--describe]
```

```bash
anagnorisis rate "/mnt/media/music/second-winter.mp3" 8.5 --memory ~/anagnorisis/memory
# wrote ~/anagnorisis/memory/2026-08-19/e45efec099d1d9f6aa33f96e96ef422e.md
```

This writes a **memory file**: your rating on line 1, the file's description underneath.

```
Rating: 8.5
File Name: second-winter.mp3
File Path: /mnt/media/music/second-winter.mp3

# Internal metadata:
file_size: 8402518 bytes
...
```

### Why a copy of the description rather than a pointer to the file

Because the file is not a reliable place to keep this. It gets renamed, reorganised, moved to another drive, deleted; if it lives on someone else's server it can vanish without warning. A rating attached to a path would quietly rot. The judgement (*this kind of thing is an 8.5 to me*) is the durable part, and it stays valid whether or not the original is still reachable.

So **memory files are an archive, not a cache.** Nothing regenerates them. They are the one thing here worth backing up.

### The filename is a content hash

Files are named `<soft_hash>.md`, where the soft hash is computed from a few sampled blocks of the file's content plus its size: 5 MiB read from a two-hour video, and over a network only those blocks are fetched. Because it describes content rather than location, **renaming or moving a file does not lose its rating.**

It is *soft* because two files sharing their sampled regions and their exact size would collide. That is the price of not reading terabytes.

### Rate the same file twice

Each rating goes into its own dated folder, so the history is kept and the most recent one wins. Changing your mind is normal and non-destructive.

### `--describe`

Without it, the memory is written from what is already known, which is instant. With it, the descriptor runs first so the memory also holds the model's sentences. Worth it for files you care about; slow in bulk.

### Why the rating is on line 1 and nowhere else

Training strips line 1 and embeds the rest. If the rating appeared lower down, the model would learn to read the score off the page instead of judging the file, and would score beautifully in training and uselessly in practice.

---

## Sorting by rating

```bash
anagnorisis sort PATH... --memory /path/to/memory [--predicted]
```

Your own ratings, best first:

```bash
anagnorisis sort /mnt/media/music --memory ~/anagnorisis/memory
#   8.50  user   /mnt/media/music/second-winter.mp3
#   6.00  user   /mnt/media/music/harrow-street-demo.mp3
```

With `--predicted`, the trained evaluator fills in everything you have not rated:

```bash
anagnorisis sort /mnt/media/music --memory ~/anagnorisis/memory --predicted
#   8.50  user   /mnt/media/music/second-winter.mp3
#   7.31  model  /mnt/media/music/unheard-b-side.mp3
#   6.00  user   /mnt/media/music/harrow-street-demo.mp3
```

The `user`/`model` column is not decoration: it is the distinction that matters. **Your ratings always take precedence.** The model is imitating you, so where you have already spoken there is nothing to predict. It only fills the gaps.

`--predicted` needs a trained evaluator and will tell you plainly if there isn't one.

---

## Training the evaluator

```bash
anagnorisis train --memory /path/to/memory [--max-steps N] [--time-budget SECONDS]
```

Reads every memory file, embeds the description below the rating line, and fits a model to predict the rating from it. **No database involved**: a folder of Markdown files is the entire training set.

```
[UniversalTrain] 3 unique rated memory entries.
[UniversalTrain] Training on 2002 samples (test set: 2).
Baseline Accuracy: 64.91%
Best Epoch: 5, Train Accuracy: 99.14%, Test Accuracy: 70.73%
```

A handful of memory files produces thousands of samples, because long descriptions are split into chunks and each chunk becomes an example.

Watch the gap between train and test accuracy. 99% against 70% means it has memorised the examples rather than learned your taste, which is expected with very few ratings; the fix is more ratings rather than more epochs. Compare test accuracy against the **baseline**: that is what you would get by guessing the most common rating, so beating it is the only evidence of having learned anything.

The loop this is all for: rate some files, train, let the model rate the rest, correct it where it is wrong. Each correction is another memory file, and the next round starts from a slightly better model.

---

## Writing descriptions

```bash
anagnorisis describe PATH... --sink cache|meta [--watch]
```

Runs the descriptor over files that have no description yet. Two destinations, and they are different kinds of thing:

- `--sink cache`: **the default.** Derived and disposable, keyed by model, belongs to one machine. What the application uses, and the one that writes nothing into your media folders.
- `--sink meta`: writes `<file>.meta` next to the file. A document that travels with the media when the folder is copied, that you can open and edit, and that is the only thing a peer ever sees. What the data server uses.

**Audio and video are sampled in three windows, not five.** Each window is described on its own and the three are folded into one description: four generations rather than six, still hearing three places in the recording.

The windows have to be *continuous*. Joining short excerpts into a single window to save a pass looks like it should work and does not: the model hears the cuts and describes the collage. Measured on one track, with the excerpts explained in the prompt. 1×30s gave *"a solo performance of a piano piece, moderate and flowing, reflective"*, while 2×15s gave *"a collection of short, fragmented musical excerpts"* and 3×10s gave *"a compilation … without listening"*. Two different tracks came out near-identical that way, which is useless for search.

**Audio is sent in fixed thirty-second windows.** The model's audio input is not a maximum but an exact length: 750 soft tokens at 40 ms each. A shorter clip produces fewer tokens and the model does not register them as sound at all, answering *"Please provide the audio you would like me to describe"* with no error raised anywhere. Every clip is therefore padded, or trimmed, to that window before it is sent. Sampled windows use the whole thirty seconds because a ten-second one costs exactly the same and carries a third of the sound; video windows stay short so the frames and the sound describe the same moment, and are padded on the way in.

**A description written without the media is thrown away.** The descriptor sometimes answers as though it never received the file: *"The audio consists entirely of requests for an audio file to be provided so that a description can be made"*, or a summary of five such refusals. This is worse than a plain failure: it is fluent, plausible, and permanent, because a `.meta` is never overwritten. Any such answer is discarded, which makes it a failure: no sidecar is written and the file is tried again on a later run. Individual segments are vetted the same way before they are folded together, so three good samples still produce a description when two of them refused.

Because both models cannot fit on an 8 GB card at once, each batch embeds, frees the embedder, describes, then frees the descriptor.

### `.meta` files belong to whoever hosts them

- **If one exists, it is left completely alone.** Never overwritten, whether a model wrote it, a cloud service wrote it, or you typed it.
- **Editing is expected**, and permanent.
- **To regenerate, delete it and run again.** That is the whole refresh mechanism: no staleness check, no force flag, nothing that can destroy your writing by accident.

With `--watch` it keeps going, describing new files as they appear. A file that is still being copied is left alone until its size and modification time hold still across two passes: describing a half-copied video would be wrong, and because `.meta` files are never overwritten, permanently so.

---

## Commands at a glance

| Command | What it does | Needs the GPU |
|---|---|---|
| `describe` | writes descriptions for files that have none | yes (descriptor) |
| `index` | embeds files and descriptions so they can be searched | yes (embedder) |
| `search` | finds files by name, content or description | no |
| `rate` | records your rating in a memory file | only with `--describe` |
| `sort` | lists files by rating, yours or the model's | only with `--predicted` |
| `score` | predicts what you would rate a file, or some text | yes (embedder) |
| `train` | fits the evaluator to your ratings | yes |
| `config` | reads and changes where things are kept | no |

Common options: `--config` (repeatable, later files win), `--project-config`, `--models`, `--no-recursive`, `--batch-size`.

## What is slow, and why

Worth knowing before you run something over a large library:

| Work | Roughly |
|---|---|
| Describing an image | ~16 s |
| Describing a text file | 0 s if short enough to keep verbatim, ~16 s if it must be summarised |
| Describing audio or video (three sampled windows) | 47 s audio, 56 s video |
| Embedding a file's content | 0.6 s (image) to 3.8 s (video) |
| Embedding a description | milliseconds |
| Searching, once indexed | milliseconds, on the CPU |

Two consequences:

- **Describing is the expensive step; embedding is not.** `index` is cheap to re-run and never invokes the descriptor. `describe` is the one to be careful with, which is also why a `.meta` file is never overwritten.
- **Each invocation loads the model afresh.** Loading costs seconds on a GPU and a minute or more on CPU-only hardware, so one command over a folder is far faster than a command per file. For continuous work use `describe --watch`, which loads once and stays.

Both models cannot be resident on an 8 GB card at the same time: the embedder needs about 4.3 GB and the descriptor about 6.8 GB at peak, so anything needing both alternates between them, freeing one before loading the other.

## Finding the files is its own cost

Before anything can be searched, the files have to be found, and on a large or slow-mounted library that can cost more than the search itself.

Listing goes through a **cached directory walk**, used by the application and the command line alike: each directory's listing is cached and keyed by that directory's modification time, so a folder is re-read only when it actually changes. Nothing has to be invalidated by hand: adding or removing a file bumps its directory's mtime, and the next walk notices.

On the measured demo set that is ~10 µs per file warm, against ~54 µs for a genuine uncached scan: about a second per 100k files either way, so the cache saves roughly five times, not the eighty times an earlier measurement suggested. That earlier figure was one-time Python imports being counted as walking, which the CLI pays once per invocation and which dwarfs the walk itself.

Two consequences worth knowing:

- **Hidden directories are skipped.** Version-control folders, thumbnail caches and trash are not user content, and with the `.meta` sink, indexing them would mean writing sidecars into them.
- **A listing can be briefly stale on filesystems that do not report mtime.** There the cache falls back to a key that changes every 10 seconds, so a new file can take that long to appear. Local disks report mtime and are unaffected.

### Files are addressed as URLs

Everything here refers to a file by its VFS URL: `osfs:///mnt/media/cat.jpg` rather than `/mnt/media/cat.jpg`. One spelling, whether the file came from the command line, the application, or a remote server, so nothing has to guess which form it has been handed.

Writers resolve the scheme away at the last moment: a `.meta` sidecar is a real file on a real disk, so `MetaSink` converts to a local path before opening it. For local files that conversion copies nothing.

## Benchmarks

Search speed and quality are tracked over time in [`benchmarks/README.md`](benchmarks/README.md), measured by `benchmarks/search_benchmark.py`. Each mode is timed separately, cold (cache read from disk) and warm (cache in RAM), with model loading excluded and reported on its own. Quality is measured by self-retrieval, whether a file's own description finds that file first, because an engine that returns nothing returns it very quickly.

## Running as a non-root user

Worth doing: the `.meta` files a run writes are owned by whoever wrote them, and as root that means you need `sudo` to delete one and regenerate it.

```bash
docker run --user $(id -u):$(id -g) -e HF_HOME=/your/cache/hf ...
```

Three things break as a non-root user unless they are handled, and all three are handled now:

- **A writable model cache.** `HF_HOME` must point somewhere writable, or transformers tries to write to `/root`.
- **A writable JIT cache.** librosa compiles with numba's `cache=True`, and numba needs to write that cache next to librosa's own source or into a per-user cache directory, neither of which a non-root container user can touch. It then fails with `cannot cache function '__o_fold': no locator available`, a traceback that points at librosa and never mentions permissions, and **every audio file fails to be described**.
- **A writable CUDA kernel cache.** torch JIT-compiles some CUDA kernels and wants to cache them under `$HOME/.cache/torch/kernels`. With no home directory it warns that the *specified kernel cache directory could not be created* and carries on with caching disabled, so unlike numba this one costs time rather than failing, which is why it went unnoticed for longer.

Both workers call `ensure_writable_cache_dirs(cfg)` before importing librosa or touching CUDA, which points `NUMBA_CACHE_DIR` and `PYTORCH_KERNEL_CACHE_PATH` at subdirectories of the cache. An explicit value in the environment always wins, so a deployment can still place them wherever it likes.

## Configuration

The engine reads a deliberately small set of settings:

| Setting | What it is |
|---|---|
| `main.project_config_path` | **settable**; the one folder holding the three below |
| `main.embedding_models_path` | **settable**; downloaded weights, kept outside it |
| `main.cache_path` | always `<project_config>/cache` |
| `main.memory_path` | always `<project_config>/memory` |
| `main.personal_models_path` | always `<project_config>/models` |
| `main.media_types_path` | always the copy inside the package |
| `embedder.*` | the embedding model and how it is applied |
| `omni.*` | the descriptor model and its prompts |

Two inputs, everything else derived, whatever a config file may say about the derived ones. `load_config` takes exactly those two:

```python
load_config(project_config_path=..., models_path=...)
```

Everything else in the application's `config.yaml` (modules, ports, secrets) is none of the engine's business. The application does not use `load_config`; it builds its own configuration and names every path itself.

**Media types** (which extensions count as images, which tag vocabularies apply) ship at `anagnorisis_core/media/media_types/` and are not configurable, because the tag vocabularies are precisely what the caches and the trained evaluator were built against. Editing them there changes each vocabulary's hash, which invalidates every cached tag list, so batch such edits.

---

## Using it as a library

The CLI is argparse over `api.py`; call the same functions directly:

```python
from anagnorisis_core import api
from anagnorisis_core.config import load_config

cfg = load_config(['config.yaml'], project_config_path='/var/lib/anagnorisis',
                  models_path='/var/lib/anagnorisis/models')

api.index(['/mnt/media/images'], cfg=cfg)

# Arguments may be plain paths or URLs; results are always URLs.
hits = api.search('a beach at sunset', ['/mnt/media/images'], cfg=cfg, mode='semantic')
# [('osfs:///mnt/media/images/dsc_0021.jpg', 0.71), ...]

api.rate('/mnt/media/images/dsc_0021.jpg', 9.0, cfg=cfg,
         memory_dir='/var/lib/anagnorisis/memory')
api.train_evaluator(cfg=cfg)
```

Long operations take a progress object with two methods: `check()` and `update(fraction, message)`. The application passes its task context so the work is pausable from the Task Manager; the CLI passes one that prints; tests pass nothing. The engine never learns who is listening.

```python
from anagnorisis_core.progress import PrintProgress
api.index(['/mnt/media'], cfg=cfg, ctx=PrintProgress())
```

### The boundary

Nothing here imports Flask, SQLAlchemy or socketio, and `tests/test_core_boundary.py` fails if that changes. Ratings and play counts live in the application's database and are passed in; the engine persists nothing but its caches and the memory files you ask it to write.

That is what lets the data server run the engine with no web application present at all.
