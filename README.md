<!--[![Join on Reddit](https://img.shields.io/reddit/subreddit-subscribers/Anagnorisis?style=social)](https://www.reddit.com/r/Anagnorisis)-->

# Anagnorisis
[![Anagnorisis Health](https://oss-health-monitor.vercel.app/api/badge/volotat/Anagnorisis?v=2)](https://github.com/volotat/OSS-Health-Monitor)

Anagnorisis - is a local recommendation system that runs on your machine, learns your taste from your own ratings, and belongs to you.
It allows you to fine-tune models on your data to predict your data preferences. You can feed it as much of your personal data as you like and not be afraid of it leaking as all of it is stored and processed locally on your own computer. Targeted for a PC or Home server with at least 8GB VRAM GPU on the board. 

## Motivation
Every recommendation system you typically use from the cloud services (i.e. Youtube, Spotify, Tiktok, Twitter (X), Facebook and so on) is owned by someone whose interests aren't yours. It optimizes for engagement and other private metrics, you can't inspect, can't correct, and can't take with you when the service inevitably dies or even use the same recommendation model for different independent applications. This project aims to solve all of that.

Anagnorisis has the other arrangement that turns the whole client-server architecture upside down. The servers are now pure data-sharing thin hosts and all the main processing, search and recommendations are happening on your machine. You rate local or remote files and other data you own on a scale of 0 to 10. This feedback then used to locally train a recommendation model that scores everything you haven't rated yet automatically. You correct what it got wrong, that correction goes to the next round of training data. You repeat these steps again and again, getting each time model that better and better aligns to your preferences.  

The big vision of this project is to provide a platform that creates a local, private model of your interests. That likes what you like and sees importance where you would see it. Then you can use this model to search and filter local and global information on your behalf in a way you would do it yourself but in a much faster and efficient way. Making this platform (in the future) a go to place to see news, recommendations and insights, and so on, tailored specifically for you. As the internet gets populated with bots and AI slop, a platform like this might create a necessary filter to be able to navigate in this chaotic information space effectively.

To find more about the project and ideas behind it you can read these articles:  
[Anagnorisis. Part 1: A Vision for Better Information Management.](https://volotat.github.io/p/anagnorisis-part-1-a-vision-for-better-information-management/)  
[Anagnorisis. Part 2: The Music Recommendation Algorithm.](https://volotat.github.io/p/anagnorisis-part-2-the-music-recommendation-algorithm/)  
[Anagnorisis. Part 3: Why Should You Go Local?](https://volotat.github.io/p/anagnorisis-part-3-why-should-you-go-local/)  
[Anagnorisis. Part 4: File Sharing is All We Need.](https://volotat.github.io/p/anagnorisis-part-4-file-sharing-is-all-we-need/)

And watch these videos:  
[Anagnorisis: Search Your Data Effectively (v0.3.1)](https://www.youtube.com/watch?v=X1Go7yYgFlY) - How to effectively search your data across all modules.  
[Anagnorisis: Music Module Preview (v0.1.6)](https://www.youtube.com/watch?v=vux7mDaRCeY) - Presentation of 'Music' module usage. To see how the algorithm works in details, please read this wiki page: [Music](wiki/music.md)  
[Anagnorisis: Images module preview (v0.1.0)](https://www.youtube.com/watch?v=S70Lp0oL7aQ) - Presentation of 'Images' module usage. Or you can read the guide at the [Images wiki](wiki/images.md) page.  

## How search works

There are three search modes, that are used for different kind of search:

- **Filename-based search** is a typical fuzzy-matching search that compares your query with file names, both local and remote.
- **Content-based search** embeds the *file itself* with the embedding model and compares embedding of your query with the embeddings of files to find the best matches. Only local files are supported for that kind of search to avoid unsolicited downloads from the remote servers. 
- **Metadata-based search** embeds a *text description* of the file: its name and path, an automatic description from the descriptor model (only local), zero-shot tags (only local), internal metadata (EXIF, ID3 tags, etc.) (also only local), and the contents of its `{filename}.meta` sidecar (local and remote in case such file exists). This special file allows you to describe local files subjectively (i.e. "photo of MY grandpa") and have a completely personalized search because of it. For the remote files it acts as a lightweight proxy for an actual file content, allowing for semantic search and recommendation in distributed networks.

Two rules the project holds to:

- **Searching never uses the GPU.** Your query is embedded on the CPU, in-process. The GPU is used only by background tasks you can see and pause on the Task Manager page.
- **Remote files are never downloaded automatically.** Background indexing reads only local files. Only `.meta` files are read automatically from the remote servers as it is serves exactly as lightweight indexing proxy for actual files.

Because searching reads from the index rather than building it, files that have not been indexed yet simply do not appear in results. The status bar reports how many are still pending.

The `.meta` sidecars provides a distributed semantic index with no protocol. Just text files next to the main files is enough to search and recommend data from the remote servers with arbitrary data on them.

## How memory works

Every time you rate a file, the project writes a small Markdown file recording everything it knew about that file at that moment. They accumulate in `project_config/memory/<date>/<soft-hash>.md` and are the material the recommendation model is trained on.

A memory file holds the rating on its first line, then the file's name and path, the zero-shot tags and fingerprint from the embedding model, the descriptor model's description of the content, internal metadata (EXIF, ID3 and so on), and the contents of the file's `.meta` sidecar if it has one. In other words, a written account of the file with text as a proxy of its content.

**Why keep an proxy instead of pointing at the file:** Because the file path is not a reliable place to keep account of. It gets renamed, reorganised, moved to another drive, deleted; if it lives on someone else's server it can disappear without warning. A rating attached to a path would quietly rot. Your score stays valid whether or not the original file is still reachable.

The file is identified by a **soft hash**: a fingerprint computed from a few sampled blocks of its content plus its size, rather than the whole file. It is fast even on large files and cheap over a network, and because it describes content rather than location, moving or renaming a file does not lose a memory about rating the file. Memory files are grouped in dated folders, and when the same file has been rated more than once the most recent entry wins, so re-rating something supersedes your earlier opinion instead of contradicting it. In future It might also allow to track the change in the preferences over time and build even better time-dependent recommendations.

Rating a file is **an explicit action you took**, which is why this is the one case that allows to fetch a remote file, so the project automatically downloading it to describe it properly before saving into a memory. Background indexing never does this. The description and memory event creation happens as a background task and shown in the tasks manager UI.

**How the recommendation model uses it:** Training walks the memory folder and reads each file as a pair: the description, and the rating you gave it. The rating line is stripped before the description is embedded, so the model never sees the score it is being asked to predict, it has to learn from the content, not read the answer off the page.

What comes out is an evaluator model that could predict a rating for a file it has never seen before, stored as its `model_rating` in the DB for fast access. Sorting and recommendation then use your own rating where you have given one and fall back to the model's guess where you have not, so the ranking is your judgement wherever it exists and the model's imitation of it everywhere else. Some modules, such as the `Music` module additionally folds in play counts, skips and how long ago something was last played to build better recommendation list.

This is what creates the loop described at the top of this README: rate some files, train, let the model rate the rest, correct it where it is wrong. Each correction becomes another memory file, and the next round starts from a slightly better model.

## The standalone core engine

The part that describes, embeds and searches files lives in its own library, [`anagnorisis_core`](anagnorisis_core/README.md), with no web server, no database and no browser just an API served as a Python package with CLI on top. The application is a front end over it. Three things use this code: the web application, the `anagnorisis` CLI, and the data server. 

After installing the package the core functions of the app are available as a command line interface with `anag`, or `anagnorisis` pretext.
Say once where things are kept, and the rest needs no paths:

```bash
anag config set project_config_path ~/Desktop/Github/Anagnorisis/project_config
anag config set embedding_models_path ~/Desktop/Github/Anagnorisis/models

anag describe /mnt/media/images                  # write descriptions for files
anag index    /mnt/media/images                  # embed files and their descriptions them so they can be searched
anag search   "a quiet street at night" /mnt/media/images --mode semantic
anag rate     /mnt/media/images/dsc_0021.jpg 9   # save the given score to the memory
# or: anag rate --text "grainy night photos" 9
anag score    /mnt/media/images/scr_0091.jpg     # what the model predicts you would rate that  
# or: anag score --text "a quiet street at night" 
anag sort     /mnt/media/images --predicted
anag train
```

Pointing `project_config_path` at the application's own `project_config/` is what makes the two share one cache, one set of ratings and one trained model, the command line and the app then see the same library. That is also how the data server annotates a shared folder without running any of the web application. See [`data_server/README.md`](data_server/README.md) for the details.

The main goal of the core package is to have single independent reusable peace of code that could be used to create arbitrary applications on top of it that are still share the same memory and recommendation model.

## Benchmarks

The benchmarks exist for tracking purpose. Note that only minor work has been done on speed, as the main goal of the project so far was to establish the right infrastructure to build upon. 

Here only a subset of benchmarks are present, see [`anagnorisis_core/benchmarks/README.md`](anagnorisis_core/benchmarks/README.md) for more information and over time tracking.

**Finding the files:**

| Regime | Time | µs/file | Est. @100,000 |
|---|---|---|---|
| uncached (cache empty) | 0.0191s | 43.61 | 4.4s |
| cold (cache on disk) | 0.0051s | 11.56 | 1.2s |
| warm (cache in RAM) | 0.0046s ±0.0001 | 10.42 | 1.0s |

**Building the index:**

| Media type | Files | Content s/file | Descriptions s/file |
|---|---|---|---|
| `audio` | 205 | 0.4285 | 0.4505 |
| `images` | 202 | 0.2144 | 0.4409 |
| `text` | 17 | 0.0833 | 1.4209 |
| `videos` | 15 | 1.5251 | 1.6546 |

| Phase | Total | s/file | Est. @100,000 |
|---|---|---|---|
| content | 155.45s | 0.3541 | 35,425s (9.8h) |
| descriptions | 230.39s | 0.5248 | 52,496s (14.6h) |
| **both, whole library** | — | — | **87,920s (24.4h)** |
| re-index, everything already cached | 0.012s | 0.03 ms | 2.8s |

Embedder load, measured once and excluded from above: 14.74s. The *descriptions* row also carries a one-off computation: indexing embeds the tag vocabularies up front, about 9,000 tags. It is paid once, and the whole-library figure includes it. 

**Writing the descriptions:**

| Media type | s/file (mean) | median | min–max | n | All 100,000 of this type |
|---|---|---|---|---|---|
| `audio` | 19.93 | 20.42 | 18.92–20.43 | 3 | 553.5h |
| `images` | 14.26 | 14.13 | 13.99–14.66 | 3 | 396.2h |
| `text` | 4.54 | 0.0 | 0.0–13.61 | 3 | 126.1h |
| `videos` | 23.68 | 23.33 | 23.12–24.60 | 3 | 657.9h |

Descriptor load, once: 11.66s.


## Running from Docker
The preferred way to run the project is from Docker. This should be much more stable than running it from the local environment, especially on Windows.

1. Make sure that you have Docker installed. In case it is not go to [Docker installation page](https://www.docker.com/get-started/) and install it. 
2. Clone this repository:
    ```bash
    git clone https://github.com/volotat/Anagnorisis.git
    cd Anagnorisis
    ```
3. Create your configuration file from the provided example:
    ```bash
    cp docker-compose.override.example.yaml docker-compose.override.yaml
    ```
4. Open `docker-compose.override.yaml` in any text editor and replace the placeholder paths with your actual folder paths. For example:
    ```yaml
    volumes:
      # Project config (database, trained models, cache)
      - /home/user/Anagnorisis-config:/mnt/project_config

      # Your image folders:
      - /home/user/Photos:/mnt/media/images/Photos

      # Your music folders:
      - /home/user/Music:/mnt/media/music/Music

      # Your text folders:
      - /home/user/Documents:/mnt/media/text/Documents

      # Your video folders:
      - /home/user/Videos:/mnt/media/videos/Videos
    ```
    Each line follows the format: `/path/on/your/computer:/mnt/media/TYPE/LABEL`  
    - Use **absolute paths** (starting with `/` on Linux/Mac, or `C:/` on Windows).  
    - `TYPE` is one of: `images`, `music`, `text`, `videos`.  
    - `LABEL` is any name you choose — it will appear as a folder name in the app.  
    
    **Only the folders you list here will be accessible from inside the container.** No other folders on your system can be reached.

5. Launch the application:
    ```bash
    docker compose up -d
    ```
    Note: if you are using Docker Desktop you have to explicitly provide access to your data folders in the Docker settings. To do so, go to Docker Desktop settings, then to Resources -> File Sharing and add the paths to your data folders.
6. Access the application at http://localhost:5001 (or whichever port you configured) in your web browser.
7. To stop the application:
    ```bash
    docker compose down
    ```

Your configuration in `docker-compose.override.yaml` is preserved between restarts. You only need to edit it once.

### Multiple Media Folders Per Module

You can mount **as many folders as you need** for each media type. Each folder will appear as a separate top-level folder in the app's file browser. For example, to add multiple image sources:

```yaml
volumes:
  - /home/user/Anagnorisis-config:/mnt/project_config
  
  # Multiple image sources:
  - /home/user/Photos:/mnt/media/images/Photos
  - /media/external/DCIM:/mnt/media/images/Phone
  - /home/user/Screenshots:/mnt/media/images/Screenshots

  # Multiple music sources:
  - /home/user/Music/MyCollection:/mnt/media/music/MyCollection
  - /media/external/Vinyl:/mnt/media/music/Vinyl

  # ...
```

Inside the app, the Images module would show three top-level folders: `Photos`, `Phone`, and `Screenshots`, each containing the files from the corresponding folder on your computer. All search, sorting, and recommendation features work across all folders seamlessly.

### Running Multiple Instances

You can run several Anagnorisis instances simultaneously (e.g. for different family members) using separate configuration files. See the `instances/` folder for examples.

1. Copy an example and customize it:
    ```bash
    cp instances/example-personal.yaml instances/personal.yaml
    ```
2. Edit `instances/personal.yaml` with your paths, a unique port, and a unique container name.
3. Start and stop with the `-f` flag:
    ```bash
    docker compose -f docker-compose.yaml -f instances/personal.yaml up -d
    docker compose -f docker-compose.yaml -f instances/personal.yaml down
    ```

Each instance needs a **unique project name** (the `name` key at the top of the file), a **unique container name**, a **unique port**, and its **own project config folder** (for separate databases and trained models). You can run as many instances as your hardware supports.

## Initialization

To avoid issues with corrupted models being downloaded, **be patient while the application is initializing for the first time**. All models are quite large and might take some time to download depending on your internet connection speed. You can check the progress in the `logs/{CONTAINER_NAME}_log.txt` file that will appear in the project's root folder. The project UI will also show the initialization status, but  for now without download progress percentages. 

If for some reason the initialization process is interrupted (for example you stopped the container while models were being downloaded), upon the next start the application will check for corrupted models and try to re-download them automatically. If this does not help, please delete the `models` folder inside the project's root folder and start the application again. This will force the application to download all models from scratch. 

## Troubleshooting

In case you encounter an error like this:
```
ERROR: for {your container name} Cannot start service anagnorisis: error while creating mount source path '/path/to/config': chown /path/to/config: operation not permitted
```

You have to create the folder specified as your project config mount target (the path before `:/mnt/project_config` in your `docker-compose.override.yaml`) manually on your host machine. Docker sometimes cannot create such folders by itself due to permission issues.

## Additional notes for installation
The Docker image (Python dependencies, PyTorch with CUDA runtime, system libraries) takes approximately 8 GB of disk space after building. On first startup the application downloads the required ML models that would take roughly 14 GB, most of it the descriptor model. If you use the project heavily with active use of external modules, their caches can grow to several additional gigabytes. As a rough total estimate, budget around 20-25 GB of free disk space before starting.

For best user experience I would recommend running the project with relatively modern Nvidia GPU with at least 8Gb of VRAM and 32Gb of RAM. At least this is the configuration I am using myself. However, the project should be able to run on lower configurations, but performance might be poor especially without CUDA-friendly GPU. Note that CPU-only mode might be significantly slower.

After initializing the project, you will find new `database` folder inside of the project config folder you specified. In this folder project's database, migrations, models and configuration file will be stored. After running the project for the first time, the `database/project.db` file will be created. That DB will store your preferences, that will be used later to fine-tune evaluation models. Try to make backups of this file from time to time, as it contains all of your preferences, and some additional data, such as playback history.

If you have a lot of data in your data folder, for the first time hash cache and embedding cache will be gathered. Please be patient, as it may take a while. The percentage of the progress will be shown in the status bar.

The project requires GPU to run properly. When running the project inside the Docker container, make sure that `NVIDIA Container Toolkit` is installed for Linux and `WSL2` for Windows.


## Modules

The application is built around a module system. Each module is a self-contained folder inside `modules/` that adds support for a new data type or functionality. Modules are **auto-discovered at startup** — dropping a module folder in and restarting is all that is needed to activate it.

Built-in modules: **Images**, **Music**, **Text**, **Videos**, **Train** (evaluator training UI).

### Installing an external module

External modules can be installed by cloning their repository directly into the `modules/` folder:

```bash
cd modules
git clone <module-repo-url>
```

Then rebuild and restart the container:

```bash
docker compose up -d --build
```

### Available external modules

| Module | Description | Status |
|--------|-------------|--------|
| [WebSearch](https://github.com/volotat/WebSearch) | Crawls and indexes websites, enabling semantic search and preference-based ranking over web content. | experimental |
| [YouTube](https://github.com/volotat/YouTube) | Treats YouTube as CDN leaving search and recommendations to Anagnorisis local algorithms. | experimental |

### Building your own module

See [`modules/_module_template/`](modules/_module_template/) for a fully documented reference implementation.


## Security notes
The project is meant to be run on the localhost only for now. The default configuration ip address is set to `127.0.0.1` inside `docker-compose.override.yaml` file. This means that the application will only be accessible from the machine it is running on. If you want to access it from other devices on your local network, you can change the port binding in your `docker-compose.override.yaml` to `0.0.0.0:5001:5001`. You can even tunnel it to the internet using services like [ngrok](https://ngrok.com/) or [cloudflare tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/). However, I would strongly recommend against exposing the service to the internet (unless you are 100% know what you are doing) as there is no proper security work has been done yet. 



## Models
The project runs two models:

[jinaai/jina-embeddings-v5-omni-small](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small) — **the embedding model**. One model for every kind of content: text, images, audio and video all land in a single shared vector space. This replaced the three separate models the project used before (CLAP for audio, SigLIP for images, Qwen3 for text), each of which had its own incompatible space.

[google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) — **the descriptor model**, which writes a natural-language description of a file. It reads images, audio, video and text. That description is what metadata search indexes and what a data server publishes about a file in its `.meta` sidecar.

All models are downloaded automatically when the project is started for the first time. This might take some time depending on the internet connection. You can see the progress inside `logs/anagnorisis-app_log.txt` file that will appear in the project's root folder if you run the project from the Docker container.

## Acknowledgments

The project uses [Flask](https://flask.palletsprojects.com/) libraries for backend and [Bulma](https://bulma.io/) as frontend CSS framework. For all ML-related stuff [Transformers](https://github.com/huggingface/transformers) and [PyTorch](https://pytorch.org/) are used. This is the main technological stack, however there are more libraries used for specific purposes.

**Huge thanks to [Dystrio](https://huggingface.co/dystrio) for optimizing the MiniCPM-o-4_5 model (used in the early days of the project) specifically for the Anagnorisis project.**

## Wiki
The project has its own wiki that is integrated into the project itself, you might access it by running the project, or simply reading it as markdown files.

Here is some pages that might be interesting for you:  
[Change history](wiki/change_history.md)  
[Philosophy](wiki/philosophy.md)  
[Music](wiki/music.md)  
[Images](wiki/images.md)  
[Roadmap](wiki/roadmap.md)

---------------	
In memory of [Josh Greenberg](https://variety.com/2015/digital/news/grooveshark-josh-greenberg-dead-1201544107/) - one of the creators of Grooveshark. Long gone music service that had the best music recommendation system I've ever seen. 
