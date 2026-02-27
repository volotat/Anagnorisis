<!--[![Join on Reddit](https://img.shields.io/reddit/subreddit-subscribers/Anagnorisis?style=social)](https://www.reddit.com/r/Anagnorisis)-->

# Anagnorisis
Anagnorisis - is a local recommendation system that allows you to fine-tune models on your data to predict your data preferences. You can feed it as much of your personal data as you like and not be afraid of it leaking as all of it is stored and processed locally on your own computer. 


The project uses [Flask](https://flask.palletsprojects.com/) libraries for backend and [Bulma](https://bulma.io/) as frontend CSS framework. For all ML-related stuff [Transformers](https://github.com/huggingface/transformers) and [PyTorch](https://pytorch.org/) are used. This is the main technological stack, however there are more libraries used for specific purposes.


To find more about the project and ideas behind it you can read these articles:  
[Anagnorisis. Part 1: A Vision for Better Information Management.](https://volotat.github.io/p/anagnorisis-part-1-a-vision-for-better-information-management/)  
[Anagnorisis. Part 2: The Music Recommendation Algorithm.](https://volotat.github.io/p/anagnorisis-part-2-the-music-recommendation-algorithm/)  
[Anagnorisis. Part 3: Why Should You Go Local?](https://volotat.github.io/p/anagnorisis-part-3-why-should-you-go-local/)  

And watch these videos:  
[Anagnorisis: Search Your Data Effectively (v0.3.1)](https://www.youtube.com/watch?v=X1Go7yYgFlY) - How to effectively search your data across all modules.  
[Anagnorisis: Music Module Preview (v0.1.6)](https://www.youtube.com/watch?v=vux7mDaRCeY) - Presentation of 'Music' module usage. To see how the algorithm works in details, please read this wiki page: [Music](wiki/music.md)  
[Anagnorisis: Images module preview (v0.1.0)](https://www.youtube.com/watch?v=S70Lp0oL7aQ) - Presentation of 'Images' module usage. Or you can read the guide at the [Images wiki](wiki/images.md) page.  

## General
Here is the main pipeline of working with the project:  
1. You rate some data such as text, audio, images, video or anything else on the scale from 0 to 10 and all of this is stored in the project database.  
2. When you acquire some amount of such rated data points you go to the 'Train' page and start the fine-tuning of the model so it could rate the data AS IF it was rated by you.  
3. New model is used to sort new data by rates from the model and if you do not agree with the scores the model gave, you simply change it.  

You repeat these steps again and again, getting each time model that better and better aligns to your preferences.  

The big vision of this project is to provide a platform that creates a local, private model of your interests. That likes what you like and sees importance where you would see it. Then you can use this model to search and filter local and global information on your behalf in a way you would do it yourself but in a much faster and efficient way. Making this platform (in the future) a go to place to see news, recommendations and insights, and so on, tailored specifically for you. As the internet gets populated with bots and AI slop, a platform like this might create a necessary filter to be able to navigate in this chaotic information space efficiently.

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
The Docker container includes Ubuntu 22.04, CUDA drives and several large machine learning models and dependencies, which results in a significant storage footprint. After the container is built it will take about 45GB of storage on your disk. 

For best user experience I would recommend running the project with relatively modern Nvidia GPU with at least 8Gb of VRAM and 32Gb of RAM . At least this is the configuration I am using myself. However, the project should be able to run on lower configurations, but performance might be poor especially without CUDA-friendly GPU. It is usually take less then 4GB of VRAM to run the project, however when training recommendation models it usually spikes up to 5-7Gb of VRAM usage.

After initializing the project, you will find new `database` folder inside of the project config folder you specified. In this folder project's database, migrations, models and configuration file will be stored. After running the project for the first time, the `database/project.db` file will be created. That DB will store your preferences, that will be used later to fine-tune evaluation models. Try to make backups of this file from time to time, as it contains all of your preferences, and some additional data, such as playback history.

If you have a lot of data in your data folder, for the first time hash cache and embedding cache will be gathered. Please be patient, as it may take a while. The percentage of the progress will be shown in the status bar.

The project requires GPU to run properly. When running the project inside the Docker container, make sure that `NVIDIA Container Toolkit` is installed for Linux and `WSL2` for Windows.

## Security notes
The project is meant to be run on the localhost only for now. The default configuration ip address is set to `127.0.0.1` inside `docker-compose.override.yaml` file. This means that the application will only be accessible from the machine it is running on. If you want to access it from other devices on your local network, you can change the port binding in your `docker-compose.override.yaml` to `0.0.0.0:5001:5001`. You can even tunnel it to the internet using services like [ngrok](https://ngrok.com/) or [cloudflare tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/). However, I would strongly recommend against exposing the service to the internet (unless you are 100% know what you are doing) as there is no proper security work has been done yet. 

## Embedding models
To make audio, visual, video and text search possible the project uses these models:  
[LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) - for audio embeddings.  
[Google/SigLIP](https://arxiv.org/pdf/2303.15343) - for image embeddings.  
[JinaAI/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) - for text embeddings.  
[MiniCPM-o-4_5](https://huggingface.co/openbmb/MiniCPM-o-4_5) - as omni-descriptor model to convert all data modalities into text descriptions. (used by `metadata-based-search`)  

All models are downloaded automatically when the project is started for the first time. This might take some time depending on the internet connection. You can see the progress inside `logs/anagnorisis-app_log.txt` file that will appear in the project's root folder if you run the project from the Docker container.

---

> [!NOTE] 
> Since verion 0.3.4 the approach to embedding generation is going to be changing dramatically. Instead of using several pretrained CLIP-like embedding models as before there are going to be only one unified omni model to convert all the data modalities into the text descriptions and then the search and recommendations will be done on the text level using text embeddings. While this approach is more demanding to the hardware, much slower and probably less accurate at the start, it allows to create a more unified and consistent search experience across all data modalities. I also believe that this approch is much more scalable with new releases of more powerful and efficient omni models. For now [MiniCPM-o-4_5](https://huggingface.co/openbmb/MiniCPM-o-4_5) is used, that with 4-bit quantization could work under 8Gb of VRAM with small but reasonable context window (unfortunately no other models, including [Gemini-3n-e2B](https://huggingface.co/google/gemma-3n-E2B) have worked for me). This approach also allows to unify semantic-based search and metadata-based search, as everything is converted into text descriptions. Even better approach would be using a [Universal-Embedding Model](https://volotat.github.io/p/there-is-a-way-to-make-training-llms-way-cheaper-and-more-accessible/) but as for my knowledge no such models yet exist.
>
> For now the old search through CLIP-like models is still available as an option in `Images` and `Music` modules, but everything in the project would be shifting towards new approach and the old semantic search will be removed in the future releases when the omni-descriptor approach is fully stable and efficient enough. 
> 
> In the future, when omni-models will be small enough and/or typical personal hardware will be powerful enough, users will be able to fine-tune the omni-descriptor model itself on their '.meta' custom made descriptions with [LORA](https://arxiv.org/abs/2106.09685)/[DORA](https://arxiv.org/abs/2402.09353)/[ELLA](https://arxiv.org/abs/2601.02232)-like approaches so automatic descriptions would be done in the user-like fashion, making the search experience even more personalized and accurate.
> 
> We will also no longer be needed to train separate recommendation models for each data modality, as everything is derived from text and eventually mapped onto a unified embedding space, the single `TransformerEvaluator` model would be responsible for grading all types of data, finally achieving one of the main goals of the project - creating a single unified model of user preferences, i.e. some form of digital twin of the user that can be used to search and filter information on the user's behalf.

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
