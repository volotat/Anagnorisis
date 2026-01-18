<!--[![Join on Reddit](https://img.shields.io/reddit/subreddit-subscribers/Anagnorisis?style=social)](https://www.reddit.com/r/Anagnorisis)-->

# Anagnorisis
Anagnorisis - is a local recommendation system that allows you to fine-tune models on your data to predict your data preferences. You can feed it as much of your personal data as you like and not be afraid of it leaking as all of it is stored and processed locally on your own computer. 


The project uses [Flask](https://flask.palletsprojects.com/) libraries for backend and [Bulma](https://bulma.io/) as frontend CSS framework. For all ML-related stuff [Transformers](https://github.com/huggingface/transformers) and [PyTorch](https://pytorch.org/) are used. This is the main technological stack, however there are more libraries used for specific purposes.


To read more about the ideas behind the project you can read these articles:  
[Anagnorisis. Part 1: A Vision for Better Information Management.](https://volotat.github.io/p/anagnorisis-part-1-a-vision-for-better-information-management/)  
[Anagnorisis. Part 2: The Music Recommendation Algorithm.](https://volotat.github.io/p/anagnorisis-part-2-the-music-recommendation-algorithm/)  
[Anagnorisis. Part 3: Why Should You Go Local?](https://volotat.github.io/p/anagnorisis-part-3-why-should-you-go-local/)  

## General
Here is the main pipeline of working with the project:  
1. You rate some data such as text, audio, images, video or anything else on the scale from 0 to 10 and all of this is stored in the project database.  
2. When you acquire some amount of such rated data points you go to the 'Train' page and start the fine-tuning of the model so it could rate the data AS IF it was rated by you.  
3. New model is used to sort new data by rates from the model and if you do not agree with the scores the model gave, you simply change it.  

You repeat these steps again and again, getting each time model that better and better aligns to your preferences.  

The big vision of this project is to provide a platform that creates a local, private model of your interests. That likes what you like and sees importance where you would see it. Then you can use this model to search and filter local and global information on your behalf in a way you would do it yourself but in a much faster and efficient way. Making this platform (in the future) a go to place to see news, recommendations and insights, and so on, tailored specifically for you. As the internet gets populated with bots and AI slop, a platform like this might create a necessary filter to be able to navigate in this chaotic information space efficiently.

## Music Module
Please watch this video to see presentation of 'Music' module usage:  
[![Watch the video](https://i3.ytimg.com/vi/vux7mDaRCeY/hqdefault.jpg?1)](https://youtu.be/vux7mDaRCeY)  
To see how the algorithm works in details, please read this wiki page: [Music](wiki/music.md)

## Images Module
Please watch this video to see presentation of 'Images' module usage:  
[![Watch the video](https://i3.ytimg.com/vi/S70Lp0oL7aQ/hqdefault.jpg?1)](https://youtu.be/S70Lp0oL7aQ)   
Or you can read the guide at the [Images wiki](wiki/images.md) page.

## Running from Docker
The preferred way to run the project is from Docker. This should be much more stable than running it from the local environment, especially on Windows. But be aware that all paths in the projects would be relative the `DATA_PATH` folder that you mount to the container. 

1. Make sure that you have Docker installed. In case it is not go to [Docker installation page](https://www.docker.com/get-started/) and install it. 
2. Clone this repository:
    ```bash
        git clone https://github.com/volotat/Anagnorisis.git
        cd Anagnorisis
    ```
3. Specify your environment variables in a `.env` file. You can see the example in the `.env.example` file.
    ```bash
        # .env.example
        CONTAINER_NAME=anagnorisis-app # The name of the Docker container
        EXTERNAL_PORT=5001 # The external port for accessing the application
        # ANAGNORISIS_USERNAME=**** # The username for accessing the application (uncomment if you want to use it)
        # ANAGNORISIS_PASSWORD=**** # The password for accessing the application (uncomment if you want to use it)
        PROJECT_CONFIG_FOLDER_PATH=/path/to/folder/Anagnorisis-app # The path to the folder where your personal database and personally trained recommendation models will be stored
        IMAGES_MODULE_DATA_PATH=/path/to/folder/Images # The path to the folder with your images data
        MUSIC_MODULE_DATA_PATH=/path/to/folder/Music # The path to the folder with your music data
        TEXT_MODULE_DATA_PATH=/path/to/folder/Text # The path to the folder with your text data
        VIDEOS_MODULE_DATA_PATH=/path/to/folder/Videos # The path to the folder with your videos data
    ```
4. Launch the application
    ```bash
        docker-compose up -d
    ```
    Note: if you are using Docker Desktop you have to explicitly provide access to `/path/to/your/data` folders in the Docker settings. Otherwise, you will not be able to access it from the container. To do so, go to Docker Desktop settings, then to Resources -> File Sharing and add the path to your data folder.
4. Access the application at http://localhost:5001 (or whichever port you configured) in your web browser.

## Initialization

To avoid issues with corrupted models being downloaded, **be patient while the application is initializing for the first time**. All models are quite large and might take some time to download depending on your internet connection speed. You can check the progress in the `logs/{CONTAINER_NAME}_log.txt` file that will appear in the project's root folder. The project UI will also show the initialization status, but  for now without download progress percentages. 

If for some reason the initialization process is interrupted (for example you stopped the container while models were being downloaded), upon the next start the application will check for corrupted models and try to re-download them automatically. If this does not help, please delete the `models` folder inside the project's root folder and start the application again. This will force the application to download all models from scratch. 

## Troubleshooting

In case you encounter and error like this:
```
ERROR: for {your application container name} Cannot start service anagnorisis: error while creating mount source path '{PROJECT_CONFIG_FOLDER_PATH}': chown {PROJECT_CONFIG_FOLDER_PATH}: operation not permitted
```

You have to create the folder specified in `PROJECT_CONFIG_FOLDER_PATH` environment variable manually on your host machine. Docker sometimes cannot create such folders by itself due to permission issues.

## Additional notes for installation
The Docker container includes Ubuntu 22.04, CUDA drives and several large machine learning models and dependencies, which results in a significant storage footprint. After the container is built it will take about 45GB of storage on your disk. 

For best user experience I would recommend running the project with relatively modern Nvidia GPU with at least 8Gb of VRAM and 32Gb of RAM . At least this is the configuration I am using myself. However, the project should be able to run on lower configurations, but performance might be poor especially without CUDA-friendly GPU. It is usually take less then 4GB of VRAM to run the project, however when training recommendation models it usually spikes up to 5-7Gb of VRAM usage.

After initializing the project, you will find new `database` folder inside of `PROJECT_CONFIG_FOLDER_PATH` folder. In this folder project's database, migrations, models and configuration file will be stored. After running the project for the first time, `{PROJECT_CONFIG_FOLDER_PATH}/database/project.db` file will be crated. That DB will store your preferences, that will be used later to fine-tune evaluation models. Try to make backups of this file from time to time, as it contains all of your preferences, and some additional data, such as playback history.

If you have a lot of data in your data folder, for the first time hash cache and embedding cache will be gathered. Please be patient, as it may take a while. The percentage of the progress will be shown in the status bar.

The project requires GPU to run properly. When running the project inside the Docker container, make sure that `NVIDIA Container Toolkit` is installed for Linux and `WSL2` for Windows.

## Security notes
The project is meant to be run on the localhost only for now. The default configuration ip address is set to `127.0.0.1` inside `docker-compose.yml` file. This means that the application will only be accessible from the machine it is running on. If you want to access it from other devices on your local network, you can change this address to `0.0.0.0`. You can even tunnel it to the internet using services like [ngrok](https://ngrok.com/) or [cloudflare tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/). However, I would strongly recommend against exposing the service to the internet (unless you are 100% know what you are doing) as there is no proper security work has been done yet. 

## Embedding models
To make audio, visual and text search possible the project uses these models:  
[LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)  
[Google/SigLIP](https://arxiv.org/pdf/2303.15343)  
[JinaAI/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)

All embedding models are downloaded automatically when the project is started for the first time. This might take some time depending on the internet connection. You can see the progress inside `logs/anagnorisis-app_log.txt` file that will appear in the project's root folder if you run the project from the Docker container.

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
