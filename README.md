<!--[![Join on Reddit](https://img.shields.io/reddit/subreddit-subscribers/Anagnorisis?style=social)](https://www.reddit.com/r/Anagnorisis)-->

# Anagnorisis
Anagnorisis - is a local recommendation system that allows you to fine-tune models on your data to predict your data preferences. You can feed it as much of your personal data as you like and not be afraid of it leaking as all of it is stored and processed locally on your own computer. 


The project uses [Flask](https://flask.palletsprojects.com/) libraries for backend and [Bulma](https://bulma.io/) as frontend CSS framework. For all ML-related stuff [Transformers](https://github.com/huggingface/transformers) and [PyTorch](https://pytorch.org/) are used. This is the main technological stack, however there are more libraries used for specific purposes.


To read more about the ideas behind the project you can read these articles:  
[Anagnorisis. Part 1: A Vision for Better Information Management.](https://medium.com/@AlexeyBorsky/anagnorisis-part-1-a-vision-for-better-information-management-5658b6aaffa0)  
[Anagnorisis. Part 2: The Music Recommendation Algorithm.](https://medium.com/@AlexeyBorsky/anagnorisis-part-2-the-music-recommendation-algorithm-ba5ce7a0fa30)  
[Anagnorisis. Part 3: Why Should You Go Local?](https://medium.com/@AlexeyBorsky/anagnorisis-part-3-why-should-you-go-local-b68e2b99ff53)  

## Running from Docker
The preferred way to run the project is from Docker. This should be much more stable than running it from the local environment, especially on Windows. But be aware that all paths in the projects would be relative the `DATA_PATH` folder that you mount to the container. 

1. Make sure that you have Docker installed. In case it is not go to [Docker installation page](https://www.docker.com/get-started/) and install it. 
2. Clone this repository:
    ```bash
        git clone https://github.com/yourusername/Anagnorisis.git
        cd Anagnorisis
    ```
3. Launch the application
    ```bash
        DATA_PATH=/path/to/your/data EXTERNAL_PORT=5001 docker-compose up -d
    ```
    Note: if you are using Docker Desktop you have to explicitly provide access to `/path/to/your/data` folder in the Docker settings. Otherwise, you will not be able to access it from the container. To do so, go to Docker Desktop settings, then to Resources -> File Sharing and add the path to your data folder.
4. Access the application at http://localhost:5001 (or whichever port you configured) in your web browser.

## Running from the local environment
In case you do not want to use Docker, you can also install the project manually with this commands. Notice that the project has only been tested on Ubuntu 22.04 with Python 3.10, there is no guarantee that it will work on any other platforms or different version of Python. For Windows users I highly recommend to use Docker as there might be some unexpected issues.

1. Clone this repository:
    ```bash
        git clone https://github.com/yourusername/Anagnorisis.git
        cd Anagnorisis
    ```

2. Recreate the Environment with following commands: 
    ```bash 
        # For Linux
        python3 -m venv .env  # recreate the virtual environment
        source .env/bin/activate  # activate the virtual environment
        pip install -r requirements.txt  # install the required packages
        # For Windows
        python -m venv .env  # recreate the virtual environment
        .env\Scripts\activate  # activate the virtual environment
        pip install -r requirements.txt  # install the required packages
    ```

3. Then run the project with command:
    ```bash  
        # For Linux
        DATA_PATH=/path/to/your/data bash run.sh
        # For Windows
        DATA_PATH=/path/to/your/data bash run.bat
    ```
4. Access the application at http://localhost:5001 (or whichever port you configured) in your web browser.

## Additional notes for installation
The Docker container includes Ubuntu 20.04, CUDA drives and several large machine learning models and dependencies, which results in a significant storage footprint. After the container is built it will take about 45GB of storage on your disk. If you want to avoid that, consider running the project from the local environment.

If `DATA_PATH` is not provided, `/project_data` folder in the project root will be used. 

After initializing the project, you will find new `Anagnorisis-app` folder inside of `DATA_PATH` folder. In this folder project's database, migrations, models and configuration file will be stored. After running the project for the first time, `{DATA_PATH}/Anagnorisis-app/database/project.db` file will be crated. That DB will store your preferences, that will be used later to fine-tune evaluation models. Try to make backups of this file from time to time, as it contains all of your preferences, and some additional data, such as playback history.

Running the project from the local environment should be somewhat more efficient as there is no Docker overhead when reading the data. 

If you have a lot of data in your data folder, for the first time hash cache and embedding cache will be gathered. Please be patient, as it may take a while. The percentage of the progress will be shown in the status bar.

The project requires GPU to run properly. When running the project inside the Docker container, make sure that `NVIDIA Container Toolkit` is installed for Linux and `WSL2` for Windows.

## Embedding models
To make audio and visual search possible the project uses these models:  
[LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)  
[Google/SigLIP](https://arxiv.org/pdf/2303.15343)  

All embedding models are downloaded automatically when the project is started for the first time. This might take some time depending on the internet connection. You can see the progress inside `container_log.txt` file that will appear in the project's root folder if you run the project from the Docker container.

## General
Here is the main pipeline of working with the project:  
1. You rate some data such as text, audio, images, video or anything else on the scale from 0 to 10 and all of this is stored in the project database.  
2. When you acquire some amount of such rated data points you go to the 'Train' page and start the fine-tuning of the model so it could rate the data AS IF it was rated by you.  
3. New model is used to sort new data by rates from the model and if you do not agree with the scores the model gave, you simply change it.  

You repeat these steps again and again, getting each time model that better and better aligns to your preferences.  

## Music Module
Please watch this video to see presentation of 'Music' module usage:  
[![Watch the video](https://i3.ytimg.com/vi/vux7mDaRCeY/hqdefault.jpg?1)](https://youtu.be/vux7mDaRCeY)  
To see how the algorithm works in details, please read this wiki page: [Music](wiki/music.md)

## Images Module
Please watch this video to see presentation of 'Images' module usage:  
[![Watch the video](https://i3.ytimg.com/vi/S70Lp0oL7aQ/hqdefault.jpg?1)](https://youtu.be/S70Lp0oL7aQ)   
Or you can read the guide at the [Images wiki](wiki/images.md) page.

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
