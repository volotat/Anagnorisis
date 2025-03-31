<!--[![Join on Reddit](https://img.shields.io/reddit/subreddit-subscribers/Anagnorisis?style=social)](https://www.reddit.com/r/Anagnorisis)-->

# Anagnorisis
Anagnorisis - is a local recommendation system that allows you to fine-tune models on your data to predict your data preferences. You can feed it as much of your personal data as you like and not be afraid of it leaking as all of it is stored and processed locally on your own computer. 


The project uses [Flask](https://flask.palletsprojects.com/) libraries for backend and [Bulma](https://bulma.io/) as frontend CSS framework. For all ML-related stuff [Transformers](https://github.com/huggingface/transformers) and [PyTorch](https://pytorch.org/) are used. This is the main technological stack, however there are more libraries used for specific purposes.


To read more about the ideas behind the project you can read these articles:  
[Anagnorisis. Part 1: A Vision for Better Information Management.](https://medium.com/@AlexeyBorsky/anagnorisis-part-1-a-vision-for-better-information-management-5658b6aaffa0)  
[Anagnorisis. Part 2: The Music Recommendation Algorithm.](https://medium.com/@AlexeyBorsky/anagnorisis-part-2-the-music-recommendation-algorithm-ba5ce7a0fa30)  
[Anagnorisis. Part 3: Why Should You Go Local?](https://medium.com/@AlexeyBorsky/anagnorisis-part-3-why-should-you-go-local-b68e2b99ff53)  


## Installation
Notice that the project has only been tested on Ubuntu 22.04, there is no guarantee that it will work on any other platforms. 


Recreate the Environment with following commands: 
``` 
    # For Linux
    python3 -m venv .env  # recreate the virtual environment
    source .env/bin/activate  # activate the virtual environment
    pip install -r requirements.txt  # install the required packages
```
```
    # For Windows
    python -m venv .env  # recreate the virtual environment
    .env\Scripts\activate # activate the virtual environment
    pip install -r requirements.txt # install the required packages
```


Initialize your database with this command: 
```
    flask --app app db init
```
This should create a new 'instance/project.db' file, that will store your preferences, that will be used later to fine-tune evaluation models. Do not forget to activate your virtual environment when executing this command.


Then run the project with command:
```  
    # For Linux
    bash run.sh
```
```  
    # For Windows
    ./run.bat
```
The project should be up and running on http://127.0.0.1:5001/  

## Downloading the models
To make audio and visual search possible the project uses the models that are based on these works:  
[LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)  
[Google/SigLIP](https://arxiv.org/pdf/2303.15343)  

First of all make sure you have git-lfs installed (https://git-lfs.com).  
Then go to 'models' folder with  
```cd models```

**Music embedder: laion/clap-htsat-fused**  
```git clone https://huggingface.co/laion/clap-htsat-fused```  

Note that not all files from the repository are necessary. If you like, you can download only the files that are needed by hand and place them in the 'models/clap-htsat-fused' folder. Here is the list of files that are necessary:  
```
    config.json
    merges.txt
    preprocessor_config.json
    pytorch_model.bin
    special_tokens_map.json
    tokenizer_config.json
    tokenizer.json
    vocab.json
```

**Image embedder: google/siglip-base-patch16-224**  
```git clone https://huggingface.co/google/siglip-base-patch16-224```

Same thing here, not all files are necessary. Here is the list of files that are essential:  
```
    config.json
    model.safetensors
    preprocessor_config.json
    special_tokens_map.json
    spiece.model
    tokenizer_config.json
```


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
