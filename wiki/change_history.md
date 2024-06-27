# Change History


## TODO list
 
### Music 
Add a way to change the rest of the metadata of the song.
Add volume control.
Add a message if there are new unindexed media files. 
Move the "edit" button to the song list element.
Add "Chain Mode" where each song is selected as the most similar to a previous song.
Add a way to restart radio session.
Explore ways to optimize music library update process.
In radio mode add "Open file destination" button to being able to move or remove bad music if found.

### Images 
Implement quality evaluation with personal evaluator.   
Implement some sort of effective resolution estimation technique.   
Add button to move image file into trash bin.  
Improve sorting by resolution performance (one idea might be caching information of image resolution).  

### Train page
Disable the start button if fine-tuning has already started.  
When refreshing the page the information about previous run should appear.
Pressing the start button should remove all the information from the graph and restart the process.
Add a button to stop the training process and save the best model so far.
After training is complete or canceled reload the evaluation models.

### Wiki

### General
Add some sort of a control panel that is active on all pages (may be hidden) and shows current GPU memory load by the system and console output.  
Create an extension that provides recent Arxiv papers (https://arxiv.org/list/cs.LG/recent).   
Add a way to export current database as .csv and import it back.
Implement usage of FAISS library for fast vector search.

## Important fixes before 0.1.0 release
Create a working docker environment to easily run the project.  

## Versions History

### Version 0.0.12 (28.06.2024)
Now when user rates a song the data of the action also stored in the DB for further analysis.  
Added display of information about the image, such as file path, file size, resolution and so on is now displayed under each image.  
Added button below the image that opens it inside its folder.  
Added search by file size, images resolution and similarity of the images.  
Implemented switching between subfolders in images.  

### Version 0.0.11 (12.06.2024)
Added 'Images' page with text-based image search using [SigLIP](https://arxiv.org/abs/2303.15343) model.  

### Version 0.0.10 (17.05.2024)
Main Readme.md file has been updated to better represent the current status of the project.  
Wiki pages has also been updated to better reflect the project at the current state.  
Removed all CDN services dependencies.  

### Version 0.0.9 (13.05.2024)
Fixed an issue, when music is not readable after indexing (wrong path was written to DB).  
Did some more investigation on how to encode embeddings to text to then train an LLM on top of it to predict user score for a different types of data. Unfortunately, no experiments show better or even comparable results to a simple feed-forward NN train from embeddings directly. So my conclusion for now, that an idea of a singular rating model produced from a pretrained LLM might be too hard to implement, so I guess the better approach would be to use separate networks for different types of data and maybe find a better way later. For now I'm going to abandon using LLM in the project at all and use some text-embedding model instead.  
Moved the project to extension-based architecture where each such extension might be developed completely separately from other extensions using general methods from the main code base. That is achieved by storing everything that relates to an extension - js, python and html code - in a separate folder in the main 'pages' folder.  
Temporarily removed LLM instance, chat and news pages as a part of rebuilding the project with new architecture.  
Now when the music library is updating the progress bar actually showing the progress of it.  
On the music library page added a way to change the music library folder directly from the UI.  
'Fine-Tuning' page is replaced with 'Train' pages that are now made as an extension and for now only deal with the music evaluator model.  
Now when music is selected in the radio mode it is rescored with the current evaluator audio model in case it was updated but not used to reindex the library yet.  

### Version 0.0.8 (30.04.2024)
Did some research on embedding integration to Llama2 with QLoRa. While the approach itself is working, unfortunately, it showed quite a low performance compared to even the simplest NN. Research data and scripts are stored in 'research/audio_embeddings'.  
Made simple network that predict ratings from audio embeddings, right now it only works with *.mp3 files and treat each rating as a separate class on the training and evaluation stage. In the future the final score will be a weighted sum of the classes and the training loss function might be changed to support that.  
Added button for starting training of the evaluator network at the fine-tuning page.
Now when the music library is updating the model based evaluation of the music rating is also updated with the current evaluation model.  
When music is selected, model-based rating is used in case there is no user rating for a particular song.  
Fronted now shows model-based ratings in case there are no user rating in light blue color.  

### Version 0.0.7 (01.04.2024)
For song information also added information about the probability with which the song has been selected. Mostly for debugging purposes.   
Added last_played data for songs to the DB and in the song preview.  
Now the new song selection probability also depends on when the last time it was played.  
Now it's possible to change the song by pressing on its cart in the radio mode.  
Now the current state of the song is constantly saved. This allows you to continue listening after the page has been reloaded. 

### Version 0.0.6 (25.02.2024)
Regex format for predicted GPS coordinates of the news has been fixed to allow negative numbers.  
Some code refactoring related to the music page.  
Some preparations and tests for extension-based architecture.  
Added some primitive control of the news sources with config.yaml.  
Added radio configuration form with an option to switch off AI DJ.  

### Version 0.0.5 (14.01.2024)
Added LLM loading indication.  
Added tables markdown extension for proper wiki rendering.  
TTS initialization changed to an on demand approach for faster start-up time.  
Added wiki pages rendering. The 'wiki' button has been removed, as everything is acceptable through the main page. 
Audio elements removed from base.html and added to music.html.
Added message about TTS initialization when AIDJ been activated.
Added music library navigation system.
Added icon change on play button when song is paused. 
Radio related code has been moved to separate js file, also some code cleanup. 
Now radio history is not lost when music page has been reloaded.

### Version 0.0.4 (06.01.2024)
Added installation info to the README.md file.  
Fixed some bugs that cause full_play_count and skip_count not being updated.  
Fixed bug that caused page to scroll up when closing song edit modal window.  
Fixed a dramatic mistake in fine-tunning, where text based data were sampled line by line instead of document based sampling as it was intended.  
Added "Self-aware" mode to the fine-tunning, that adds main code of the project to the LLM training data. 
Better representation of radio mode play history.  
Tabs on the music page are responsive now.  
Now TextPredictor is a singleton class, that should eliminate problems with copying LLM in several processing.  
Added "The Gardian" news aggregator.  

### Version 0.0.3 (16.11.2023)
Radio mode converted to a state based approach to see everything that is happening at the backend in a real time and stack prepared announcements and songs in a buffer.  
Images of song covers now send from backend to frontend as base64 images.  
There is now multiple prompts for AI DJ. It should help to avoid repetitions.  
Added a edit button that allows to change song metadata from the frontend UI. Right now only the lyrics of the song is actually changes.  

### Version 0.0.2 (10.11.2023)
Added skip multiplier to music recommendation algorithm.  
Added display of the music cover to the music page.  
Added internal playback history and switching to the previous track.  
Added play button icon change when paused.  
Now you can start playing random song after the page music loaded by pressing the play button.  
Now you can change current song play time by clicking on the progress bar.  
Added initial version of AI DJ mode.  
Added 'config.yaml' that contains all currently available settings of the application.  

### Initial version 0.0.1 (01.09.2023)
Initial sever and frontend architecture implemented.  
Basic news rating using LLAMA 2 implemented.  
Basic music player with user's ratings implemented.  
Front page of the server displays the README.md file.  
