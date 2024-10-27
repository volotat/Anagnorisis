# Change History


## TODO list
 
### Music 
Add a way to change the rest of the metadata of the song.  
Add volume control.  
Add a message if there are new unindexed media files.  
Move the "edit" button to the song list element.  
Add "Chain Mode" where each song is selected as the most similar to a previous song.  
Add a way to restart the radio session.  
Explore ways to optimize the music library update process.  
In radio mode add an "Open file destination" button to be able to move or remove bad music if found.  

### Images 
Implement some sort of effective resolution estimation technique.   
Improve sorting by resolution performance (one idea might be caching information of image resolution).   
Add a way to select only some particular folders for displaying images from.  
Remove from DB images that no longer exist in the media folder and have no user rating. This should reduce the size of the DB as embeddings took quite a lot of space.  
Add a way to create new folders in the UI.  
Add a way to copy path to a folder trough the UI.  
Make start rating bar sensitive to fractional value to allow more fine-tuned user rating.  
Make number of images per page adjustable via config.yaml file.  
Make number of columns of images presented dependent on the screen size.  

### Train page
Disable the start button if fine-tuning has already started.  
When refreshing the page the information about the previous run should appear.  
Pressing the start button should remove all the information from the graph and restart the process.  
Add a button to stop the training process and save the best model so far.  
After training is complete or canceled reload the evaluation models.  

### Wiki

### General
Add some sort of a control panel that is active on all pages (may be hidden) and shows current GPU memory load by the system and console output.  
Create an extension that provides recent Arxiv papers (https://arxiv.org/list/cs.LG/recent).    
Add a way to export the current database as .csv and import it back.  
Implement usage of FAISS library for fast vector search.  
Update main readme.md file and write about 'images' page, also need to add a way to download SigLIP model.  
When the folder name contains '[' and ']' symbols it is not correctly read by python scripts for some reason.  
Redo music page representation to more closely resemble images page and apply its improvements file hash cashing and metadata search.  
Add automatic database backup generation from time to time to prevent loss of data in case of a failure.  
Find a way for more optimal embeddings storage in the DB.  
Create a roadmap for the project.  
Separate the file-management part of the project into separate scripts.  

## Important fixes before 0.2.0 release
Create a working docker environment to easily run the project.  

## Versions History

### Version 0.1.1 (27.10.2024)
Added a way to select multiple files in the 'Images' module to perform some actions on them, such as deleting or moving into another folder.  
Added batch deletion of images in the 'Images' module.  
Added batch moving of images in the 'Images' module.  
Added a way to search for similar images to the selected one in the 'Images' module. To activate it, you need to place path to the image into the search field or press the 'Find similar' button nearby the image.  
If shift is pressed while selecting images, all images between are ether selected or deselected if there were no images selected beforehand. Otherwise all images from the first to the last selected are selected or deselected.   
Now there are numbers of images in each folder and total number of images including subfoldes displayed in the 'Images' module beside the folder name.  

### Version 0.1.0 (21.10.2024)
Images embeddings caching moved from file-based storage solution to the database. While it is significantly increased the size of the database, the retrieval is much faster now and there might be a way to optimize the compression of the embeddings in the future. It also allow to store embeddings of the images that was removed to still use them for the training process.  
In 'Training' module MAPE calculation is slightly change to better represents small values that might results as negatives before and make not much sense as a percentage.  
In 'Images' module now evaluating model hash and embedding model hash are stored in the database to better track where the values comes from and when they need to be updated.  
Gathering image embeddings process was completely rewritten for better performance and to be able to use the new caching mechanism. Now it takes about 1 minute per 10000 images to gather embeddings for the first time.  
Now there is no need to restart the server to update the image embeddings model. It is done automatically when the training is completed.  
It is now possible to update 'Images' media folder from the UI.  
Sort by rating process in 'Images' module was improved and optimized.  
When training process is shown in the UI, decimation is now performed 'manually' as the method integrated in the 'chart.js' does not seems to work with dynamically changing graphs.  
Image module is now fully functional.  
New 'Images' page is added to the wiki with the description of how to use 'Images' module.  

### Version 0.0.16 (12.10.2024)  
Many improvements in file reading and filtering performance on 'Images' module. These improvements will be spread to other modules later in the development.  
Added caching mechanism for hash gathering in 'Images'.  
Added file list caching mechanism to cache current state of each subfolder with respect to its last modification time.  
Added mechanism to train and use evaluation model for images.  
In 'Train' module added a button to activate the image evaluation model training process.  
Improved search by resolution and proportion performance, but limited maximum number of images for processing by 10000 as the result is not yet cached and there is no indication of the current progress.  
Added callback for embedding gathering to better represent internal processes to the user in 'Images'.  
Now pressing enter in the search bar leads to automatic call to search method in 'Images'.  
Model rating in 'Images' now stored as a float in the database to allow more fine-tuned sorting by this value.  

### Version 0.0.15 (07.10.2024)
Added mechanism to rate images with UI and store it in DB in a more efficient way that it was done with music. The music DB should also be change later to follow the same principals.

### Version 0.0.14 (29.09.2024)
Added current number of images processed at the embedding extracting stage, as sometimes it could take some amount of time at first extraction.  
The GLIP model is no longer loaded into RAM before image search methods are called.  
'Music' wiki page is updated with the description of the current version of music recommendation algorithm.  
Added display of LaTeX formulas in markdown files.  
Added 'Videos' page with simple data gathering and generating/displaying preview of videos.  

### Version 0.0.13 (02.07.2024)
Fixed issue when switching pages in images was resetting the current active folder.  
Added button that sends image file into a trash bin.  
Added pagination at the bottom of the page in images.  
Added search status text in images.  
Added full size image preview with PhotoSwipe.  
Implemented fast cache for image embeddings for faster search.  

### Version 0.0.12 (28.06.2024)
Now when a user rates a song the data of the action is also stored in the DB for further analysis.  
Added display of information about the image, such as file path, file size, resolution and so on is now displayed under each image.    
Added button below the image that opens it inside its folder.   
Added search by file size, images resolution and similarity of the images.   
Implemented switching between subfolders in images.  

### Version 0.0.11 (12.06.2024)
Added 'Images' page with text-based image search using [SigLIP](https://arxiv.org/abs/2303.15343) model.  

### Version 0.0.10 (17.05.2024)
The main Readme.md file has been updated to better represent the current status of the project.  
Wiki pages have also been updated to better reflect the project at the current state.  
Removed all CDN services dependencies.  

### Version 0.0.9 (13.05.2024)
Fixed an issue, when music is not readable after indexing (wrong path was written to DB).  
Did some more investigation on how to encode embeddings to text to then train an LLM on top of it to predict user score for a different types of data. Unfortunately, no experiments show better or even comparable results to a simple feed-forward NN train from embeddings directly. So my conclusion for now, that an idea of a singular rating model produced from a pre-trained LLM might be too hard to implement, so I guess the better approach would be to use separate networks for different types of data and maybe find a better way later. For now I'm going to abandon using LLM in the project at all and use some text-embedding model instead.  
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
Fronted now shows model-based ratings in case there are no user ratings in light blue color.  

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
Added message about TTS initialization when AIDJ was activated.
Added music library navigation system.
Added icon change on play button when song is paused. 
Radio related code has been moved to a separate js file, also some code cleanup. 
Now radio history is not lost when the music page has been reloaded.

### Version 0.0.4 (06.01.2024)
Added installation info to the README.md file.  
Fixed some bugs that cause full_play_count and skip_count not being updated.  
Fixed bug that caused page to scroll up when closing song edit modal window.  
Fixed a dramatic mistake in fine-tuning, where text based data were sampled line by line instead of document based sampling as it was intended.  
Added "Self-aware" mode to the fine-tuning that adds the main code of the project to the LLM training data. 
Better representation of radio mode play history.  
Tabs on the music page are responsive now.  
Now TextPredictor is a singleton class that should eliminate problems with copying LLM in several processes.  
Added "The Gardian" news aggregator.  

### Version 0.0.3 (16.11.2023)
Radio mode converted to a state based approach to see everything that is happening at the backend in real time and stack prepared announcements and songs in a buffer.  
Images of song covers are now sent from backend to frontend as base64 images.  
There are now multiple prompts for AI DJ. It should help to avoid repetitions.  
Added an edit button that allows to change song metadata from the frontend UI. Right now only the lyrics of the song are actually changed.  

### Version 0.0.2 (10.11.2023)
Added skip multiplier to music recommendation algorithm.  
Added display of the music cover to the music page.  
Added internal playback history and switching to the previous track.  
Added play button icon change when paused.  
Now you can start playing random songs after the page music loaded by pressing the play button.  
Now you can change current song play time by clicking on the progress bar.  
Added initial version of AI DJ mode.  
Added 'config.yaml' that contains all currently available settings of the application.  

### Initial version 0.0.1 (01.09.2023)
Initial server and frontend architecture implemented.  
Basic news rating using LLAMA 2 implemented.  
Basic music player with user's ratings implemented.  
Front page of the server displays the README.md file.  
