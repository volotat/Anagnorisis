# Change History


## TODO list
 
### Music 
Add a way to change the rest of the metadata of the song.  
Add volume control.   
Move the "edit" button to the song list element.  
Add "Chain Mode" where each song is selected as the most similar to a previous song.  

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
Implement usage of FAISS library for fast vector search.  
Update main readme.md file and write about 'images' page, also need to add a way to download SigLIP model.  
When the folder name contains '[' and ']' symbols it is not correctly read by python scripts for some reason.   
Add automatic database backup generation from time to time to prevent loss of data in case of a failure.  
Find a way for more optimal embeddings storage in the DB.  
Implement automatic model downloading at the fresh start of the project.

## Versions History

### Version 0.2.0 (04.04.2025)
* Docker container for running the project configured and extensively tested to make running the project easier.
* Database is now automatically created if it doesn't exist, and migrations are applied on application startup, removing the need for manual `flask db` commands.
* Introduced `DATA_PATH` environment variable and `--data-folder` argument to manage all user-specific data (database, models, config as well as user's personal data) in a central location (defaults to `./project_data`).
* Implemented a system where user settings in `{DATA_PATH}/Anagnorisis-app/config.yaml` override the base `config.yaml` for proper configuration personalization that do not get overwritten after the `git pull`. The user config is created automatically from the default if it doesn't exist.
* Required embedding models (`SigLIP`, `CLAP`) are now downloaded automatically on first use if not found in the configured `models_path`.
* Revised `README.md` with updated setup instructions (including Docker) and adjusted `run.sh`/`run.bat` for the new data folder structure and automatic migrations.
* Changed default host to `0.0.0.0` for it to be compatible with Docker. 
* Added `huggingface-hub` to `requirements.txt` (necessary for automatic models download).

### Version 0.1.7 (29.03.2025)
* Added a Docker-based test (`tests/test_docker_build.sh`) to automatically verify project installation and startup in a clean environment.
* Removed the old `_music_v0.1.1` module code, outdated research folders, and documentation. Removed the `TTS` dependency from `requirements.txt`.
* Added initial `text` configuration section to `config.yaml`.
* Created a reusable `FileGridComponent.js` for displaying file grids and integrated it into the 'Images', 'Music', and 'Text' modules for consistency.
* Created a reusable `PaginationComponent.js` for pagination and integrated it into the 'Music', and 'Text' modules.
* Method `show_search_status` separated into a `CommonSocketEvents` and now targeted for each folder separately. This approach will later be used for almost all socket events.
* In `Music` module the playlist now is automatically cleaned up in case any error occurs during the playback. This should prevent the playlist being not accessible after the error.
*   **New 'Text' Module:** Introduced a new 'Text' module for viewing and editing `.txt` and `.md` files:
    *   Includes folder navigation and file grid display.
    *   Features a modal window for viewing content with Raw/Markdown/HTML tabs (Markdown/HTML rendering not yet implemented).
    *   Allows basic text editing and saving.

### Version 0.1.6 (01.03.2025)
* Implemented control of music playback (play/pause, next/previous track) through browser's `navigator.mediaSession` API. This allows control from external devices and browser UI elements that support media sessions (like media keys on keyboards, lock screen controls, and some bluetooth devices).
* Fixed issues that prevented `full_play_count` and `skip_count` from being correctly incremented and stored in the database when songs are played or skipped. These counts are now accurately tracked and persisted for use in the recommendation algorithm.
* Implemented saving and restoring of the music module's state using `localStorage`.  The current playlist, current song, playback position, and play/pause status are now preserved across page refreshes, providing a more seamless user experience.
* Resolved an issue that prevented searching for music files by their full file path within the `Music` module's library view. Searching by file path now functions as expected.
* Re-enabled and fixed the 'Find similar' button in the `Music` module. This feature now correctly initiates a similarity-based search for music tracks.
* Improved playlist behavior: When playing music from the library view, if files are selected, only the selected files are added to the playlist and played. If no files are selected, all files in the current view are played (as before).
* Added a `Last played` column to the music library grid view in the `Music` module. This column displays the last time each song was played, providing users with playback history information directly in the library view.
* `Open` button in the music library grid view in the `Music` module now works correctly. This button allows users to open the folder containing the selected music file directly from the library view.

### Version 0.1.5 (24.02.2025)
* Added metadata caching mechanism and integrated in both `Images` and `Music` modules. New `CachedMetadata` class in `pages/file_manager.py` to handle caching of file metadata, improving performance by reducing redundant metadata extraction. Metadata is cached using file hashes as keys, making the cache robust to file renaming and moving. Cache entries are automatically invalidated after three months to ensure data freshness.
* Now process of gathering metadata is properly displayed in the status bar in the `Images` and `Music` modules displaying the current percent and total number of files to process.
* Limitations on `resolution` and `proportion` search in the `Images` module have been removed, allowing users to search for images of any resolution or proportion.
* Model training for `Music` module has been properly restored and tested.
* While training the metric has been updated from `1 - MAPE` to `1 / (1 + MAPE)` as it better reflects the quality of the model and cannot give negative values.
* Fixed an issue with sorting files in the `Music` module.

### Version 0.1.4 (16.02.2025)
* `ask.py` script and related functionality is now moved to a separate project called [InsightCoder](https://github.com/volotat/InsightCoder).
* `Recommendation engine` is now separated into its own module and supports creating a list of recommendation based on the same algorithm as it was used in the radio mode. It supposed that the same module would be used for all other types of data supported by the project in the future.
* `Star rating` module is now supports fractional values and is more responsive to the user input. Database entries related to the user's ratings are now also stored as floats.
* `Music` module are now supports filtering files by `rating`, `similarity`, `file size`, `random` and `recommendation`.
* Returned the ability to rate songs in the `Music` module. Play and skip counts are yet to be implemented.
* Number of files per page in the `Music` are set to 30 as aquering metadata for music files is quite slow for now.
* Some primitive experiments with p2p connection were done in the `Research` folder.
* Improved extension loading mechanism using importlib.import_module for better code organization and maintainability.
* Added configurable database path via database_path setting in `config.yaml`.
* Minor UI improvements in the `Music` module

### Version 0.1.3 (04.02.2025)
*   The project now checks for the existence of the image/music evaluator models before attempting to load them. If they don't exist, a dummy model is created to prevent errors. This ensures the application can start even if the evaluator models haven't been trained yet.
*   Replaced the old audio embedding model `m-a-p/MERT-v1-95M` with `LAION-AI/CLAP`. New audio embedding model is based on the CLIP architecture that allows to make text search over the local music library.
*   Significantly refactored the music embedder code. Removed the `AudioEmbedder` class from the main code. Implemented a new `MusicSearch` class in the `Music` page module, analogous to the `ImageSearch` class in the `Images` module, to handle music library searches.
*   The `pages/music/serve.py` file was updated to reflect the changes in the way music is processed and embeddings are handled.
*   When importing the database from a `.csv` file, empty fields are now correctly handled as `None` instead of empty string that caused errors.
*   Some more info has been added to `project_info` folder to better handle AI interactions.
*   The `ask.py` script can now use prompts from `.txt` files, allowing for more complex and reusable queries.
*   Updated the `requirements.txt` file to include the sentencepiece library.
*   Restored song metadata representation in the song's control panel. It now correctly displays album's image, artist, album, and title as well as current rating of the song. Changing the song's metadata and rating is not yet available.

### Version 0.1.2 (28.01.2025)
*   File `requirements.txt` has been updated to include all necessary packages for the project without particular versions.
*   Roadmap of the project has been added to the wiki.
*   A complete overhaul of the 'music' module has been done. The old version was moved to `pages/_music_v0.1.1`. The new module is more similar to the 'images' module, using a grid-based layout for music files and folders, and has many of its improvements, such as file-hashing mechanism for faster file access, and a similar approach to the database interaction. Some of the features from the old 'music' module are still missing, but they will be added in the future. Note, that recommendations are not currently working as the recommendation model is not yet utilized. Embedding model is also going to be changed, so music embeddings are not yet in use. There is no way to rate the music for now either. The old `js/radio.js` file was removed, and all the radio functionality was moved into the new `js/main.js`. Also `js/library.js` was removed and its functionality were also added to the `js/main.js`. The `pages/music/data` folder was also removed, and the background sound file was moved to `static/background-music`.
*   The music DB model (`MusicLibrary`) has been changed to resemble the images DB model. It now includes `id`, `hash`, `file_path`, `user_rating`, `user_rating_date`, `model_rating`, `embedding`, `embedder_hash`, `full_play_count`, `skip_count`, and `last_played` fields. Many of the old fields are not used anymore. Embeddings are not yet stored in the DB.
*   The `ImagesLibrary` model now also includes `model_hash` and `embedder_hash` fields. The model rating is also now a float number for images.
*   Added a separate folder `project_info` with general data about the project and its structure, and `ask.py` script that uses "Gemini 2.0 Flash-exp" (over 1m tokens of context) to answer questions about the project and help in its development. The `ask.py` sends the entire codebase to Google's servers, so users should avoid including any sensitive data in the project. The response of the model is saved into `project_info/llm_response.md` file for review. The user prompt is stored to the `project_info/llm_prompt.txt` file for review.
*   Whole database now could be downloaded as a `.csv` file via the download button on the navbar.
*   Added import into the database from a `.csv` file using the upload button on the navbar. The import only adds new data and does not remove any data that already in the DB.
*   The project now uses a file list caching mechanism (`CachedFileList`) and a file hash caching mechanism (`CachedFileHash`) in `pages/file_manager.py`, for better performance of file management. The old way of database access was removed. The new one uses the same approach as in 'images' module.
*   The `pages/utils.py` file was created and the methods `convert_size` and `convert_length` were moved to it.
*   The rating components are now a separate .js file `pages/StarRating.js` that is used in both 'images' and 'music' modules.
*   The database now stores the embedding model's hash (`model_hash`) to prevent using old rating models.
*   The project is now more strictly divided by extensions, as all of the data related to an extension, including the DB models, server-side python code, html page and front-end javascript code is now stored in a separate folder. The base code now acts as a base for all the extensions. The extensions are automatically discovered.
*   The navbar now contains a button to download the whole DB as a .csv file.
*   The navbar now contains a button to upload the whole DB from a .csv file.

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
