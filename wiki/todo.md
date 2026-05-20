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
Make number of images per page adjustable via config.yaml file.  
Make number of columns of images presented dependent on the screen size. 
Add BRISQUE score filter for simple image quality assessment. 
Replace SigLIP model with more powerful MobileClip2.

### Train page
Disable the start button if fine-tuning has already started.  
When refreshing the page the information about the previous run should appear.  
Pressing the start button should remove all the information from the graph and restart the process.  
Add a button to stop the training process and save the best model so far.  
After training is complete or canceled reload the evaluation models.  

### Wiki
Complete self-hosting guide.

### General
Add some sort of a control panel that is active on all pages (may be hidden) and shows current GPU memory load by the system and console output.  
Create an extension that provides recent Arxiv papers (https://arxiv.org/list/cs.LG/recent).    
Implement usage of FAISS library for fast vector search.    
When the folder name contains '[' and ']' symbols it is not correctly read by python scripts for some reason.   
Add automatic database backup generation from time to time to prevent loss of data in case of a failure.  
Find a way for more optimal embeddings storage in the DB.  
Add display of the current version of the project into the header. 
Add pop-up messages about new version of the project available to download and some button that allows to download it directly from the UI and restart the server.
When starting the Docker container, if the provided path to the share folder does not exists the error should be displayed and the container should not start.
Add a way to create new folders in the UI.  
Add a way to copy path to a folder trough the UI.  
When gathering hashes of files make the cache saving once in a while, as this process might take quite a long time especially for video files.
All active filters should be automatically gathered from the backend and displayed in the UI, instead of being hardcoded in the frontend.
In the recommendation engine replace `full_play_count` and `skip_count` with `play_time` that dynamically calculate how much time user spent listening or watching the file. This should better reflect user preferences and be consistent across different types of data (at least music and videos).

Replace all the hash gathering occurrences with a single function that handles it in a unified way and notify the UI about the progress.
Try to build a safe fail when the hashing algorithm is changes so we don't lose data in the DB that is hash related.

### Ideas
Add new downloadable module for 'Deep Research'-like functionality that uses user-trained text evaluating model to search for relative information adjusted to user preferences.
