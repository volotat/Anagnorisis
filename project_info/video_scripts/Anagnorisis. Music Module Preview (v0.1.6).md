## Anagnorisis - Music Module Preview (Version 0.1.6)

Welcome to ‘Anagnorisis’. ‘Anagnorisis’ is a local recommendation system designed to let you search, filter, and enjoy your personal data securely, all on your local machine. Anagnorisis could learn from *your* feedback to provide a truly personalized experience. All models are trained and stored privately on your device only.

Today, I’m gonna present to you the ‘Music’ module. Let’s see how to use it.

### Setting up Music Library Path

First, set up the path to your local music library. Copy the folder path into this field, and press the ‘Save’ button.

This action informs the system about your local music collection, enabling it to index your library and prepare for personalized music experiences.

### Browsing Music Library

Now you can explore your music library. Navigate it just like any familiar file browser. Click on a folder to view its content and any subfolders it contains.

On the left, you'll see all the music files from the currently selected folder and any subfolders within it, presented in a clear, organized view.

### Filtering and Searching Music Library

The module offers powerful filtering options to help you find the perfect music for any moment. You can search for music files using a text query. Simply type anything you have in mind and press the ‘Search’ button.

For instance, you could search for ‘Happy music’ or ‘Christmas songs,’ and the search engine will intelligently suggest music from your local library that best matches your query.

Beyond text search, you can also use special keywords to filter your music library in various ways.

The ‘file size’ keyword lets you filter by the size of the music file. ‘length’ helps you find tracks of a specific duration.  If you're looking to declutter your library, ‘similarity’ can help identify potential duplicates. Keyword ‘random’ is perfect for exploring your library. And ‘rating’ allows you to prioritize music based on your personal scores.

Among these keywords there is a special option: ‘recommendation’. It will generate a list of music files based on the recommendation algorithm that we will see closer in just a bit.

### Playlist Panel

After filtering the files to your desire press the 'Play as playlist’ button to start the listening session. When you activate the playlist, a convenient playlist panel appears on the right side of the screen.  Here, you can see the current playlist, easily view upcoming tracks, and navigate through the songs.

### Music Playback and Rating

When music is playing, you'll find a control panel at the bottom of the screen. This panel allows you to pause, resume, skip tracks, and manage your listening session.

To personalize your music experience, you can rate the currently playing song using the star rating bar on the right.  Your rating is saved directly into the local database and becomes a key factor in training your personal music evaluation model.


### Training the Music Evaluator Model 

To enable proper recommendations, you will need to train your personal music evaluation model to understand your musical taste on a deeper level. After you've rated a good number of songs, go to the 'Train' page and press the 'Train music evaluator' button.

This initiates the training of your personal music evaluation model.  Once training is complete, the 'rating' and ‘recommendation’ filters will become more finely tuned to your unique preferences, offering a truly personalized music journey. When you see a music that is not rated as you would expect, you can always adjust the rating and retrain the model to reflect your specific tastes. Making the system more accurate and personalized over time.

This process is highly optimized and could be easily perform in a couple of minutes on almost any modern hardware.

### Recommendation Engine

In contrast to many other centralized services, the recommendation algorithm in Anagnorisis is completely open and transparent. Here exactly how it works:

#### Internal Rating
Anagnorisis's music recommendation system keeps two scores for each song: your **user rating** and an **internal score** that adapts based on your listening behavior. To calculate this **internal score** the engine first calculates the **internal rating** of the song.  If **user rating** provided, the system prioritizes that. If not, it uses a **model-predicted rating** and in case even that score is not available it uses average of all other user-provided ratings.

#### Adjusted Rating
To keep the playlist fresh and engaging, the system adjust the rating before using it in further calculations. First, it ensures no song has a zero rating by setting a minimum score making each song having a chance of being played. Then, to emphasize highly-rated songs, it squares the adjusted rating. This adjusted rating becomes a key factor in the final recommendation score

#### Skip Score
Beyond that system keeps track of how often you skip a song and how ofter you listen it fully. Both of these numbers are then used to calculate the *skip score* that increase a change of recommending the song that you actually like to listen.

#### Last Played Score
In the end, the system also takes into account when you last listened to a song. The longer it has been since you last heard it, the more likely it is to going to be recommended, and vice verse.

#### Final Score
All of these scores are then multiplied together to calculate the final **internal score** and the respective selection probability for each song. 


### Project Status
Anagnorisis is under active development, with continuous improvements and new features on the horizon. You can find the link to the GitHub repository in the description below.

Visit the Anagnorisis GitHub repository to download the project, and stay updated on the latest enhancements.

