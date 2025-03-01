Welcome to the ‘Anagnorisis’ 'Images' module. ‘Anagnorisis’ is an open-source local recommendation system designed to let you search, filter and process your personal data, securely on your local machine with the help of an AI-based recommendation engine that learns from your feedback.

First, set up your image library path. Copy your local image folder path into the 'Path to local image library' field and press the ‘Save’ button. This initiates a process that gathers image hashes, setting up the caching system.

Once this is done, click on any image to enlarge it. You’ll see a rating bar at the bottom, allowing you to rate images from 0 to 10. Ratings are stored in the database for later use in training your personalized image evaluation model.

The module supports text-based semantic image search. Enter any text describing an image you want to find and press the ‘Search ‘button. The system will retrieve images most relevant to your query.

There are multiple filtering options included in the module. For example ‘Text search’ might be used to filter out categories like 'NSFW' or other specific categories of images.

‘Similarity filtering’ groups visually similar images, allowing you to quickly identify and remove duplicates and see clusters of images that are presented in your image library.

‘Resolution’ filtering might help you find images that do or do not meet your size requirements.

‘Proportion’ filtering provides a clear way to go from wide to tall images in your dataset.

‘File size’ filtering might help you find and eliminate excessively large or small files.

‘Random’ filtering enables quick dataset exploration.

And finally, ‘Rating-based’ filtering is effective for excluding low-quality or unwanted images once your model is trained, prioritizing your ratings when available, or using model predictions when not. 

To train the model, you have to rate some amount of images, proceed to the ‘Train’ page and press ‘Train image evaluator’ button. After training, return to the ‘Images’ page. Now the system will rate new images as if it was rated by you. If you do not agree with the rating you can simply change it and train the model again, getting the model that is more and more aligned to your preferences., 

Thank you for your attention. Till the next time.