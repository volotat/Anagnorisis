* Clone the current repository state to a test folder.
```
mkdir -p ../Anagnorisis-test
git ls-files -z | rsync -av --files-from=- --from0 ./ ../Anagnorisis-test
```

* Go to the test folder.
```
cd ../Anagnorisis-test
```

* Set up .env file so the proper folders are used for testing.
```
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

* Run the Docker container with the test .env file.
```
docker-compose up -d
```

* Wait until all the models are downloaded and the application is fully started. Watch the progress in the `logs/{CONTAINER_NAME}_log.txt` file. Or even better break the downloading process by stopping the container and make sure that all the corrupted models are correctly identified and re-downloaded upon the next start.

* Access the application at http://localhost:5001 (or whichever port you configured) in your web browser.

* Check that all the modules are opens and show their files correctly.

* After opening all the modules, check the logs to make sure there were no silent errors.



1. Download fresh install of the project from the git into a new folder to avoid any prior data in the folder affecting the test results.