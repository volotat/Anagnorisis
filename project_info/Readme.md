The data in this folder is a collection of facts and trivia about the project so one can easily ask questions about the projects via LLMs or similar AI tools. An example of such question-answer script may be found in the 'ask.py' file. You can run this file with the command:
```bash
cd project_info
python ask.py
```
It will combine all the codebase into a single string as well as current diff and send it to Google's "gemini-2.0-flash-exp" model alongside with the question. You have to provide your own key in GEMINI_API_KEY environment variable. This might be very helpful for anyone who would like to participate in the development of the project or just to know more about it.