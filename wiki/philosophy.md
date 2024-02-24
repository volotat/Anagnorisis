# Local Recommendation Engine 

This project is an attempt to enhance the efficiency and depth of data analysis in the current era of abundant online information. Historically, individuals have relied on major corporations to provide tools for sorting through vast amounts of information via recommendation engines. However, these recommendation engines have been centralized within large corporations, inaccessible and unmodifiable by end-users.

A notable shift in recent years has been the significant advancements in Machine Learning, particularly the rise of multi-modal models. This progress now allows for the transfer of recommendation engines from big clusters to end users' machines. This is achieved by training models locally to estimate user preferences without depending on distant servers. The primary objective of this project is to bring this vision to life by providing a toolkit with specific functionalities:

* Automating the extraction of data from diverse sources, including websites, a user's hard drive, and potentially through peer-to-peer networks in the future.
* Utilizing machine learning models to filter and sort data according to the user's preferences.
* Implementing user interface for rating data, facilitating easy and intuitive feedback.
* Associating user feedback with the original data, establishing a valuable loop for continuous improvement.
* Enabling the local training of a model tailored to the user's preferences.

![anagnorisis data flow chart](/static/anagnorisis_data_flow_chart.png)

At this stage, the project is in the draft phase. 

The foundational philosophy of the project revolves around two core principles: the retention and control of user data. This approach empowers individuals to leverage their data exclusively for their unique requirements.

In the context of aligning AI models, a crucial question arises: "Whose values should the model reflect?" The answer put forth by this project – your values. In striving for a perfect representation of individual values, the model may develop an internalized understanding of the user, potentially preserving their personality for future exploration with sufficient data.

The concept of anagnorisis underscores the creation of a system, akin to an agent, capable of autonomously navigating vast datasets on behalf of the user. Its purpose is to filter, process, analyze, and present only the most pertinent information in a manner tailored to the end user's preferences and needs.

## Recommendation-engine approaches
There are two main ways to create AI recommendation engine. First is modular one where we have multiple models for each type of data that extracts embeddings of the data and then use this embeddings as a way to find similarity between known and unknown datapoints. The other method, that I would really prefer to use is to have single multi-modality model that is fine-tuned on the known data and it's scores and used to predict similarities or new scores directly as a simple text output from the model. The second one is much more flexible and interactable, while the first one is much much less resource hungry. That's why in this project where it is possible I will try to use general model in all available cases and in other cases use embedding-based approach as a temporal solution.

The multi-modal solution is also preferable as with enough about of data it could create some sort of model of the person it is trying to imitate.