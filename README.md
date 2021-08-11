# SUTD 50.045 Information Retrieval Project 2021  
### Music Recommendation Engine  
Team: Lim Jun Hao, Teresa Liew, Zenn Png  

Dataset Used:  
https://data.mendeley.com/datasets/3t9vbwxgr5/2  

Usage:  
- pip install -r requirements.txt  
- Run App using 'Flask run', the file app.py will be run automatically by Flask server  
- Navigate to the webpage using the URL  
  (The app might take a while to start as the word2vec API is being loaded into the environment)
- Current model used in the web app: VSM
- Response time: ~ 5 seconds

Short Project Description:  
In this project, we aim to recommend songs to users based on their query. Using natural language, users will indicate their mood and the type of music that they would like to listen to. Upon processing their query, our IR system will use various models to retrieve songs that are deemed to be relevant as recommendations for the user.  

