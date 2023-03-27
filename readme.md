# Web Scraped Job Board and Job Clustering
---
# Interactive dashboard: [click here](https://asteriosds-web-scraped-job-board-job-board2-z98535.streamlit.app/)
---
## Project Description:
---
### I built a scraper with bs4 which scraped any Data Analyst, Scientist and Engineer jobs. Then I pushed them in my local postgres db. I deployed these jobs as a job board with the help of Streamlit. I then wanted to automate the scraper to scrape, update the db and the dashboard. I did this by building a docker image of my local db, pushed it to Docker Hub and then built a workflow in Github Actions which starts by downloading the image, spinning up a container, running the scraper, pushing csvs to the repo and saving and pushing the image back to Docker Hub. So the automated job board was over. Recently I extracted the job skills from the job ads (NER) and clustered them based on their similarity with the help of Word2vec and KMeans. I then perforn dimensionality reduction with t-SNE to plot the results. Results are not satisfying yet, next step is to build a second scraper to scrape another website and gather more job data. Enjoy!

---
### Contents

- Jobs.ipynb
- auto_req.txt
- avg.pkl
- df_c.csv
- embeddings_3d.csv
- job_board.csv
- job_board.py
- requirements.txt
- skill_df.csv
- skill_db_relax_20.json
- token_dist.json

### Details:
1. #### **Jobs.ipynb** contains the web scraper, the functions built for accesing the database, pushing jobs into the database (only if they are new listings), fetching data from the database and cleaning it.
2. #### **auto_req.txt** is the file needed to set up the libraries for the Github Actions vm.
3. #### **avg.pkl** Dictionary containing num_clusters as keys and silhouette scores as values, from KMeans.
4. #### **df_c.csv** Dataframe containing the data for clustering.
5. #### **embeddings_3d.csv** Dataframe with the reduced vector representations from word2vec.
6. #### **job_board.csv** is the csv file created by the aforementioned pipeline.
7. #### **job_board2.py** is the python file that the dashboard is built on.
8. #### **requirements.txt** is the file with the environment requirements needed for Streamlit deployment.
9. #### **skill_df.csv** Dataframe containing the skills extracted for each job along with their cluster assignment.
10. #### **skill_db_relax_20.json** and **token_dist.json** json files needed for the skillNer library/db.
