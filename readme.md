# Web Scraped Job Portal Project
---
# Interactive dashboard: [click here](https://asteriosds-web-scraped-job-board-job-board-h41of5.streamlit.app/)
---
## Project Description:
---
### This project acts as a precursor to a full stack data science project. In a full stack project typically you ping an api or you scrape the web for data. Here, I have built a webscraper using BeautifulSoup4 and requests-html that scrapes a job website looking for data related jobs. Then you typically push the data in a database onsite or on the cloud. I pushed it on an onsite postgres db to access it later. Then you fetch the data from the database, clean them and export them. The last step is to deploy the insights or the model you built from the cleaned data. Streamlit was used to deploy the insights generated.

---
### Contents

- Job Portal (Webscraping_postgress_cleaning).ipynb
- job_board.csv
- job_board.py
- requirements.txt

### Details:
1. #### **Job Portal (Webscraping_postgress_cleaning).ipynb** contains the web scraper, the functions built for accesing the database, pushing jobs into the database (only if they are new listings), fetching data from the database and cleaning it.
2. #### **job_board.csv** is the csv file created by the aforementioned pipeline.
3. #### **job_board.py** is the python file that the dashboard is built on.
4. #### **requirements.txt** is the file with the environment requirements neede for Streamlit deployment.
