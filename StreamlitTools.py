#importing libraries
import streamlit as st
import numpy as np
import os
import pandas as pd 
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from spacy.matcher import PhraseMatcher
import skillNer
import plotly
import plotly.graph_objs as go
from streamlit_option_menu import option_menu
from collections import Counter
from plotly import tools
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

im = Image.open("Forward-MENA-logo.png")

st.set_page_config(
    page_title="ForsaTech Skilling Tools",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


#Navigation Bar
Menu = option_menu(None, ["Home", "Scraping","Skills Extraction","Score Matching", "Comparative Analysis"],icons=['house','pen','search',"code", "code"],menu_icon="cast", default_index=0, orientation="horizontal", styles={"container": {"padding": "0!important", "background-color": "#B0C4DE"},"icon": {"color": "black", "font-size": "25px"}, "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},"nav-link-selected": {"background-color": "#4F6272"},})



#Setting Conditions 
video_file = open('FMQ_Digital-World_Music-only-2.mp4', 'rb')
video_bytes = video_file.read()
if Menu == "Home": st.video(video_bytes)
if Menu == "Home": st.title("FIND OUT ABOUT THE TOP SKILLS NEEDED IN THE MARKET TODAY!")
if Menu == "Home":st.write("77% believe there is a gap between the job requirementsand the university graduate qualifications. As the world moves at a rapid pace towards digitalization, the gap  is exponentially growing. Accordingly, we have taken on the role of filling this gap.")
if Menu == "Home":st.write("We will help you learn about the most in-demand careers, assess your skills, get certified and find the right job. To do so, please surf our offered tools to go through a career aspiration path.")

if Menu== "Scraping": 
    import time
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from webdriver_manager.chrome import ChromeDriverManager

    job_descriptions, job_titles, Positions = [], [], []

    # function to run code based on input
    def scrapeWeb(query, website):
        driver = webdriver.Chrome(ChromeDriverManager().install())
        if dropdown_value == "Bayt.com":
            # code A logic
            base_url = "https://www.bayt.com/en/"
            # web page url
            driver.get(base_url)
            time.sleep(1)
            # input search query
            ## look for search bar location
            search_bar = driver.find_element(By.ID, "text_search")
            ## enter text
            search_bar.send_keys(query)
            time.sleep(1)
            ## look for search button loacation
            search_button = driver.find_element(By.ID, "search_icon_submit")
            ## click button
            search_button.click()
            time.sleep(1)
            # get job urls
            job_urls = driver.find_elements(By.XPATH, "//a[@data-js-aid='jobID']")
            job_urls = [elem.get_attribute("href") for elem in job_urls]

            # loop over urls
            
            for url in job_urls:
                # open url
                driver.get(url)
                
                try:
                    # get job description
                    job_description = driver.find_element(By.XPATH, "//div[@class='card-content is-spaced']")
                    job_description = job_description.text
                except:
                    break
                
                # get job title
                job_title = driver.find_element(By.ID, "job_title")
                job_title = job_title.text

                # append to list
                job_descriptions.append(job_description)

                # append to list
                job_titles.append(job_title)


                # append to list
                Positions.append(query)

                # sleep
                time.sleep(1)
                
                
            output_data = pd.DataFrame(data={"Positions":Positions, "job title":job_titles, "job description":job_descriptions})
            
            return output_data
            
        elif dropdown_value == "DaleelMadani.com":
                # code B logic
            base_url = "https://daleel-madani.org/jobs/"
            
            driver.get(base_url)
            time.sleep(1)
        
            # input search query
            ## look for search bar location
            search_bar = driver.find_element(By.ID, "edit-search-api-views-fulltext")
            ## enter text
            search_bar.send_keys(query)
            time.sleep(1)
            ## look for search button loacation
            search_button = driver.find_element(By.ID, "edit-submit-jobs-index-dm-jobs")
            ## click button
            search_button.click()
            time.sleep(1)

            # get job urls
            job_urls = driver.find_elements(By.XPATH, "//div[@class='field-item even']//h4//a")
            job_urls = [elem.get_attribute("href") for elem in job_urls]

            # loop over all jobs

            for url in job_urls:
                # open url
                driver.get(url)

                # get job description
                job_description = driver.find_element(By.XPATH, "//div[@class='field field-name-body field-type-text-with-summary field-label-above']")
                job_description = job_description.text

                # get job title
                job_title = driver.find_element(By.XPATH, "//div[@class='field-item even']//h1")
                job_title = job_title.text

                # append to list
                job_descriptions.append(job_description)

                # append to list
                job_titles.append(job_title)

                # append to list
                Positions.append(query)

                # sleep
                time.sleep(1)

            output_data = pd.DataFrame(data={"Positions":Positions, "job title":job_titles, "job description":job_descriptions})
            return output_data
        
        elif dropdown_value == "jobsforlebanon.com":
            base_url = "https://www.jobsforlebanon.com"
            driver.get(base_url)
            time.sleep(1)

            # input search query
            ## look for search bar location
            search_bar = driver.find_element(By.ID, "ui-search-terms")
            ## enter text
            search_bar.send_keys(query)
            time.sleep(1)
            ## look for search button loacation
            search_button = driver.find_element(By.XPATH, "//input[@type='submit']")
            ## click button
            search_button.click()
            time.sleep(1)

            # get job urls
            job_urls = driver.find_elements(By.XPATH, "//div[@class='catalogue-job']")
            job_urls = [elem.get_attribute("data-href") for elem in job_urls]
            
            for url in job_urls:
                # open url
                driver.get(url)

                # get job description
                job_description = driver.find_element(By.XPATH, "//div[@itemprop='description']")
                job_description = job_description.text

                # get job title
                job_title = driver.find_element(By.XPATH, "//main[@class='jobad-main job']//h1")
                job_title = job_title.text

                # append to list
                job_descriptions.append(job_description)

                # append to list
                job_titles.append(job_title)

                # append to list
                Positions.append(query)

                # sleep
                time.sleep(1)
            
            output_data = pd.DataFrame(data={"Positions":Positions, "job title":job_titles, "job description":job_descriptions})
            
            return output_data

        else:
            return None

    # set up streamlit app
    st.title('ForsaTech Job Scraping Tool:')

    # dropdown for user to choose code
    dropdown_value = st.selectbox('Select a website to scrape:', ['Bayt.com', 'DaleelMadani.com', 'jobsforlebanon.com'])

    # input text box for user to enter input
    input_text = st.text_input('Enter Job Position:')

    # Create a button to download the output file
    from io import BytesIO
    def download_button(df):
        output = BytesIO()
        excel_writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(excel_writer, sheet_name='Sheet1', index=False)
        excel_writer.save()
        processed_data = output.getvalue()
        return processed_data

    # search button to run code
    if st.button('Search'):
        # run code and display output
        output_data = scrapeWeb(input_text, dropdown_value)
        if output_data is not None:
            st.dataframe(output_data)
            processed_data = download_button(output_data)
            st.download_button(label='Download Results',
                        data=processed_data,
                        file_name='output.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


if Menu == "Skills Extraction": 

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Define a list of relevant skills
allSkillsdf = pd.read_excel("allSkillsdf - ST.xlsx")
allSkillsList = allSkillsdf["OrgSkill"]
#lemmatize and lower case the skills list
lemmatizer = WordNetLemmatizer()
allSkillsList = [lemmatizer.lemmatize(skill.lower()) for skill in allSkillsList]


#creating a function to extract skills and their counts from job descriptions
def get_skills(text, skills_list):
    all_skills_names, all_skills_counts = [] , []
    # Tokenize the job description text and remove stop words
    tokens = [word.lower() for word in word_tokenize(text) if word.lower() not in stopwords.words("english")]

    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Count the occurrences of each skill in the lemmatized tokens
    skills_counts = Counter()
    for i in range(len(lemmatized_tokens)):
        if lemmatized_tokens[i] in skills_list:
            # Handle single-word skills
            skills_counts[lemmatized_tokens[i]] += 1
        else:
            # Handle multi-word skills
            skill_concat = lemmatized_tokens[i]
            if i < len(lemmatized_tokens) - 1:
                skill_concat += " " + lemmatized_tokens[i+1]
                if skill_concat in skills_list:
                    skills_counts[skill_concat] += 1

    # Print the extracted skills and their frequencies
    for skill, count in skills_counts.items():
        all_skills_names.append(skill)
        all_skills_counts.append(count)

    return all_skills_names

#tiltle of tool
st.title("Skills Extraction Tool:")
#enter the job description
job_description = st.text_area('Enter the job description:')

if st.button('Run'):
    skillsExt = get_skills(job_description, allSkillsList)
    unique_skills = list(set(skillsExt))
    allSkillsdf = allSkillsdf.apply(lambda x: x.str.lower())
    allSkillsdf["skill"] = [lemmatizer.lemmatize(skill.lower()) for skill in allSkillsdf["OrgSkill"]]
    unique_skills_df = pd.DataFrame(unique_skills, columns=['skill'])
    skillsType = pd.merge(unique_skills_df, allSkillsdf, on='skill', how='left')
    skillsType = skillsType.applymap(lambda x: x.title())
    
    st.table(skillsType[["OrgSkill", "type"]])
    
    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=800, background_color='white', colormap='inferno', max_words=50).generate_from_text(' '.join(unique_skills))

    # Plot the WordCloud image
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(8,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot()



if Menu == "Score Matching": 
    # Load data from CSV file
    df = pd.read_excel('FinalSkills - used.xlsx')

    # Create list of unique job names
    job_names = df["Original JT"].unique().tolist()
    job_names.sort()

    # Sidebar widgets for selecting job name
    selected_job = st.sidebar.selectbox("Select a job", job_names)

    # Filter DataFrame based on selected job
    job_df = df[df["Original JT"] == selected_job]

    # Get the top 5 SS and top 5 HS repeated or counted skills in the DataFrame of the selected job
    ss_skills = job_df[job_df["type"] == "ss"]["OrgSkill"].value_counts().sort_values(ascending=False).index.tolist()
    hs_skills = job_df[job_df["type"] == "hs"]["OrgSkill"].value_counts().sort_values(ascending=False).index.tolist()

    # Dictionary to map proficiency level to multiplication factor
    proficiency_mapping = {"Beginner": 1/3, "Intermediate": 2/3, "Proficient": 1}

    # Sidebar widgets for selecting SS and HS skills with proficiency level
    select_ss_skills = st.sidebar.multiselect("Select the Soft Skills you acquire:", ss_skills, format_func=lambda x: x.title())
    ss_proficiency = {}
    for skill in select_ss_skills:
        proficiency = st.sidebar.radio(f"Select your proficiency level in  '{skill.title()}':", ["Beginner", "Intermediate", "Proficient"])
        ss_proficiency[skill] = proficiency

    select_hs_skills = st.sidebar.multiselect("Select the Hard Skills you acquire:", hs_skills, format_func=lambda x: x.title())
    hs_proficiency = {}
    for skill in select_hs_skills:
        proficiency = st.sidebar.radio(f"Select your proficiency level in  '{skill.title()}':", ["Beginner", "Intermediate", "Proficient"])
        hs_proficiency[skill] = proficiency

    # Calculate matching score
    total_ss_skills = len(ss_skills)
    total_hs_skills = len(hs_skills)
    total_skills = total_ss_skills + total_hs_skills
    matching_score = 0

    # Calculate score for selected SS skills
    for skill in select_ss_skills:
        proficiency = ss_proficiency[skill]
        if skill in ss_skills[:5]:
            matching_score += 0.065 * proficiency_mapping[proficiency]
        else:
            matching_score += (35 / (total_skills - 10) / 100) * proficiency_mapping[proficiency]

    # Calculate score for selected HS skills
    for skill in select_hs_skills:
        proficiency = hs_proficiency[skill]
        if skill in hs_skills[:5]:
            matching_score += 0.065 * proficiency_mapping[proficiency]
        else:
            matching_score += (35 / (total_skills - 10) / 100) * proficiency_mapping[proficiency]

    # Round matching score to 2 decimal places
    matching_score = round(matching_score * 100, 2)

    # Display job name as title and matching score in the center of the screen
    st.title("ForsaTech Matching Score Calculation Tool:")
    video_url = "https://youtu.be/YKACzIrog24"
    st.video(video_url)

    st.header(f"Matching score for {selected_job} is: {matching_score}%")

    if matching_score <= 65:
        st.write("To enhance your skills in this field, you can start a learning path course or training on our website and get certified!")
    else:
        st.write("Continue learning more skills to gain competitive edge! The sky is the limit.")


if Menu == "Comparative Analysis": 
    #title of the page
    st.title('Comparative Analysis:')
    st.subheader("ForsaTech vs UN Escwa Skills")

    # Load the Excel sheets into dataframes
    df1 = pd.read_excel("UN vs Forsatech HS.xlsx")
    df2 = pd.read_excel("UN vs Forsatech SS.xlsx")
    df3 = pd.read_excel("Matching % UN vs ForsaTech - st.xlsx")

    # Define a dropdown menu to select the column name
    options = ["All"] + sorted(list(df1.columns))
    column_name = st.selectbox("Select a Job Title:", options=options)

    if column_name != "All":
        # Filter the dataframes based on the selected column
        df1_filtered = df1[df1[column_name].notnull()]
        df2_filtered = df2[df2[column_name].notnull()]
        percentage_value = df3.loc[0, column_name]

        # Display the filtered dataframes in tables
        col1, col2 = st.columns(2)
        with col1:
            st.write('Missing Hard Skills:')
            st.table(df1_filtered[column_name])

        with col2:
            st.write('Missing Soft Skills:')
            st.table(df2_filtered[column_name])

        # Display the percentage value 
        st.subheader(f"{percentage_value:.0%} of skills are matching between ForsaTech and UN skills for {column_name}.")

        # Calculate the counts of rows in each table
        table1_count = len(df1_filtered[column_name])
        table2_count = len(df2_filtered[column_name])

        # Create a pie chart showing the percentage of rows in each table
        total_count = table1_count + table2_count
        fig = px.pie(values=[table1_count, table2_count], names=["Hard Skills", "Soft Skills"], title="Type of Missing Skills:")
        st.plotly_chart(fig)

    else:
        # calculate total length of all columns
        df1_length = sum([len(df1[col]) for col in df1.columns])
        df2_length = sum([len(df2[col]) for col in df2.columns]) 
        Total_length = df1_length + df2_length
    
        fig1 = px.pie(values=[df1_length, df2_length], names=["Hard Skills", "Soft Skills"], title="Total Missing Skills Type:")
        #st.plotly_chart(fig1)

        # merge all columns into one list, drop duplicates, and get the length
        unique_HS = len(df1.apply(pd.Series).stack().unique())
        unique_SS = len(df2.apply(pd.Series).stack().unique())
        Total_Unique = unique_HS + unique_SS

        # fig2 = px.pie(values=[unique_HS, unique_SS], names=["Hard Skills", "Soft Skills"], title="Total Unique Missing Skills Type:")
        # st.plotly_chart(fig2)

        fig2 = go.Figure(data=[go.Pie(labels=["Hard Skills", "Soft Skills"], 
                               values=[unique_HS, unique_SS], 
                               textinfo='percent+value',
                               hovertemplate='%{label}: %{value}<br>%{percent}',
                               text=['{}: {}'.format(l, v) for l, v in zip(["Hard Skills", "Soft Skills"], [unique_HS, unique_SS])])])

        fig2.update_layout(title="Total Unique Missing Skills Type:")
        #st.plotly_chart(fig2)


        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1)
        with col2:
            st.plotly_chart(fig2)


        HS_series = df1.stack().value_counts()  # flatten DataFrame to Series and get value counts
        top_10_HS = HS_series.head(10).index.tolist()  # get top 10 values as a list

        SS_series = df2.stack().value_counts()  # flatten DataFrame to Series and get value counts
        top_10_SS = SS_series.head(10).index.tolist()  # get top 10 values as a list


        st.subheader("Top 10 needed skills according to UN:")

        # Display the filtered dataframes in tables
        col1, col2 = st.columns(2)
        with col1:
            st.table(pd.DataFrame(top_10_HS, columns=['Hard Skills']))

        with col2:
            st.table(pd.DataFrame(top_10_SS, columns=['Soft Skills']))


        df4 = pd.read_excel("Forsatech SoftSkills.xlsx")
        df5 = pd.read_excel("Forsatech HardSkills.xlsx")

        FHS_series = df5.stack().value_counts()  # flatten DataFrame to Series and get value counts
        Ftop_10_HS = FHS_series.head(10).index.tolist()  # get top 10 values as a list

        FSS_series = df4.stack().value_counts()  # flatten DataFrame to Series and get value counts
        Ftop_10_SS = FSS_series.head(10).index.tolist()  # get top 10 values as a list


        st.subheader("Top 10 needed skills according to ForsaTech:")

        # Display the filtered dataframes in tables
        col3, col4 = st.columns(2)
        with col3:
            st.table(pd.DataFrame(Ftop_10_HS, columns=['Hard Skills']))

        with col4:
            st.table(pd.DataFrame(Ftop_10_SS, columns=['Soft Skills']))



            
