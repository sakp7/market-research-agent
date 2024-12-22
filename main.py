import os
import requests
import json
import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from bs4 import BeautifulSoup

GROQ_API_KEY = st.secrets['groq_api']
SERPER_API_KEY = st.secrets['serper_api']

client = ChatGroq(api_key=GROQ_API_KEY)

def search_market_research_tool(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()
        articles = []
        if 'organic' in data:
            for item in data['organic']:
                article = {
                    'title': item.get('title', 'No title available'),
                    'link': item.get('link', 'No link available'),
                    'snippet': item.get('snippet', 'No snippet available')
                }
                articles.append(article)
        return articles
    except requests.exceptions.RequestException as e:
        st.write(f"Error occurred during API call: {e}")
        return []

def fetch_article_content(url):
    content = ""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            content += heading.get_text() + "\n"
        for paragraph in soup.find_all('p'):
            content += paragraph.get_text() + "\n"
        return content.strip()
    else:
        return ""

def collect_datasets_tool(company_name):
    datasets = search_market_research_tool(f"{company_name} AI datasets")
    valid_sources = ["kaggle.com", "huggingface.co", "github.com"]
    dataset_links = []

    for dataset in datasets:
        if any(source in dataset['link'].lower() for source in valid_sources):
            dataset_links.append(f"[{dataset['title']}]({dataset['link']})")
    
    if dataset_links:
        return "\n".join(dataset_links)
    else:
        return "No relevant datasets found from Kaggle, Hugging Face, or GitHub."

def generate_use_case_tool(company_name, industry):
    prompt = f'''
        Based on the following market research, propose GenAI solutions for {company_name} in the {industry} industry, like document search, automated report generation, and AI-powered chat systems for internal or customer-facing purposes.
     
    '''
    response = client.invoke([("system", "You are an assistant that helps generate AI use cases."), ("human", prompt)])
    return response.content.strip()

def generate_report(company_name, industry):
    use_case_results = generate_use_case_tool(company_name, industry)
    datasets = collect_datasets_tool(company_name)
    market_research = search_market_research_tool(f"trends in {industry} industry in india")

    report_content = f"AI Use Case Report for {company_name}\n\n"
    report_content += f"Company Name: {company_name}\n"
    report_content += f"Company Industry: {industry}\n\n"
    report_content += f"Market Research for {industry} Industry:\n"
    for research in market_research:
        report_content += f"- {research['snippet']} ({research['link']})\n"
    report_content += f"\nGenerated Use Cases:\n{use_case_results}\n\n"
    report_content += f"Generated Links for Use Cases:\n{datasets}\n"

    return report_content

st.title("AI Use Case Report Generator")
company_name = st.text_input("Enter the company name:", "Swiggy")
industry = st.text_input("Enter the company industry (e.g., e-commerce, food-delivery):", "food-delivery")

if st.button("Generate Report"):
    if company_name and industry:
        with st.spinner("Processing your request..."):
            report_content = generate_report(company_name, industry)
            st.write(report_content)
            st.download_button("Download Report", report_content, file_name=f"{company_name}_use_case_report.txt", mime="text/plain")
    else:
        st.write("Please enter a valid company name and industry.")
