# import statements

!pip install openai langchain langchain-community langchain-core tiktoken
!pip install -U langchain-openai

from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import tiktoken
import openai
from openai import OpenAI
import os
import getpass
import pandas as pd

# read csv and remove extra custom stopwords that threw the model off when present
df = pd.read_csv('/content/topics_for_2022.csv', engine='python', on_bad_lines='skip')
df['text'] = df['text'].apply(lambda x: x.replace("[deleted]", "").replace("[removed]", "").strip())

# read OpenAI API key, security purposes
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter key: ")

# model definition
llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-3.5-turbo-1106", max_tokens=100, temperature=0, top_p=1.0, seed=3)

# function that defines prompt, called for each post, extracted only content from API response
def generate_summary(text):
  template = """Write a 1 sentence summary of the following text and identify the major and secondary themes, the speaker, relationship of speaker to person being referred to, major emotion conveyed, and impact on psychological state:
            {text}
            """
  prompt_template = PromptTemplate(template=template, input_variables=["text"])
  return llm(prompt_template.format(text=text)).content

df['post_summaries'] = df['text'].apply(generate_summary)

df.to_csv('post_summaries_2022.csv')