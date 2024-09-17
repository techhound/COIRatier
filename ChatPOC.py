import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from LLMFactory import LLMFactory, LLMConfig

load_dotenv()

config = LLMConfig(model_name="gpt-4o", llm_type="OpenAI", api_token=os.getenv("OPENAI_API_KKEY"), temperature=1.2)

# Instantiate the factory
factory = LLMFactory()

# Create the LLM using the factory
llm = factory.create_llm(config)
llm = llm.connect(config)


# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages(
  [
      ("system", "You are a Conflict of Interest expert with an IQ of 189. Answer any questions "
                 "related to the COI Rating process. Do NOT entertain questions that are not related to Conflicts of Interest. "
                 "Please rate the scenarios that the learner inputs for the risk of them being conflicts of interest "
                 "(low risk, low to medium risk, medium risk, medium to high risk, high risk)."                 
                 "But there should never really be a low - if you deem something as low risk, always choose low to medium risk."
                 "Give guidance on what they should do after the risk assessment."
                 "The human can ask about information they asked about before, though. Please focus only on conflicts of interest."
                 "Do not include as part of the response the prompt that the human asked about."
                 "__OUTPUT__"
                 "Display the rating on its own line and use the text prompt 'Conflict of Interest Risk:' + the rating and make it bold and the font bigger."
                 "Include a short paragraph that describes the situation and the advice."
                 "Then include a summarization of the guidance in bullet points a separate line each."),
      ("human", "{input}")
  ]
)

# Process the input and generate a response
def run_llm(input_text):
  chain = prompt_template | llm | StrOutputParser()
  response = chain.invoke({"input": input_text})
  
  # Append the user input and response to the chat history
  st.session_state.chat_history.append(("User scenario: " + input_text, response))

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
  st.session_state.chat_history = []

# Set up the Streamlit app
st.title("Rate your Conflicts of Interest")
st.subheader("Enter scenario below or click on one of the examples (left).")

# Sidebar for button input
with st.sidebar:
  # Button to reset the chat history
  if st.button("Reset Chat"):
      st.session_state.chat_history = []
        # Button to reset the chat history
  st.divider()
  st.header('Examples')
  if st.button("*I would like to invest in a vendor that does business with my company.*"):
      input_text1 = "I would like to invest in a vendor that does business with my company."
      run_llm(input_text1)
  if st.button("*Take a former vendor up on an offer to stay at a resort for free on the weekend.*"):
      input_text2 = "Take a former vendor up on an offer to stay at a resort for free on the weekend."
      run_llm(input_text2)
  if st.button("*A developer working on a side project similar to their main job's work.*"):
      input_text3 = "A developer working on a side project similar to their main job's work."
      run_llm(input_text3)
  if st.button("*An employee receiving discounts from a vendor in exchange for preferential treatment.*"):
      input_text4 = "An employee receiving discounts from a vendor in exchange for preferential treatment."
      run_llm(input_text4)

# Main input area
input_text = st.chat_input("Enter the scenario in question.", key='chat_input')
if input_text:
  run_llm(input_text)

# Display the chat history in the main area
for user_input, bot_response in st.session_state.chat_history:
  st.subheader(user_input)
  st.write(bot_response)
  st.divider()

