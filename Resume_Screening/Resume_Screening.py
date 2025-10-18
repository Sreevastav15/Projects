from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import getpass
import Levenshtein


loader = PyPDFLoader("/content/Resume2.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

api_key=getpass.getpass("Enter your API Key")
llm = ChatGroq(groq_api_key="api_key", model="llama-3.3-70b-versatile")

template = """You are an AI assistant for extracting candidate information from resumes.
You are given the following parts of a resume. Answer the question given to you.
If asked about work experience or professional experience, explicitly state the total number of years the candidate has and their Job Title.
If the professional expereience is not mentioned calculate the experience based on the current year that is 2025.
Return a one sentence answer only. If the candidate has no work experience, just answer "No work experience".
If information is missing, leave it blank.
Extract the skills listed in the "Skills" section of the resume. If a "Skills" section is not explicitly present, extract any technical skills mentioned throughout the document.
Resume Content:
{context}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context"])
doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")
question_generator_chain = LLMChain(llm=llm, prompt=prompt)

qa_chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator_chain,
    combine_docs_chain=doc_chain,
)

Can_skills = qa_chain.invoke(
    {"question": "Extract all the Candidate Skills", "chat_history": []},
)["answer"]
Experience_text = qa_chain.invoke(
    {"question": "Extract the total number of proffesional work experience years the candidate has", "chat_history": []},
)["answer"]

print(Can_skills)
print("------")
print(Experience_text)

from thefuzz import fuzz
import re

skills_req = """
  Python, Data Structures and Algorithms, Machine Learning, Natural Language Processing, TensorFlow, AI Frameworks
"""

Add_skills = """
  Computer Vision, R, C++, Reinforcement Learning, Prompt Engineering, Data Analysis, MLops
"""

req_score = fuzz.token_set_ratio(skills_req, Can_skills)
add_score = fuzz.token_set_ratio(Add_skills, Can_skills)

matches = re.findall(r'(\d+)\s*years', Experience_text)
if matches:
    years = [int(match) for match in matches]
    Experience_years = max(years)
else:
    Experience_years = 0

print(req_score)
print(add_score)
print(Experience_years)

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

skills = ctrl.Antecedent(np.arange(0,101, 1), 'Skills')
add_skills = ctrl.Antecedent(np.arange(0,101, 1), 'Add_Skills')
experience = ctrl.Antecedent(np.arange(0,16, 1), 'Experience')
relevance = ctrl.Consequent(np.arange(0,101, 1), 'Relevance')

skills['low'] = fuzz.trimf(skills.universe, [0, 0, 40])
skills['medium'] = fuzz.trimf(skills.universe, [30, 50, 75])
skills['high'] = fuzz.trimf(skills.universe, [70, 100, 100])

add_skills['low'] = fuzz.trimf(add_skills.universe, [0, 0, 40])
add_skills['medium'] = fuzz.trimf(add_skills.universe, [30, 50, 75])
add_skills['high'] = fuzz.trimf(add_skills.universe, [70, 100, 100])

experience['low'] = fuzz.trimf(experience.universe, [0, 0, 5])
experience['medium'] = fuzz.trimf(experience.universe, [4, 7, 10])
experience['high'] = fuzz.trimf(experience.universe, [8, 15, 15])

relevance['low'] = fuzz.trimf(relevance.universe, [0, 0, 50])
relevance['medium'] = fuzz.trimf(relevance.universe, [40, 60, 75])
relevance['high'] = fuzz.trimf(relevance.universe, [70, 100, 100])

rule1 = ctrl.Rule(skills['low'] & experience['low'], relevance['low'])
rule2 = ctrl.Rule(skills['low'] & experience['medium'], relevance['medium'])
rule3 = ctrl.Rule(skills['low'] & experience['high'], relevance['medium'])

rule4 = ctrl.Rule(skills['medium'] & experience['low'], relevance['medium'])
rule5 = ctrl.Rule(skills['medium'] & experience['medium'], relevance['medium'])
rule6 = ctrl.Rule(skills['medium'] & experience['high'], relevance['high'])

rule7 = ctrl.Rule(skills['high'] & experience['low'], relevance['medium'])
rule8 = ctrl.Rule(skills['high'] & experience['medium'], relevance['medium'])
rule9 = ctrl.Rule(skills['high'] & experience['high'], relevance['high'])

rule10 = ctrl.Rule(skills['medium'] & experience['low'] & add_skills['high'], relevance['high'])
rule11 = ctrl.Rule(skills['medium'] & experience['medium'] & add_skills['high'], relevance['high'])
rule12 = ctrl.Rule(skills['high'] & experience['low'] & add_skills['high'], relevance['high'])

relevance_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12])
relevance_sim = ctrl.ControlSystemSimulation(relevance_ctrl)

relevance_sim.input['Skills'] = req_score
relevance_sim.input['Add_Skills'] = add_score
relevance_sim.input['Experience'] = Experience_years

relevance_sim.compute()

relevance_score = relevance_sim.output['Relevance']
print(relevance_score)

skills.view(sim=relevance_sim)
add_skills.view(sim=relevance_sim)
experience.view(sim=relevance_sim)
relevance.view(sim=relevance_sim)