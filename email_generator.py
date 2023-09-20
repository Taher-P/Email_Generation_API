import os
import uvicorn
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers.utils.hub import move_cache
from ctransformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline as hf_pipeline
from huggingface_hub import HfApi
from fastapi import FastAPI, HTTPException, Request
from typing import List
import nest_asyncio
from pydantic import BaseModel,model_validator,ValidationError,root_validator
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from abc import ABC
import nest_asyncio
from fastapi.middleware.cors import CORSMiddleware

# Here we Setup the Model which includes [Model, Tokenizer, Pipeline] #and End the email with {signature}.

# OG Model = "TheBloke/Llama-2-7B-Chat-GGML", model_file='llama-2-7b-chat.ggmlv3.q4_K_S.bin', hf=True
nest_asyncio.apply()
app = FastAPI()
def setup_model():

    model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML", model_file='llama-2-7b-chat.ggmlv3.q4_K_S.bin', hf=True)

    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=250, do_sample=True, temperature=0.8, repetition_penalty=1.1, device_map="auto")

    llm = HuggingFacePipeline(pipeline=pipeline)

    return llm

llm = setup_model()
# Define a Pydantic model for request data (input command)
class BasePromptTemplate(BaseModel):
    @root_validator(pre=True,skip_on_failure=True)
    def validate_variables(cls, values):
        if not values.get('template'):
            return values
        return values
class EmailRequest(BaseModel):
    client_email: str
    client_name: str
    client_company: str
    email_description: str

def generate_email(client_email, client_name,client_company,email_description): # Here we define the PromptTemplate and setup LLMchain for generating Email Script based on the Input.

    template = """
    You are an Intelligent Email Generator Tool write a Business Marketing Email to our Client Name:{client_name} for there company Client Company:{client_company} with the email address as {client_email} based on the following Email description:{email_description}. Strictly start the email with Subjectline an Greetings. Keep email breif and under 250 Words.
    """
    prompt = PromptTemplate(template=template, input_variables=['client_email','client_name','email_description','client_company'])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    generated_email = llm_chain.run({'client_email': client_email, 'client_name': client_name,'client_company':client_company,'email_description': email_description,})

    return generated_email

# Define a route to generate an email based on input data
@app.post("/generate-email/")
def generate_email_endpoint(request_data: EmailRequest):
    try:
        # Get the input data from the request
        client_email = request_data.client_email
        client_name = request_data.client_name
        client_company = request_data.client_company
        email_description = request_data.email_description

        # Use the generate_email function to generate the email content
        generated_email = generate_email(client_email, client_name, client_company, email_description)

        return {"generated_email": generated_email}
    except Exception as e:
        return {"error": str(e)}

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 12000))
    uvicorn.run(app, host="0.0.0.0", port=port)
