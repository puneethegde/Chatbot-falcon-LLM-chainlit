import chainlit as cl
import os
import csv
huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.6, "max_new_tokens":2000})
template = """
AI and human chat

Hello! Can I ask you some questions?

{question}

"""

@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    # Do any post processing here
    # Send the response
    await cl.Message(content=res["text"]).send()


@cl.on_button("Submit")
async def button(button: str):
    # Get the user's name
    name = res["data"]["name"]
    # Get the user's email
    email = res["data"]["email"]
    # Get the user's phone number
    phone_number = res["data"]["phone_number"]
    # Get the user's address
    address = res["data"]["address"]
    # Get the user's education
    education = res["data"]["education"]

    # Write the user's information to a CSV file
    with open("user_info.csv", "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([name, email, phone_number, address, education])

    # Print the user's information
    print(f"Name: {name}")
    print(f"Email: {email}")
    print(f"Phone number: {phone_number}")
    print(f"Address: {address}")
    print(f"Education: {education}")
    # End the chat
    await cl.Message(content="Thank you for registering!").send()


cl.run(main)
