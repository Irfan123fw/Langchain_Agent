from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool
import chainlit as cl
from dotenv import load_dotenv
load_dotenv()
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.vectorstores import Chroma

import os
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, max_output_tokens=2048, convert_system_message_to_human=True)

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["text/plain"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    with open(file.path, "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text)

    # load it into Chroma and save it to disk
    db = Chroma.from_texts(chunks, embeddings, collection_name="groups_collection", persist_directory="Brown"
                            )

    retriever = db.as_retriever(search_kwargs={"k": 5})

    tool = create_retriever_tool(
        retriever,
        "search_state_of_union",
    "Searches and returns documents regarding the state-of-the-union.",
    )
    tools = [tool]
    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()
    cl.user_session.set("tools", tools)

@cl.on_message
async def main(message: cl.Message):
    tools = cl.user_session.get("tools")
    agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True, remember_intermediate_steps=True,
                                                        memory_key="chat_history")
    cb = cl.AsyncLangchainCallbackHandler()

    result = agent_executor({"input": message.content}, callbacks=[cb])

    answer = result["output"]
    await cl.Message(content=answer).send()
