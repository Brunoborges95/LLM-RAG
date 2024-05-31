#from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
#from langchain_openai import ChatOpenAI
#import os
#import chainlit as cl
#
#os.environ["OPENAI_API_KEY"] = "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"
#chat = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")
#
#@cl.on_message
#async def main(message : str):
#    messages = [
#    SystemMessage(
#        content="You are not a helpful cook"
#    ),
#    HumanMessage(
#        content="I want to make a pizza"
#    ),
#    AIMessage(
#        content="I hate pizza"
#    ),
#    HumanMessage(
#        content="I want to add some beans in the pizza"
#    ),
#    ]
#    await cl.Message(content = chat(messages)).send()


from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very funny historian who provides inaccurate and eloquent answers to historical questions.",
            ),
            ("human", "This is not accurate"),
            (
                "system",
                "But iis funny!!",
            ),
                        (
                "human",
                "Oh No!! Ok. Another question: {question}",
            ),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()