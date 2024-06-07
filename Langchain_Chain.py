from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
)


class Chain:
    def __init__(self, llm, retriever, memory=None):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory

    def RetrievalChain_without_prompt(self, max_tokens_limit=1000):
        if self.memory is not None:
            return ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=self.retriever,
                memory=self.memory,
                verbose=True,
                max_tokens_limit=max_tokens_limit,
            )
        else:
            return ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=self.retriever,
                verbose=True,
                max_tokens_limit=max_tokens_limit,
            )

    def RetrievalChain_with_prompt(
        self, message_template, context_template, max_tokens_limit=1000
    ):
        general_system_template = message_template + context_template
        general_user_template = "Question:```{question}```"
        messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template),
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        if self.memory is not None:
            return ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=self.retriever,
                memory=self.memory,
                verbose=True,
                max_tokens_limit=max_tokens_limit,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
            )
        else:
            return ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=self.retriever,
                verbose=True,
                max_tokens_limit=max_tokens_limit,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
            )

    def RetrievalChain_with_few_shot_examples(
        self,
        few_shot_examples,
        message_template,
        context_template,
        max_tokens_limit=1000,
    ):
        general_system_template = message_template + context_template
        general_user_template = "Question:```{question}```"
        few_shot_template = ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        )
        messages = [
            FewShotChatMessagePromptTemplate(
                example_prompt=few_shot_template, examples=few_shot_examples
            ),
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template),
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        if self.memory is not None:
            return ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=self.retriever,
                memory=self.memory,
                verbose=True,
                max_tokens_limit=max_tokens_limit,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
            )
        else:
            return ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=self.retriever,
                verbose=True,
                max_tokens_limit=max_tokens_limit,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
            )
