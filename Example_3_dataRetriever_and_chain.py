from resources import constants

from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.callbacks import get_openai_callback


# Loads personal texts for model
loader = TextLoader('PATH-TO-YOUR-TEXT-FILE.txt')
documents = loader.load()

# Text splitters are needed to split long texts into chunks
# Must experiment to figure out the best chunk_size and chunk_overlap! (params of RecursiveCharacterTextSplitter)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Creating vectorstore index is needed for model to efficiently look through your provided data
# Essentially it is changing the text into a vector that becomes a series of numbers that hold the semantic 'meaning' of the text
# There are many different vector stores, now we use Facebook AI Similarity Search (FAISS)
embeddings = OpenAIEmbeddings(openai_api_key=constants.APIKEY) # ADD OWN constants.py DOCUMENT IN PROJECT WITH YOUR OPENAI API KEY!
vector_db = FAISS.from_documents(texts, embeddings)

retriever = vector_db.as_retriever()



query = 'ADD HERE WHAT YOU WANT TO ASK THE MODEL'

####################################################################
# If you want you can print some stuff to find out more about the models behavior in relation to the query:
docs = retriever.get_relevant_documents(query)
print('Most relevant docs to your query:')
print("\n\n".join([x.page_content[:200] for x in docs[:5]]))
# Read a bit about embeddings in lanchain docs: https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/
text_embedding = embeddings.embed_query(query)
print(f"Here's a sample of the first 5 embeddings resulting from your query, holding numbers: {text_embedding[:5]}...")
print (f"The complete embedding length is {len(text_embedding)}")
#####################################################################


# The template can be used to give instructions on how to answer
# (A lot of variety possible here, learn a bit from; https://github.com/gkamradt/langchain-tutorials/blob/main/LangChain%20Cookbook%20Part%201%20-%20Fundamentals.ipynb )
# Leave in {summaries} and {question} for these input variables must be in the prompt template the chain that will be built after
template = """
ADD HERE THE CONTEXT FOR YOUR MODEL 
{summaries}
{question}
"""


# Yet to determine optimal chain_type (https://docs.langflow.org/components/chains#:~:text=chain_type%3A%20The%20chain%20type%20to,that%20prompt%20to%20an%20LLM.)
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=ChatOpenAI(openai_api_key=constants.APIKEY, model="gpt-3.5-turbo-1106", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=template,
            input_variables=["summaries", "question"],
        ),
    },
)


# For the chain we define a chat history that we will fill with every answer and loop back into new prompts, this way we can converse with the model
chat_history = []
# Openai Callback is not necessary but nice if you want to print statistics about the prompt later.
with get_openai_callback() as cb:
    result = chain.invoke(input={"question": query, "chat_history": chat_history})

print(result['answer'])
print('€' + str(cb.total_cost))
print('Answer generation costs: ' + str(cb.completion_tokens))
print('Prompt/Text input costs: ' + str(cb.prompt_tokens))
print('Total tokens spent: ' + str(cb.total_tokens))


# EXAMPLE of how to invoke a second question on this chain
result2 = chain.invoke(input={"question": 'ADD YOUR SECOND QUESTION HERE', "chat_history": chat_history})
print(result2['answer'])





#######################################################################################
#######################################################################################
# EXAMPLE: Retriever implemented in chain (but no Prompt Template), for more efficiency and less prompt control
'''
# We create a chain to keep feeding LLM own outputs and stay focussed
chain = ConversationalRetrievalChain.from_llm(
    llm = ChatOpenAI(openai_api_key=constants.APIKEY, model="gpt-3.5-turbo-1106"),
    retriever = retriever
)

chat_history = []
choice = input('Ask ChatGPT? (Y/N)')
if choice == 'Y':
    result = chain.invoke(input={"question": query, "chat_history": chat_history})
    print(result['answer'])
    chat_history.append((query, result['answer']))

    #Now you can chain.invoke(input= NEW_QUESTION, chat_history= chat_history)
'''


# EXAMPLE: Prompt template implemented in chain via pipeline (but no Retriever), for more prompt control but less efficiency:
'''
template = 'Only answer using three words'
prompt_template = ChatPromptTemplate.from_messages([
    ('system', template),
    ('human', query)
])
chatPromptTemplate = prompt_template.format()
pipelineChain = prompt_template.pipe(ChatOpenAI(openai_api_key=constants.APIKEY, model="gpt-3.5-turbo-1106"))
chat_history_pipelineChain = []
result = pipelineChain.invoke(input={"question": query, "chat_history": chat_history_pipelineChain})
print(result)
'''
#######################################################################################
#######################################################################################



