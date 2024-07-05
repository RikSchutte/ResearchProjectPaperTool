import os
import re

from resources import constants
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import PyPDFLoader
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from openpyxl import Workbook
from datetime import datetime



class SciPlow:
    costs_euro = 0
    costs_tokens = 0
    n_values = []

    def main(self):
        print('Starting')
        count = 0
        folder_path = 'resources/articles'
        files = sorted(os.listdir(folder_path), key=lambda x: int(x.split()[0]))

        for filename in files:
            if filename.endswith('.pdf'):
                count += 1
                if count > 150:
                    break

            if count in {17, 37, 38, 48, 64, 68, 70, 73}:


                print(filename)

                file_path = os.path.join(folder_path, filename)
                retriever = self.create_text_retriever(file_path)
                query = 'From the research paper text find the amount of participants who finished all experiments, do not include drop out participants.'
                self.query_text(retriever, query)

        print('FINISHED using: ' + str(self.costs_tokens) + ' tokens (€' + str(self.costs_euro) + ')')


    @staticmethod
    def create_text_retriever(file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=constants.APIKEY)
        vector_db = FAISS.from_documents(texts, embeddings)
        return vector_db.as_retriever()


    def query_text(self, retriever, query):
        template = """
        Answer by only giving the number, if you can not find an answer write null.
        {summaries}
        {question}
        """
        template = """
        If you can not find an answer write null.
        {summaries}
        {question}
        """

        # Yet to determine optimal chain_type (https://docs.langflow.org/components/chains#:~:text=chain_type%3A%20The%20chain%20type%20to,that%20prompt%20to%20an%20LLM.)
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=ChatOpenAI(openai_api_key=constants.APIKEY, model="gpt-4o", temperature=0),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=template,
                    input_variables=["summaries", "question"],
                ),
            },
        )

        chat_history = []
        with get_openai_callback() as cb:
            result = chain.invoke(input={"question": query, "chat_history": chat_history})
        self.costs_euro += cb.total_cost
        self.costs_tokens += cb.total_tokens

        parsed_result = self.extract_integer(result['answer'])
        print(parsed_result)
        self.n_values.append(parsed_result)

        result2 = chain.invoke(input={"question": 'Why could you not find the final amount of participants?', "chat_history": chat_history})
        print(result2['answer'])



    @staticmethod
    def extract_integer(text):
        match = re.search(r'\d+', text)
        if match:
            # Return the first integer found in the text
            return match.group()
        else:
            # If no integer found, return the whole string
            return text

    def make_excell(self):
        wb = Workbook()
        ws = wb.active
        for i, num in enumerate(self.n_values, start=1):
            ws.cell(row=i, column=1, value=num)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"output_nValues{timestamp}.xlsx"

        wb.save(file_name)
        os.system(f'start excel "{file_name}"')



runnable = SciPlow()
runnable.main()
# runnable.make_excell()




