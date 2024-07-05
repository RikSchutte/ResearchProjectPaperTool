import os
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openpyxl import Workbook
from datetime import datetime

from resources import constants

class SciPlowTxt:
    llm = ChatOpenAI(openai_api_key=constants.APIKEY, model='gpt-4o')
    n_values = []
    tokens = 0

    def main(self):
        print('start!')
        count = 0

        folder_path = 'resources/participants_or_resultsmethods'
        files = os.listdir(folder_path)
        sorted_files = sorted(files, key=self.extract_number)

        for file_name in sorted_files:
            print(file_name)
            file_path = os.path.join(folder_path, file_name)
            self.run_model(open(file_path).read())


    @staticmethod
    def extract_number(filename):
        # Use regex to find the trailing number
        match = re.search(r'(\d+)(?=\D*$)', filename)
        if match:
            return int(match.group(0))
        else:
            print('error 2')
            return 0


    def run_model(self, file_text):
        query = 'From the research paper text find the amount of participants who finished all experiments, do not include drop out participants: "' + file_text + '"'

        template = 'Answer by only giving the number, if you can not find an answer write null.'
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', template),
            ('human', query)
        ])
        chatPromptTemplate = prompt_template.format()
        result = self.llm.invoke(chatPromptTemplate)
        self.tokens += self.llm.get_num_tokens(chatPromptTemplate)
        print(result.content)
        self.n_values.append(result.content)


    def make_excell(self):
        wb = Workbook()
        ws = wb.active
        for i, num in enumerate(self.n_values, start=1):
            ws.cell(row=i, column=1, value=num)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"output_nValues{timestamp}.xlsx"

        wb.save(file_name)
        os.system(f'start excel "{file_name}"')



runnable = SciPlowTxt()
runnable.main()
runnable.make_excell()
print('tokens used: '+ str(runnable.tokens))
