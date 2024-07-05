from resources import constants

from langchain_openai import ChatOpenAI

query = input('Please enter your query: ')
llm = ChatOpenAI(openai_api_key=constants.APIKEY, model="gpt-3.5-turbo-1106")
choice = input('Ask ChatGPT? (Y/N)')
if choice == 'Y':
    print('...')
    print(llm.invoke(query))