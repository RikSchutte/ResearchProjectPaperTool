from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from resources import constants


llm = ChatOpenAI(openai_api_key=constants.APIKEY, model='gpt-3.5-turbo-1106')


query = input('Please enter your query: ')
# Modified template
template = 'Only answer using three words'
prompt_template = ChatPromptTemplate.from_messages([
    ('system', template),
    ('human', query)
])
chatPromptTemplate = prompt_template.format()

print('Tokens needed for this query: ' + str(llm.get_num_tokens(chatPromptTemplate)))
choice = input('Ask ChatGPT? (Y/N)')
if choice == 'Y':
    print('...')
    result = llm.invoke(chatPromptTemplate)
    print(result)