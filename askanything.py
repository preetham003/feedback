# Feedback libraries
from flask import Flask, jsonify, Response, request
from flask_cors import CORS
import warnings
from dotenv import load_dotenv 
import pandas as pd
from langchain.chains import SequentialChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import SimpleMemory
from langchain.chains import RetrievalQA
import os
import weaviate
import cohere
import json
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import OpenAI 
from langchain.chains import ConversationChain
from categories import original_categories
import os
import ast

from bs4 import BeautifulSoup
import tiktoken  
from datetime import datetime
import csv 
# Get the directory of your current script
project_dir = os.path.dirname(__file__)

memory = ConversationBufferWindowMemory()

load_dotenv()

# Flask framework initialize app variable
app = Flask(__name__)
CORS(app)

# Initialize the keys#
os.environ["OPENAI_API_KEY"] = "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"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Authorize with Weaviate API
auth_config = weaviate.AuthApiKey(api_key='TodhpKV9Rdw19kqppOqvLtWxyp6V5xG1pkdd')

os.environ['cohere_api_key'] = "zkBT3nLbycu1KWM4XZO2JBAIOpKpDs6wY1bdNbqf"
#cohere api key
cohere_api_key = os.getenv('cohere_api_key')


os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "YOUR API KEY"
os.environ["LANGCHAIN_HUB_API_URL"] = "https://api.hub.langchain.com"
os.environ["LANGCHAIN_HUB_API_KEY"] = "YOUR API KEY"

# Connecting to the Cohere API
co = cohere.Client(cohere_api_key)
def weaviate_client():
    
    # Connect to Weaviate client
    client = weaviate.Client(
        url="https://feedback-reviews-vc4hcam8.weaviate.network",
        auth_client_secret=auth_config,
        additional_headers={
            "X-Cohere-Api-Key": cohere_api_key,
        }
    )
    return client


def convert_query_to_vector(client,query):
    query_vector = OpenAIEmbeddings().embed_query(query)
    search_result = client.query.get("ReviewEmbeddings", ["review"]).with_near_vector({"vector": query_vector,"certainty": 0.7}).with_limit(5).do()
    reviews_result = search_result['data']['Get']['ReviewEmbeddings']
    return reviews_result
 # return top 5 results


def rerank_responses(query, responses, num_responses=5):
    reranked_responses = co.rerank(
        model='rerank-english-v2.0',
        query=query,
        documents=responses,
        top_n=num_responses,
    )
    rerank_result = [{"review":rerank.document['text']} for rerank in reranked_responses]
    return rerank_result

# reviews are inserted in conversational buffer memory
def insert_memory(original_question,ai_answer):
    #input queries are stored in conversation buffer memory
    memory.save_context({"input": original_question},{"outputs":ai_answer})


def question_summarize(query):
    prompt_template = """
        
        #Question
        {intial_query}

        #INSTRUCTION
        - You are an AI assistant equipped with extensive hotel reviews.
        - you will be given two thing, a question and user conversatation history. 
        - user question will be either about hotel reviews or may be about conversatation history.
        - you will come across three scenarios, 1) user question is about hotel reviews, 2) user question is about conversatation history, 3) user question is not releted of any of these. 
        - if the user question is about hotel reviews, you will say the question is valid and you will rephrase the question.
        - if the user question is about conversatation history, you need to descide if you can answer the question based on the conversatation history or not. 
            and if you can answer the question based on the conversatation history, you should say the question is valid and also give the answer
        - if the question is not realted to any of these, you should say the question is invalid and asnwer is Sorry I can't answer the question. 


        #About Output
        - the Output should be a proper valid json object only with 3 keys the question, validatyStatus, updateQuestion
        - "question" contains the original question.
        - "validityStatus" is a boolean flag indicating if the question is valid.
        - "updateQuestion" is a string representing the reframed question.
        - "answer" is you answer for the question if you dont have an answer return false.

        #Example

        "question": "what does people say about my rooms",
        "validityStatus": true,
        "updateQuestion": "what does people say about my rooms?",
        "answer": false
         Though Process : The question looks valid and is seems it is realted to hotel, the answere key is false because it can not be answere without a valid hotel reviews
        
        "question": "tell me more about it",
        "validityStatus": true,
        "updateQuestion": "what does people say about my rooms, get more details? ", # also add the context of the previous answer from the history to understand what was responed to the user earlier and what more can be given.
        "answer": false
        Though Process : This quesiton is a follow up question and we need to refame the question with the previous question and the answer.

        "question": "How is the weather today?",
        "validityStatus": false,
        "updateQuestion": "How is the weather today?"
        "answer": false
        Though Process : the Question is not realted to hotel

        "question": "what was my previous question?",
        "validityStatus": true,
        "updateQuestion": "what was my previous question?"
        "answer": Your previous question was 'How is the weather today?'
         Though Process : the user is asking about the previous question. hence simply pic the previous quesiton from history and add it to answer section

        #History 
        {memory}

    """.format(intial_query=query,memory=memory)
    question_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # using convesation for final question 
    conversation = ConversationChain(llm=question_llm, verbose=True, memory=ConversationBufferWindowMemory())
    question_result_format = conversation.predict(input=prompt_template)
    question_result = question_result_format.replace("json", "").replace("```", "")
    # Convert the JSON string to a Python object (dictionary in this case)
    question_result_json = json.loads(question_result)
    return question_result_json



def get_ai_agent_response(reviews):
    """
    input params: user query and weaviate database hotel reviews
    output response: user query and hotel reviews based on ConversationChain Agent give the answer
    """

    prompt_template = """
    # INTRODUCTION
    You are an AI assistant tasked with answering user queries based on hotel reviews.

    # Expected Input
    You will receive a user query and a list of hotel reviews as input.

    # Expected Output
    - Do not generate additional queries in response
    - Generate the Answer to user questions based on the reviews
    - Do not provide a summary of the reviews, directly address the user's inquiry without generating introduction.
    - Keep the response short and focused.

    User Query: {text}
    #RESPONSE
    """
    #Generate a precise answer to the user's query based on the reviews. for short response
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt, output_key="recommendations")

    chain_summary = SequentialChain(
        chains=[chain],
        input_variables=["text"],
        output_variables=["recommendations"],
        verbose=True,
    )

    results = chain_summary({"text": reviews})
    recommendations = results["recommendations"]

    return recommendations




def post_process_output(output):
    """
    Perform additional post-processing or filtering on the model's output.
    This function can be customized based on specific requirements.
    """
    # Remove unnecessary HTML tags or unwanted content
    cleaned_output = remove_html_tags(output)

    # Add any other post-processing steps as needed

    return cleaned_output

def remove_html_tags(text):
    """
    Remove HTML tags from the text.
    """
    

    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text(separator=" ", strip=True)
    return cleaned_text

log_file_path = os.path.join(project_dir, 'log_file.csv')

fieldnames = ['Timestamp', 'Query', 'Response', 'NumTokens']

def log_to_csv(query, response, num_tokens):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    response_without_html = response
    with open(log_file_path, 'a', newline='') as log_file:
        csv_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        csv_writer.writerow({'Timestamp': timestamp, 'Query': query, 'Response': response_without_html, 'NumTokens': num_tokens})

# ... (existing code)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


@app.route("/askanything", methods=['POST', 'GET'])
def askanything():
    try:
        query = json.loads(request.data)
        if "time_flag" in query:
            if query['time_flag'] == True:
                # memory is clear
                memory.clear()
        if "query" in query:
            client_connection = weaviate_client()

            user_query = question_summarize(query['query'])

            if user_query['validityStatus'] == True:
                if user_query['answer'] != False:
                    ai_result_format = user_query['answer']
                    num_tokens = num_tokens_from_string(ai_result_format,"cl100k_base")  # Count tokens in the response
                elif user_query['updateQuestion'] != "":
                    updated_question = user_query['updateQuestion']

                    get_resulst_weaviate = convert_query_to_vector(client_connection, updated_question)
                    reviews_result = prepare_weviate_output(get_resulst_weaviate)

                    reranks_result = rerank_responses(query['query'], reviews_result, num_responses=5)

                    ai_result = get_ai_agent_response(reranks_result)
                    ai_result_format = ai_result.replace("\n", "").replace("```", "").replace("#RESPONSE", "").replace(
                        "#Assistant", "").replace("User Query:", "")
                    num_tokens = num_tokens_from_string(ai_result_format,"cl100k_base") # Count tokens in the response

                ai_agent_response = {
                    "question": query['query'],
                    "answer": ai_result_format,
                    "message": "reviews details fetch successfully",
                    "status": True
                }

            elif user_query['validityStatus'] == False:
                ai_result_format = "Sorry, can't help you with this."
                num_tokens = num_tokens_from_string(ai_result_format,"cl100k_base")  # Count tokens in the response

                ai_agent_response = {
                    "question": query['query'],
                    "answer": ai_result_format,
                    "message": "something went wrong, please try again.",
                    "status": False
                }

            insert_memory(query['query'], ai_result_format)
            
            # Log the query, response, and number of tokens to CSV
            log_to_csv(query['query'], ai_result_format, num_tokens)

            return jsonify(ai_agent_response)

    except Exception as e:
        return handle_exception(e, query['query'])
    # except Exception as e:
    #     return handle_exception(e,query['query'])


def prepare_weviate_output(weviate_reviews):
    weaviate_output = []
    for elem in weviate_reviews:
        new_elem = {"text": elem['review']}
        weaviate_output.append(new_elem)
    return weaviate_output


def handle_exception(exception,query):
    print(str(exception))
    if "Error code: " in str(exception):
        error_message = str(exception).split("{'message':")[1].split("',")
        print(error_message[0])
        ai_agent_response = {
            
            "question": query,
            "answer":"server is busy, try after some time.",
            "message":error_message[0],
            "status":False
        }
    else:
        ai_agent_response = {
            "question": query,
            "answer":"internal server error",
            "message":"something went wrong, please try agin.",
            "status":False
        }
    return ai_agent_response


def ai_agent_category_review(reveiws):
    
    """
    input params : user query and weaviate database hotel reviews
    output response : user query and hotel reviews based on ConvesationChain Agent give the answer
    
    """
    prompt_template = """
    # INTRODUCTION
    You are an AI assistant tasked with answering user queries based on hotel reviews.

    # Expected Input
    You will receive a list of hotel reviews and categories as input.

    # Expected Output
    - Generate the answer based on input reviews, and the category should be returned in English.
    - The final answer is a summary and recommendations of the category based on reviews. It should be very specific to the input category and reviews.
    - Recommendations response should include only three points, and the summary should be in the form of a statement.
    - The response should be in JSON format and should not include any unnecessary data for easy reading and display, with keys like "summary" and "recommendation".

    User input
    {text}
    """

    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt, output_key="recommendations")

    chain_summary = SequentialChain(
        chains=[chain],
        input_variables=["text"],
        output_variables=["recommendations"],
        verbose=True,
    )

    results = chain_summary({"text": reveiws})
    recommendations = results["recommendations"]

    return recommendations

@app.route("/allcategories",methods=['GET'])
def feedback_categories():
    try:
        if len(original_categories)>0:
            category_list = []
            # Iterate through each list in all_categories
            for categories_nested in original_categories:
                # Add each item in the row to the set
                category_list.append(categories_nested['category'].replace("_"," "))
                    
            #[item for row in all_categories for item in row
            allcategory_response = {
                "answer": category_list,
                "message": "all categories found successfully",
                "status": True
            }
        else:
            allcategory_response = {
                "answer": original_categories,
                "message": "categories are not found.",
                "status": True
            }
    except Exception as e:
        allcategory_response = {
                "answer": "something went wrong, please try again",
                "message": "internal server",
                "status": True
            }
        
    return allcategory_response


def allcategories():
    # get the file from system 
    # Construct the path to the CSV file
    csv_filepath = os.path.join(project_dir, 'results', 'review_with_sentiments.csv')
    df = pd.read_csv(csv_filepath)
    # unamed columns removed 
    df = df.drop(columns=['Unnamed: 0'])
    df.head()
    
    categories = df[['date', 'categories',"sentiment_score","review"]]
    return categories



@app.route("/category_recomendations/",methods=['GET'])
def feedback_categorie_review():
    try:
        categories = allcategories() #get all categores from allcategories() function 
        categories_dict = categories.to_dict(orient="records")
        
        reviews_list = []
        input_category = request.args.get('input_category')
        
        find_list_categories = get_category_list(input_category)
        for inputcategory in find_list_categories:

            for category_name in categories_dict:
                #prepared all categories in list
                categories_list = [x for x in category_name['categories']] 
                # Convert the list to a string
                convert_category_string = ''.join(categories_list)

                # Replace single quotes with double quotes
                category_json_string = convert_category_string.replace("'", "\"")

                # Load the string as JSON
                category_json_data = json.loads(category_json_string)
                categories_list = list(category_json_data.keys())
                
                if len(categories_list)>0:
                    for category in categories_list:
                        # if any category name between space then will add underscore ex: swimming pool to swimming_pool
                        # if any category uppercase then will change lowercase 
                        category_format = category.replace(" ", "_") 
                        input_category_format = inputcategory.replace(" ", "_") 
                        
                        if category_format.lower() == input_category_format.lower():
                            #every category matching review append in list 
                            if "review" in category_name:
                                # Log an informational message
                                reviews_list.append({"review":category_name['review'],"category":input_category_format})

        if len(reviews_list)>0:
            ai_agent_category_response = ai_agent_category_review(reviews_list)
            # print(ai_agent_category_response)
            response_format = json.loads((ai_agent_category_response).replace("#OUTPUT", "").replace("# Output", ""))
            response_list = []
            response_dict = {}
            if len(response_format)>0:
                if "recommendation" in response_format:
                    for index,recommendation_data in enumerate(response_format['recommendation']):
                        response_list.append({"point":recommendation_data})
                    response_dict['recomendation'] = response_list
                if "summary" in response_format:
                    response_dict['summary'] = response_format['summary']
            
            ai_agent_response = {
                "answer": response_dict,
                "message": "category recommendations and summary details fetch successfully.",
                "status": True
            }
        else:
            ai_agent_response = {
                "answer": reviews_list,
                "message": "category recommendations and summary details are not found.",
                "status": True
            }
        return jsonify(ai_agent_response)
    except Exception as e:
        return handle_exception(e,input_category)



def get_category_list(input_category):
    # Construct the path to the CSV file
    csv_filepath = os.path.join(project_dir, 'categories_data.csv')
    df = pd.read_csv(csv_filepath)
    # unamed columns removed 
    csv_categories = df[['original_category','mapped_categories']].to_dict(orient="records")
    # print("csv_categories",csv_categories)
    mapped_categories_list = []
    for csv_categories_data in csv_categories:
        if "original_category" in csv_categories_data :
            if input_category.replace(" ","").lower() == csv_categories_data['original_category'].lower():
                if "mapped_categories" in csv_categories_data:
                    mapped_categories_data = csv_categories_data['mapped_categories']

                    # Convert the string to a list using ast.literal_eval
                    mapped_categories_list = ast.literal_eval(mapped_categories_data)

                    # Now mapped_categories_list is a Python list
                    # print(mapped_categories_list,type(mapped_categories_list))

    return mapped_categories_list


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80,debug=False)
    
    
