import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from modules.translator import parse_relation, translate
from dotenv import load_dotenv
load_dotenv()

def call_json_output_parser(phrase):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(model="gpt-4o", temperature=0.7,api_key=OPENAI_API_KEY)

    system_prompt = system_prompt = """
    AI is responsible for parsing a dialogue containing questions and answers and generating a list of questions based on the content. The questions should be engaging, interactive, and relevant to the conversation's context. The whole chat has to be conversational and engaging, with the AI acting as a teacher conducting an interactive training session. The AI should provide theoretical knowledge, definitions, and motivational questions to encourage student participation.

    The max amount of sequential ACTIONS is 3. After that a question engaging the user must be asked. That question must be sequential, making the flow interactive and not take different paths.

    There are three types of question_type: MULTIPLECHOICE, SINGLECHOICE, and ACTION. There are two types of `type`: QUESTION and ACTION.

    - Type ACTION is used to provide context, and Type QUESTION is used to display question options. 
    - ACTION must always be followed by a QUESTION or another ACTION, but ultimately, the sequence must end with a QUESTION.
    - ACTIONS cannot include options. 
    - The maximum amount of characters per ACTION is 150 characters.

    Remember, each question should consist of an ACTION with the question content and a QUESTION with the possible answers. For example: 
    - ACTION: "What is the capital of France?" 
    - QUESTION: "a) Paris b) London c) Berlin d) Madrid" (The question name should be an empty string.)

    This is mandatory to fulfill the requirements of the task above, becuase it is used after in a computer program and that the QUESTION name is always empty as the name of the question is passed in the ACTION before.

    If an ACTION's content is lengthy, it should be split into multiple ACTION entries. Make the flow interactive, avoiding long sequences of ACTIONs without questions.

    Ensure each question is directly related to the previous content to maintain logical, sequential flow. Include at least one MULTIPLECHOICE question to enhance engagement and assess comprehension. For example, when the dialogue says "Are you ready?!", it should be followed by a SINGLECHOICE question with only positive options like "Yes!" or "Absolutely!"

    If there are many lessons in the text, consider everything as one lesson.
    
    It is important to take the content of the previous lesson literally, we just want to format it.

    Include the emojis.

    At the end of the conversation, the AI should provide a summary of key points and encourage further exploration of the topic, ending with a goodbye message.

    Interactive questions must not open new paths but instead focus on continuing the current explanation or segment. For example, if the limit is reached, provide options like "Continue with the current topic" or "Learn about the next concept" to ensure a smooth flow without diverging into alternative topics.

    Formatting Instructions: {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{phrase}")
    ])

    class Question(BaseModel):
        id: str = Field(description="the id of the question (numeric incremental)")
        type: str = Field(description="the type of the question, it can be 'QUESTION' or 'ACTION'")
        name: str = Field(description="the content of the question")
        question_type: str = Field(description="the type of the question, it can be 'SINGLECHOICE','MULTIPLECHOICE' or 'ACTION'")
        options: list[str] = Field(description="the options of the question, it must be a list of strings. If the question is not a multiple choice question, it must be an empty list")
        answer: list[str] = Field(description="List of the options that are correct")

    class Relation(BaseModel):
        parent: str = Field(description="the id of the parent question")
        child: str = Field(description="the id of the child question")

    class Response(BaseModel):
        id: str = Field(description="the id of the response, should be an emty string")
        type: str = Field(description="the type of the response, should be DYNAMIC")
        initial: str = Field(description="the id of the initial question/action")
        elements: list[Question] = Field(description="the generated questions (must be question type as passed)")
        relations: list[Relation] = Field(description="the relations between the questions, (order of the questions)")
    
    parser = JsonOutputParser(pydantic_object=Response)

    chain = prompt | model | parser
    
    return chain.invoke({
        "phrase":phrase,
        "format_instructions": parser.get_format_instructions()
    })

def parse_text(text):
    res = call_json_output_parser(text)
    mapped_questions = list(map(translate,res["elements"]))
    parsed_relations = list(map(parse_relation,res["relations"]))
    output = {
        "elements": mapped_questions,
        "relations": parsed_relations
    }
    return output