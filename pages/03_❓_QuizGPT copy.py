import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser

question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context, make 2 questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }}
                ]
            }}
        ]
     }}
    ```
    Your turn!
    Questions: {context}
""",
        )
    ]
)


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        import json

        text = (
            text.replace("```", "")
            .replace("json", "")
            .replace(", ]", "]")
            .replace(", }", "}")
        )
        return json.loads(text)


@st.cache_data(show_spinner="File is uploaded...")
def handle_file(file):
    # file 저장하기
    upload_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(upload_content)
    # file load, split하기
    loader = UnstructuredFileLoader(file_path)

    load_docs = loader.load()

    return load_docs


@st.cache_data(show_spinner="Generating Quiz...")
def generate_quiz(docs):
    question_chain = question_prompt | llm
    formatting_chain = formatting_prompt | llm

    final_chain = {"context": question_chain} | formatting_chain | JsonOutputParser()

    response_json = final_chain.invoke({"context": docs})

    return response_json


@st.cache_data(show_spinner="Searching Wikipedia...")
def search_wiki(topic):
    retriever = WikipediaRetriever(top_k_results=5)
    wikipedia_docs = retriever.get_relevant_documents(topic)
    st.session_state["click"] = False
    return "\n\n".join(page.page_content for page in wikipedia_docs)


llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

if "click" not in st.session_state:
    st.session_state["click"] = False

st.set_page_config(page_title="QuizGPT", page_icon="❓")

st.title("QuizGPT")

st.markdown(
    """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
)

with st.sidebar:
    choice = st.selectbox("choose whay you want to use", ["Your own file", "Wikipedia"])
    docs = None
    if choice == "Your own file":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file", type=["pdf", "txt", "docx"]
        )
        if file:
            load_docs = handle_file(file)
            if st.session_state.get("file") != file.name:
                st.session_state["file"] = file.name
                st.session_state["click"] = False

            docs = load_docs[0].page_content

    else:
        topic = st.text_input("Name of the Article")
        if topic:
            docs = search_wiki(topic)
            if st.session_state.get("topic") != topic:
                st.session_state["topic"] = topic
                st.session_state["click"] = False

if docs:
    st.divider()
    placeholder = st.empty()
    if st.session_state["click"] == False:
        button = placeholder.button("Start Quiz")
        if button:
            st.session_state["click"] = True
            placeholder.empty()

    if st.session_state["click"]:
        json = generate_quiz(docs)

        with st.form("quiz"):
            for question in json["questions"]:
                value = st.radio(
                    label=question["question"],
                    options=[answer["answer"] for answer in question["answers"]],
                    index=None,
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                elif value is not None:
                    st.error("Wrong!")
            st.form_submit_button("Submit")
