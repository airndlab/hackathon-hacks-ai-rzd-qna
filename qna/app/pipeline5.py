import os

from haystack.utils import ComponentDevice

device = ComponentDevice.from_str("cuda:0")

from haystack import Document
from haystack.components.converters import PyPDFToDocument, DOCXToDocument
from haystack.components.converters.csv import CSVToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from pathlib import Path

DATASET_DIR_PATH = os.getenv('DATASET_DIR_PATH')


def index_store(main_docs_dir: str = DATASET_DIR_PATH):
    global document_store
    orgs_dirs = Path(main_docs_dir)

    file_type_router = FileTypeRouter(mime_types=[
        "application/pdf",
        "text/csv",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"])

    pdf_converter = PyPDFToDocument()
    docx_converter = DOCXToDocument()
    csv_converter = CSVToDocument()

    document_joiner = DocumentJoiner()
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="word", split_length=800, split_overlap=100)

    document_store = InMemoryDocumentStore()
    document_writer = DocumentWriter(document_store)
    document_embedder = SentenceTransformersDocumentEmbedder(
        model="intfloat/multilingual-e5-large", device=device)

    preprocessing_pipeline = Pipeline()

    preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
    preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
    preprocessing_pipeline.add_component(instance=docx_converter, name="docx_converter")
    preprocessing_pipeline.add_component(instance=csv_converter, name="csv_converter")
    preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")
    preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
    preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
    preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

    preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
    preprocessing_pipeline.connect(
        "file_type_router.application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "docx_converter.sources")
    preprocessing_pipeline.connect("file_type_router.text/csv", "csv_converter.sources")

    preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
    preprocessing_pipeline.connect("docx_converter", "document_joiner")
    preprocessing_pipeline.connect("csv_converter", "document_joiner")

    preprocessing_pipeline.connect("document_joiner", "document_cleaner")
    preprocessing_pipeline.connect("document_cleaner", "document_splitter")
    preprocessing_pipeline.connect("document_splitter", "document_embedder")
    preprocessing_pipeline.connect("document_embedder", "document_writer")

    for org_dir in orgs_dirs.iterdir():
        if org_dir.is_dir():
            source_docs = list(org_dir.glob("**/*"))

            preprocessing_pipeline.run({
                "file_type_router": {"sources": source_docs},
                "pypdf_converter": {"meta": {"organization": org_dir.name}},
                "csv_converter": {"meta": {"organization": org_dir.name}},
                "docx_converter": {"meta": {"organization": org_dir.name}},
            })


index_store()

from haystack import component
from typing import List

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder

MODEL_NAME = os.getenv('MODEL_NAME', 'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4')
MODEL_URL = os.getenv('MODEL_URL', 'http://vllm:8000/v1')


@component
class QueryExpander:
    def __init__(self):
        self.system_prompt = """

          """

        self.user_prompt_template = """

            """

        builder = ChatPromptBuilder(variables=["query", "user_info"])
        llm = OpenAIChatGenerator(
            api_key=Secret.from_token("VLLM-PLACEHOLDER-API-KEY"),
            model=MODEL_NAME,
            api_base_url=MODEL_URL,
            generation_kwargs={
                "max_tokens": 2048,
                "temperature": 0.3
            },
            timeout=600
        )
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=builder)
        self.pipeline.add_component(name="llm", instance=llm)
        self.pipeline.connect("builder", "llm")

    @component.output_types(queries=List[str])
    def run(self, query: str, user_info: str):
        messages = [
            ChatMessage.from_system(self.system_prompt),
            ChatMessage.from_user(self.user_prompt_template)
        ]

        result = self.pipeline.run({
            'builder': {
                'template': messages,
                'query': query,
                'user_info': user_info
            }
        })

        response_text = result['llm']['replies'][0].content

        def extract_json_array(text):
            last_bracket_idx = text.rfind(']')
            if last_bracket_idx == -1:
                return None, text
            first_bracket_idx = text.rfind('[', 0, last_bracket_idx)
            if first_bracket_idx == -1:
                return None, text
            json_str = text[first_bracket_idx:last_bracket_idx + 1]
            remaining_text = text[:first_bracket_idx].strip()
            return json_str, remaining_text

        json_str, remaining_text = extract_json_array(response_text)

        expanded_queries = []

        if json_str:
            try:
                expanded_queries = json.loads(json_str)
            except Exception:
                return {"queries": query}

        return {"queries": expanded_queries}


from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder


@component
class MultiQueryTextEmbedder:
    def __init__(self, embedder: SentenceTransformersTextEmbedder, top_k: int = 1):
        self.embedder = embedder
        self.embedder.warm_up()
        self.results = []
        self.ids = set()
        self.top_k = top_k

    @component.output_types(embeddings=List[List[str]])
    def run(self, queries: List[str]):
        self.results = []
        for query in queries:
            self.results.append(self.embedder.run(query))
        return {"embeddings": self.results}


@component
class MultiQueryInMemoryRetriever:
    def __init__(self, retriever: InMemoryEmbeddingRetriever, filters=None, top_k: int = 1):

        self.retriever = retriever
        self.results = []
        self.ids = set()
        self.top_k = top_k
        self.filters = filters

    def add_document(self, document: Document):
        if document.id not in self.ids:
            self.results.append(document)
            self.ids.add(document.id)

    @component.output_types(documents=List[Document])
    def run(self, emdeddings: List[List[str]], filters=None):
        for emdedding in emdeddings:
            result = self.retriever.run(query_embedding=emdedding['embedding'], filters=filters, top_k=self.top_k)
            for doc in result['documents']:
                self.add_document(doc)
        self.results.sort(key=lambda x: x.score, reverse=True)
        return {"documents": self.results}


from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.utils import Secret

expander = QueryExpander()
text_embedder = MultiQueryTextEmbedder(
    SentenceTransformersTextEmbedder(model="intfloat/multilingual-e5-large", device=device))
retriever = MultiQueryInMemoryRetriever(InMemoryEmbeddingRetriever(document_store))
generator = OpenAIChatGenerator(
    api_key=Secret.from_token("VLLM-PLACEHOLDER-API-KEY"),
    model=MODEL_NAME,
    api_base_url=MODEL_URL,
    generation_kwargs={
        "max_tokens": 4096,
        "temperature": 0.3},
    timeout=600)

chat_system_prompt = """
You are a high-class support chatbot for "РЖД" (RZD), a Russian railway company.

Your task is to provide accurate answers related to RZD, based on the provided context.

**Rules to follow**:
- Address the user respectfully, using "вы", and by their name if provided.
- If the user asks about themselves, consider the additional user information provided.
- Say **exactly** "Я не знаю ответа на ваш вопрос" if:
   1. The input is not a question.
   2. The answer is not in the provided context.
   3. The question is unrelated to RZD.
- Never generate information outside the provided context.
- Answer in detail, making maximum use of the information provided in context if it is relevant to the question.
- Where applicable, for ease of reading, format the answer using line breaks and bulleted lists.
- Limit your answer to 10-12 sentences

**Additional Instructions**:
- After providing the answer, include a JSON object that lists the document names and paragraph numbers (if specified at the beginning of the content) that were used to generate the answer.
- **Do not include** the JSON object if your answer is exactly "Я не знаю ответа на ваш вопрос."
- The format should be:
  `[{"document": <Document Name>, "paragraph": <Paragraph Number>}, {"document": <Document Name>, "paragraph": <Paragraph Number>}]`
- `<Document Name>` is exactly what is taken from `document.meta['file_path']`).
- If a paragraph number is not specified at the beginning of the document content, set `"paragraph"` to `"0"`.
- Only include documents in the list if you used them to generate the answer.
- Ensure that the JSON is properly formatted.
- Here are some examples of how to format the JSON:
  - `[{"document": "Коллективный договор", "paragraph": "1.1"}, {"document": "Правила техники безопасности", "paragraph": "2.3"}]`
  - `[{"document": "Регламент компании", "paragraph": "0"}]`
  - `[{"document": "Руководство по управлению персоналом", "paragraph": "4.5"}]`
- Do not invent or alter document names; use only the names provided in the context.

A lot depends on this answer—triple-check it!
"""

chat_user_template = """
{% if user_name %}
User Name: {{ user_name }}
{% endif %}
{% if user_info %}
Additional User Info:
{{ user_info }}
{% endif %}

<context>
{% for document in documents %}
Document: {{ document.meta['file_path']}}
Content:
{{ document.content }}
{% endfor %}
</context>

Question: {{ question }}
Answer:
"""
chat_system_message = ChatMessage.from_system(chat_system_prompt)
chat_user_message = ChatMessage.from_user(chat_user_template)
chat_messages = [chat_system_message, chat_user_message]
chat_prompt_builder = ChatPromptBuilder(variables=["documents", "question", "user_name", "user_info"])

changes_system_prompt = """
You are a high-class support chatbot for "РЖД" (RZD), a Russian railway company.

Your task is to inform the user about opportunities, benefits, incentives, material assistance, and compensations that are available to them, based on the provided context and considering any changes in their personal information.

**Rules to follow**:
- Address the user respectfully, using "вы", and by their name if provided.
- Consider the additional user information provided, especially changes in their personal data.
- Provide detailed information about new opportunities or benefits that are now available to the user due to the changes in their information.
- Do not include any information that is not in the provided context.
- Never generate information outside the provided context.
- Where applicable, for ease of reading, format the answer using line breaks and bulleted lists.
- Limit your answer to 10-12 sentences

**Additional Instructions**:
- After providing the information, include a JSON object that lists the document names and paragraph numbers (if specified at the beginning of the content) that were used to generate the information.
- **Do not include** the JSON object if there are no relevant opportunities or benefits to inform the user about.
- The format should be:
  `[{"document": <Document Name>, "paragraph": <Paragraph Number>}, {"document": <Document Name>, "paragraph": <Paragraph Number>}]`
- `<Document Name>` is exactly what is taken from `document.meta['file_path']`.
- If a paragraph number is not specified at the beginning of the document content, set `"paragraph"` to `"0"`.
- Only include documents in the list if you used them to generate the information.
- Ensure that the JSON is properly formatted.
- Here are some examples of how to format the JSON:
  - `[{"document": "Коллективный договор", "paragraph": "1.1"}, {"document": "Правила внутреннего распорядка", "paragraph": "2.3"}]`
  - `[{"document": "Положение о материальной помощи", "paragraph": "0"}]`
  - `[{"document": "Программа поддержки сотрудников", "paragraph": "4.5"}]`
- Do not invent or alter document names; use only the names provided in the context.

A lot depends on this answer—triple-check it!
"""

changes_user_template = """
{% if user_name %}
User Name: {{ user_name }}
{% endif %}
{% if user_info %}
Additional User Info:
{{ user_info }}
{% endif %}

Changes in User Information:
{% for key, value in changes.items() %}
- {{ key }}: {{ value }}
{% endfor %}

<context>
{% for document in documents %}
Document: {{ document.meta['file_path'] }}
Content:
{{ document.content }}
{% endfor %}
</context>

Inform the user about the new opportunities or benefits available to them based on their changes.
"""

changes_system_message = ChatMessage.from_system(changes_system_prompt)
changes_user_message = ChatMessage.from_user(changes_user_template)
changes_messages = [changes_system_message, changes_user_message]

chat_prompt_builder = ChatPromptBuilder(variables=["documents", "question", "user_name", "user_info", "changes"])

from haystack import Pipeline

basic_rag_pipeline = Pipeline()

basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", chat_prompt_builder)
basic_rag_pipeline.add_component("llm", generator)
basic_rag_pipeline.connect("text_embedder.embeddings", "retriever")
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")

from pydantic import BaseModel
from typing import List, Optional


class Reference(BaseModel):
    document: str
    paragraph: str


class ModelResponse(BaseModel):
    answer: str
    references: Optional[List[Reference]] = None


def get_chat_response(question: str, user_name: str = "", user_info: str = "", user_org: str = "") -> ModelResponse:
    expanded_questions = expander.run(
        query=question,
        user_info=user_info
    )['queries']

    response = basic_rag_pipeline.run({
        "text_embedder": {"queries": expanded_questions},
        "prompt_builder": {"question": question,
                           "template": chat_messages,
                           "user_name": user_name,
                           "user_info": user_info},
        "retriever": {
            "filters": {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.organization", "operator": "==", "value": "RZD"},
                    {"field": "meta.organization", "operator": "==", "value": user_org},
                ]
            }}
    },
        include_outputs_from={"prompt_builder"})

    response_text = response["llm"]["replies"][0].content

    references = None
    try:
        # Находим индекс последнего символа ']'
        last_bracket_idx = response_text.rfind(']')
        if last_bracket_idx != -1:
            # Находим индекс соответствующего символа '[' перед ']'
            first_bracket_idx = response_text.rfind('[', 0, last_bracket_idx)
            if first_bracket_idx != -1:
                json_str = response_text[first_bracket_idx:last_bracket_idx + 1]

                json_str_corrected = re.sub(
                    r'"paragraph"\s*:\s*(\d+\.\d+\.\d+)',
                    r'"paragraph": "\1"',
                    json_str
                )

                try:
                    references_list = json.loads(json_str_corrected)
                    references = [Reference(**item) for item in references_list]
                    response_text = response_text[:first_bracket_idx].strip()
                except json.JSONDecodeError as e:
                    print(f"Ошибка при парсинге JSON: {e}")
    except Exception as e:
        print(f"Общая ошибка при обработке ответа: {e}")

    gc.collect()
    torch.cuda.empty_cache()

    return ModelResponse(answer=response_text, references=references)


import gc
import torch
import re
import json

# Предполагаем, что этот словарь содержит соответствия между английскими и русскими терминами
ENG_TO_RUS = {
    'work_experience': 'стаж работы',
    'number_of_children': 'количество детей',
    "region": "регион",
    'age': 'возраст',
    'position': 'должность',
}


def get_benefits_response(
        changes_dict: dict,
        user_name: str = '',
        user_info: str = '',
        user_org: str = ''
) -> ModelResponse:
    russian_changes = {}
    for key, value in changes_dict.items():
        russian_key = ENG_TO_RUS.get(key, key)  # Если нет соответствия, оставляем оригинальный ключ
        russian_changes[russian_key] = value

    # Формируем первый вопрос для ретривера
    retriever_question = 'Льготы, поощрения, материальная помощь, компенсации которые зависят от '
    retriever_question += ' и '.join(russian_changes.keys())
    print(retriever_question)

    # Формируем второй вопрос для prompt_builder
    prompt_builder_question = 'Какие льготы, поощрения, материальная помощь, компенсации полагаются сотрудникам со '
    prompt_builder_question += ', '.join([f"{key}: {value}" for key, value in russian_changes.items()])
    print(prompt_builder_question)

    # Запускаем RAG пайплайн
    response = basic_rag_pipeline.run({
        "text_embedder": {"queries": [retriever_question]},
        "prompt_builder": {
            "question": prompt_builder_question,
            "template": changes_messages,
            "user_name": user_name,
            "user_info": user_info,
            "changes": russian_changes
        },
        "retriever": {
            "filters": {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.organization", "operator": "==", "value": "RZD"},
                    {"field": "meta.organization", "operator": "==", "value": user_org},
                ]
            }
        }},
        include_outputs_from={"prompt_builder"})

    response_text = response["llm"]["replies"][0].content

    # Извлечение JSON-объекта по символам [ и ] с конца строки
    references = None
    try:
        # Находим индекс последнего символа ']'
        last_bracket_idx = response_text.rfind(']')
        if last_bracket_idx != -1:
            # Находим индекс соответствующего символа '[' перед ']'
            first_bracket_idx = response_text.rfind('[', 0, last_bracket_idx)
            if first_bracket_idx != -1:
                json_str = response_text[first_bracket_idx:last_bracket_idx + 1]

                # Предобработка JSON-строки для корректного парсинга
                json_str_corrected = re.sub(
                    r'"paragraph"\s*:\s*([\d\.]+)',
                    r'"paragraph": "\1"',
                    json_str
                )

                try:
                    references_list = json.loads(json_str_corrected)
                    references = [Reference(**item) for item in references_list]
                    # Удаляем JSON-объект из ответа
                    response_text = response_text[:first_bracket_idx].strip()
                except json.JSONDecodeError as e:
                    print(f"Ошибка при парсинге JSON: {e}")
    except Exception as e:
        print(f"Общая ошибка при обработке ответа: {e}")

    gc.collect()
    torch.cuda.empty_cache()

    return ModelResponse(answer=response_text, references=references)
