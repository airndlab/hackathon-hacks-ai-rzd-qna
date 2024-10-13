import os

from haystack.utils import ComponentDevice

device = ComponentDevice.from_str("cuda:0")

from haystack import Document, Pipeline
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
            source_docs = list(org_dir.glob("*/*"))

            preprocessing_pipeline.run({
                "file_type_router": {"sources": source_docs},
                "pypdf_converter": {"meta": {"organization": org_dir.name}},
                "csv_converter": {"meta": {"organization": org_dir.name}},
                "docx_converter": {"meta": {"organization": org_dir.name}},
            })


index_store()

MODEL_NAME = os.getenv('MODEL_NAME', 'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4')
MODEL_URL = os.getenv('MODEL_URL', 'http://vllm:8000/v1')

from haystack import component
from typing import List

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder


@component
class QueryExpander:
    def __init__(self):
        self.system_prompt = """
          You are part of an information system that processes user queries.

          Your task is to process a given user query and generate a list of queries that are similar in meaning and suitable for searching in the database.

          *Input:*

          - The user's query.
          - Additional information about the user (age, gender, number of children, veteran status, years of experience, etc.).

          *Instructions:*

          1. *If the user's query is about themselves* (contains words like "мне", "обо мне", "у меня", etc.):
            - Rephrase the query and generate a list of queries that include the user's additional information.
            - Incorporate relevant user details into the expanded queries to make them more specific.

          2. *If the user's query is not about themselves*:
            - *a.* If the query is noisy or inconvenient for searching, rephrase it to make it clearer and more suitable for search.
            - *b.* If the query is fully correct and suitable for searching, return it unchanged.

          3. *If the user's query contains multiple questions or topics*:
            - Decompose it into separate queries.

          4. *Do not make up any information* that is not present in the user's query or additional information.

          5. *Ensure that the generated queries match the meaning of the original query*.

          6. *The answer must be in the form of a list of strings in the format ["str1", "str2"]*.

          7. *Generate the answer exclusively in Russian*.

          ---

          *Structure:*

          Follow the structure shown below in the examples to generate the expanded queries.

          ### Пример 1:
          *Пользовательский запрос:*
          "Какие льготы мне положены?"

          *Дополнительная информация о пользователе:*
          - Стаж работы: 15 лет
          - Количество детей: 2

          *Расширенные запросы:*
          ["Льготы для сотрудников со стажем работы 15 лет", "Положенные льготы при наличии 2 детей", "Социальные программы для работников с 15-летним стажем и двумя детьми"]

          *Объяснение:* Пользователь спрашивает о себе, но не указывает конкретную информацию. Мы включаем дополнительную информацию о стаже и количестве детей в расширенные запросы.

          ### Пример 2:
          *Пользовательский запрос:*
          "Расскажите о программах повышения квалификации"

          *Дополнительная информация о пользователе:*
          - Должность: Машинист
          - Стаж работы: 5 лет

          *Расширенные запросы:*
          ["Программы повышения квалификации для машинистов", "Курсы для сотрудников со стажем работы 5 лет", "Обучение и развитие для машинистов в РЖД"]

          *Объяснение:* Пользователь не спрашивает непосредственно о себе, но мы можем уточнить запрос, используя его должность и стаж.

          ### Пример 3:
          *Пользовательский запрос:*
          "Какие документы нужны для получения материальной помощи и как их оформить?"

          *Дополнительная информация о пользователе:*
          - Нет

          *Расширенные запросы:*
          ["Перечень документов для получения материальной помощи", "Как оформить документы на материальную помощь"]

          *Объяснение:* Вопрос содержит несколько тем. Мы декомпозируем его на отдельные запросы без добавления дополнительной информации.

          ### Пример 4:
          *Пользовательский запрос:*
          "Правила охраны труда в компании"

          *Дополнительная информация о пользователе:*
          - Нет

          *Расширенные запросы:*
          ["Правила охраны труда в компании", "Нормативы безопасности на рабочем месте"]

          *Объяснение:* Вопрос ясен и не требует изменений или дополнительной информации. Мы возвращаем его без изменений и добавляем синонимичный запрос.

          ### Пример 5:
          *Пользовательский запрос:*
          "У меня появился статус ветерана труда, что мне теперь положено?"

          *Дополнительная информация о пользователе:*
          - Статус ветерана труда: Да
          - Регион: Север

          *Расширенные запросы:*
          ["Льготы для ветеранов труда в РЖД", "Положенные компенсации сотрудникам со статусом ветерана труда на Севере", "Льготы для работников на севере","Привилегии для работников-ветеранов труда"]

          *Объяснение:* Пользователь сообщает об изменении статуса. Мы используем эту информацию для формирования расширенных запросов.

          ---

          *A lot depends on this answer—triple-check it!*

          """

        self.user_prompt_template = """
            Пользовательский запрос: "{{ query }}"

            Дополнительная информация о пользователе:
            {% if user_info %}
            {{ user_info }}
            {% else %}
            - Нет
            {% endif %}

            Расширенные запросы:
            """

        builder = ChatPromptBuilder(variables=["query", "user_info"])
        llm = OpenAIChatGenerator(
            api_key=Secret.from_token("VLLM-PLACEHOLDER-API-KEY"),
            model=MODEL_NAME,
            api_base_url=MODEL_URL,
            generation_kwargs={
                "max_tokens": 2048,
                "temperature": 0.3},
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
    def run(self, emdeddings: List[List[str]], top_k=1, filters=None):
        for emdedding in emdeddings:
            result = self.retriever.run(query_embedding=emdedding['embedding'], filters=filters, top_k=top_k)
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

*Rules to follow*:
- Address the user respectfully, using "Вы", and by their name if provided.
- If the user asks about themselves, consider the additional user information provided.
- Say *exactly* "Я не знаю ответа на ваш вопрос" if:
   1. The input is not a question.
   2. The answer is not in the provided context.
   3. The question is unrelated to RZD.
- Never generate information outside the provided context.
- Answer in detail, making maximum use of the information provided in context if it is relevant to the question.
- Where applicable, for ease of reading, format the answer using line breaks and bulleted lists.
- Limit your answer to 10-12 sentences

*Additional Instructions*:
- After providing the answer, include a JSON object that lists the document names and paragraph numbers (if specified at the beginning of the content) that were used to generate the answer.
- *Do not include* the JSON object if your answer is exactly "Я не знаю ответа на ваш вопрос."
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

changes_system_prompt = """
You are a high-class support chatbot for "РЖД" (RZD), a Russian railway company.

Your task is to inform the user about opportunities, benefits, incentives, material assistance, and compensations that are available to them, based on the provided context and considering any changes in their personal information.

*Rules to follow*:
- Address the user respectfully, using "Вы", and by their name if provided.
- Consider the additional user information provided, especially changes in their personal data.
- Provide detailed information about new opportunities or benefits that are now available to the user due to the changes in their information.
- Do not include any information that is not in the provided context.
- Never generate information outside the provided context.
- Where applicable, for ease of reading, format the answer using line breaks and bulleted lists.
- Limit your answer to 10-12 sentences

*Additional Instructions*:
- After providing the information, include a JSON object that lists the document names and paragraph numbers (if specified at the beginning of the content) that were used to generate the information.
- *Do not include* the JSON object if there are no relevant opportunities or benefits to inform the user about.
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
    )['queries'][:5]
    print(expanded_questions)
    if len(expanded_questions) == 1:
        top_k = 5
    elif len(expanded_questions) == 2:
        top_k = 3
    elif len(expanded_questions) == 3:
        top_k = 2
    else:
        top_k = 1
    print(top_k)

    response = basic_rag_pipeline.run({
        "text_embedder": {"queries": expanded_questions},
        "prompt_builder": {"question": question,
                           "template": chat_messages,
                           "user_name": user_name,
                           "user_info": user_info},
        "retriever": {
            "top_k": top_k,
            "filters": {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.organization", "operator": "==", "value": "RZD"},
                    {"field": "meta.organization", "operator": "==", "value": user_org},
                ]
            },
        }
    },
        include_outputs_from={"retriever", "prompt_builder"})

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
                    references = [Reference(*item) for item in references_list]
                    response_text = response_text[:first_bracket_idx].strip()
                except json.JSONDecodeError as e:
                    print(f"Ошибка при парсинге JSON: {e}")
    except Exception as e:
        print(f"Общая ошибка при обработке ответа: {e}")

    gc.collect()
    torch.cuda.empty_cache()

    print(len(response['retriever']['documents']))

    return ModelResponse(answer=response_text, references=references)


import gc
import torch
import re
import json

# Расширенный словарь соответствий между английскими и русскими терминами
ENG_TO_RUS = {
    'title': 'заголовок',
    'description': 'описание',
    'person_name': 'имя',
    'position': 'должность',
    'organization': 'организация',
    'region': 'регион',
    'sex': 'пол',
    'age': 'возраст',
    'child_count': 'количество детей',
    'work_years': 'стаж работы',
    'veteran_of_labor': 'ветеран труда',
}

# Словарь для генитивного падежа (родительный падеж)
RUS_GENITIVE = {
    'заголовок': 'заголовка',
    'описание': 'описания',
    'имя': 'имени',
    'должность': 'должности',
    'организация': 'организации',
    'регион': 'региона',
    'пол': 'пола',
    'возраст': 'возраста',
    'количество детей': 'количества детей',
    'стаж работы': 'стажа работы',
    'ветеран труда': 'ветерана труда',
}

# Словарь для инструментального падежа (творительный падеж)
RUS_INSTRUMENTAL = {
    'заголовок': 'заголовком',
    'описание': 'описанием',
    'имя': 'именем',
    'должность': 'должностью',
    'организация': 'организацией',
    'регион': 'регионом',
    'пол': 'полом',
    'возраст': 'возрастом',
    'количество детей': 'количеством детей',
    'стаж работы': 'стажем работы',
    'ветеран труда': 'ветераном труда',
}


def get_benefits_response(
        changes_dict: dict,
        user_name: str = '',
        user_info: str = '',
        user_org: str = '',
        top_k=5
) -> ModelResponse:
    # Сопоставление английских ключей с базовыми русскими словами
    russian_changes_base = {}
    for key, value in changes_dict.items():
        russian_key_base = ENG_TO_RUS.get(key, key)
        russian_changes_base[russian_key_base] = value

    # Формирование словаря для retriever_question с генитивным падежом
    russian_changes_genitive = {}
    for key in russian_changes_base.keys():
        russian_key_genitive = RUS_GENITIVE.get(key, key)
        russian_changes_genitive[russian_key_genitive] = russian_changes_base[key]

    # Формирование словаря для prompt_builder_question с инструментальным падежом
    russian_changes_instrumental = {}
    for key in russian_changes_base.keys():
        russian_key_instrumental = RUS_INSTRUMENTAL.get(key, key)
        russian_changes_instrumental[russian_key_instrumental] = russian_changes_base[key]

    # Формируем первый вопрос для ретривера
    retriever_question = 'Льготы, поощрения, материальная помощь, компенсации которые зависят от '
    retriever_question += ' , '.join(russian_changes_genitive.keys())

    # Формируем второй вопрос для prompt_builder
    prompt_builder_question = 'Какие льготы, поощрения, материальная помощь, компенсации полагаются сотрудникам со '
    prompt_builder_question += ', '.join([f"{key}: {value}" for key, value in russian_changes_instrumental.items()])

    # Запускаем RAG пайплайн
    response = basic_rag_pipeline.run({
        "text_embedder": {"queries": [retriever_question]},
        "prompt_builder": {
            "question": prompt_builder_question,
            "template": changes_messages,
            "user_name": user_name,
            "user_info": user_info,
            "changes": russian_changes_instrumental
        },
        "retriever": {
            "filters": {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.organization", "operator": "==", "value": "RZD"},
                    {"field": "meta.organization", "operator": "==", "value": user_org},
                ]
            },
            "top_k": top_k
        }},
        include_outputs_from={"retriever", "prompt_builder"})

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
                    references = [Reference(*item) for item in references_list]
                    # Удаляем JSON-объект из ответа
                    response_text = response_text[:first_bracket_idx].strip()
                except json.JSONDecodeError as e:
                    print(f"Ошибка при парсинге JSON: {e}")
    except Exception as e:
        print(f"Общая ошибка при обработке ответа: {e}")

    gc.collect()
    torch.cuda.empty_cache()

    return ModelResponse(answer=response_text, references=references)


rzd_dict = """
А — министр путей сообщения
АБ — автоблокировка
АБ-Е2 — АБ единый ряд, второе поклоение
АБТ — АБ с тональными рельсовыми цепями
АБТЦ — АБ с тональными рельсовыми цепями и централизованным размещением аппаратуры
АЗ — заместитель А
АЗ-1 — первый заместитель А
АИС ЭДВ -автоматизированная информационная система организации перевозок с применением электронной дорожной ведомости
АКП — автоматические контрольные пункты
АЛС-ЕН — автоматическая локомотивная сигнализация единая непрерывная. В отличие от АЛСН, выдаёт информацию о состоянии 6 блок-участков до 200 км/ч и имеет ещё много новшеств. Работает на микропроцессорной технике
АЛСК — Автоматическая Локомотивная Сигнализация Комбинированного съёма информации
АЛСН — Автоматическая Локомотивная Сигнализация Непрерывного действия
АЛСО — автоматическая локомотивная сигнализация, применяемая как самостоятельное средство сигнализации и связи
АЛСТ — Автоматическая Локомотивная Сигнализация Точечного съема информации
АЛСЧ — автоматическая локомотивная сигнализация частотного типа (или просто частотная). Сбор информции до 5 блок-участков включительно при скоростях до 200 км/ч
АПК ДК — аппаратно-программный комплекс диспетчерского контроля
АРВ — автономные рефрижераторные вагоны
АРМ — автоматизированное рабочее место
АРМ ТВК — АРМ товарного кассира
АРС — Система Автоматического Регулирования Скорости
АСУ КП — автоматизированная система управления контейнерным пунктом
АСУ ГС — АСУ грузовой станцией
АСУ СС — АСУ сортировочной станцией
АТДП — автоматика и телемеханика (для) движения поездов
АТ — автоматика и телемеханика
АТС — автоматика, телемеханика, связь
АФТО — агентство фирменного транспортного обслуживания отделения дороги
АХУ — административно- хозяйственное управление МПС
Б — багажные вагоны
БМРЦ — блочная маршрутно-релейная централизация стрелок и сигналов
БП — почтово-багажные вагоны
БП — блок-пост
В — вагонная служба
ВНИИАС — Всероссийский научно-исследовательский и проектно-конструкторский институт информатизации, автоматизации и связи
ВНИИЖГ — Всероссийский научно-исследовательский институт железнодорожной гигиены
ВНИИЖТ — Всероссийский научно-исследовательский институт железнодорожного транспорта
ВНИИТИ — Всероссийский научно-исследовательский институт тепловозов и путевых машин
ВНИИУП — Всероссийский научно-исследовательский институт управления МПС; в прошлом ВНИИАС
ВНР — начальник рефрижераторной секции
ВОДЧ — дистанция водоснабжения и канализации
ВП — восстановительный поезд
ВПР — машина Выправочно-Подбивочно-Рихтовочная
ВПРС — машина Выправочно-Подбивочно-Рихтовочная Стрелочная
ВР — вагоны-рестораны
ВРЗ — вагоноремонтный завод
ВЦ — вычислительный центр управления дороги
ВЧ — вагонный участок, начальник вагонного участка
ВЧД — вагонное депо, начальник вагонного депо
ВЧДР — зам.начальника вагонного депо по ремонту
ВЧДЭ — зам.начальника вагонного депо по эксплуатации
ВЧГ — главный инженер вагонного участка
ВЧГЭ — главный энергетик вагонного участка
ВЧЗ — зам. начальника вагонного участка
ВЧЗр — зам. начальника вагонного участка по резерву проводников, начальник резерва проводников
ВЧИ — инструктор производственного обучения
ВЧК — начальник отдела кадров вагонного участка
ВЧОС — осмотрщик вагонов
ВЧОР — осмотрщик-ремонтник вагонов
ВЧРз — зам. начальника резерва, старший нарядчик
ВЧЮ — юрист вагонного участка
ВЭ — вагоны-электростанции
ГАЛС — горочная автоматическая локомотивная сигнализация
ГАЦ — горочная автоматическая централизация
ГВЦ — главный вычислительный центр МПС РФ
ГИД — График исполненного движения
ГТСС — государственный институт по проектированию сигнализации, централизации, с вязи и радио на железнодорожном транспорте
ГПЗУ — горочное программно-задающее устройство
ГУП ДКРМ — дирекция по комплексной реконструкции, капитальному ремонту и строительству объектов МПС
ГУ КФЦ — кредитно-финансовый центр МПС
ГУ ЦВКО — центр по взаимодействию с компаниями — операторами МПС
ГУП ЦСС — центральная станция связи МПС
Д — служба перевозок (движения?)
ДВ — отдел специальных и негабаритных перевозок службы перевозок
ДГ — начальник оперативно-распорядительного отдела службы Д
ДГКУ — дрезина с гидропередачей калужская усиленная
ДГПН — дежурный по направлению
ДГП — доpожный диспетчеp
ДГС — старший дорожный диспетчер
ДДЭ — Дежурный Диспетчер Эксплуатации (в метро)
ДИСПАРК — автоматизированная система пономерного учета и определения дислокации вагонного парка МПС
ДИСК — Дистанционная Информирующая Система Контроля (потомок ПОНАБа)
ДИСКОН — то же, что и ДИСПАРК, только для контейнеров
ДИСТПС — аналог ДИСПАРК для Тягового и Подвижного Составов
ДИСССПС — аналог ДИСПАРК для Специального Самоходного Подвижного Составов
ДК — диспетчерский контроль
ДЛ — пассажирская станция
ДНБ — Начальник кондукторского резерва
ДНЦ — поездной yчастковый (yзловой) диспетчеp
ДНЦВ — вагонно-pаспоpядительный диспетчеp
ДНЦО — дежypный по отделению
ДНЦС — старший диспетчеp
ДНЦТ — локомотивный диспетчеp; по другим сведениям, не прменяется
ДНЧ — ревизор движения отдела перевозок
ДОП — Дирекция обслуживания пассажиров
ДПКС — дежурный пункт дистанции контактной сети
ДР — деповской ремонт; старший ревизор службы перевозок
ДРС — дорожно-распорядительная связь
ДС — начальник станции
ДСГ — главный инженер станции
ДСД — главный кондуктор (составитель поездов)
ДСЗ — зам. начальника станции
ДСЗМ — зам. ДС по грузовой работе
ДСЗО — зам. ДС по оперативной работе
ДСЗТ — зам. ДС по технической работе
ДСИ — инженер железнодорожной станции
ДСМ — заместитель начальника станции по грузовой работе
ДСП — дежурный по станции; динамический стабилизатор пути
ДСПГ — дежурный по горке
ДСПГО — оператор при дежурном по сортировочной горке
ДСПП — дежурный по парку
ДСПФ — дежурный по парку формирования
ДСТК — начальник контейнерного отделения станции
ДСТКП — заведующий контейнерной площадкой
ДСЦ — маневровый диспетчер
ДСЦМ — станционный грузовой диспетчер
ДСЦП — дежурный поста централизации в метро
ДЦ — диспетчерская централизация стрелок и сигналов
ДЦ-МПК — диспетчерская централизация на базе микроЭВМ и программируемых контроллеров
ДЦФТО — дорожный центр фирменного транспортного обслуживания
ДЦХ — поездной диспетчер в метро
ЕДЦУ — единый диспетчерский центр управления
ЕК ИОДВ — единый комплекс интегрированной обработки дорожной ведомости
ЕМЦСС — единая магистральная цифровая сеть связи
ЕСР — единая сетевая разметка станций
ЖОКС — многожильное кабельное соединение между вагонами электропоезда
ЗКУ — комендатура военных сообщений
ЗТК — начальник товарной конторы
ЗУБ — землеуборочная машина Балашенко
ИВЦ — информационно-вычислительный центр (один на каждой железной дороге)
ИДП — Инструкция по движению поездов и маневровой работе на железных дорогах
ИСИ — Инструкция по сигнализации на железных дорогах
К — купейные вагоны; начальник контейнерной службы дороги
КАС ДУ — комплексная автоматизированная система диспетчерского управления
КБ — купейные вагоны с буфетами
КВР — капитально-восстановительный ремонт
КГМ — комплекс горочный микропроцессорный
КГУ — контрольно-габаритные устройства (верхнего габарита, устанавливаются перед мостами с ездой понизу)
КК — козловой кран
КЛУБ — Комплексное Локомотивное Устройство Безопасности
КОМ — машина для очистки кюветов
КП — контрольный пост; колесная пара
КПА — контрольный пункт автосцепки
КПД — Контроль Параметров Движения (электронный скоростемер)
КР — купейные вагоны с радиоузлом; капитальный ремонт
КРП — контрольно-ремонтный пункт; капитальный ремонт с продлением срока эксплуатации
КСАУ СП — комплексная система автоматизированного управления сортировочным процессом
КСАУ СС — комплексная система автоматизированного управления сортировочной станцией; состоит из КСАУ СП и информационно-планирующего уровня станции (ИПУ)
КТП — Комплектная Трансформаторная Подстанция
КТПО — Комплектная Трансформаторная Подстанция Подъёмно-Опускного типа
КТСМ — Комплекс Технических Средств Многофункциональный/Модернизированый (потомок ДИСКа)
Л — пассажирская служба
ЛАЗ — Линейно-Аппаратный Зал связи
ЛБК — отделенческая группа по учету, распределению и использованию мест
ЛВОК — начальник вокзала
ЛВЧД — вагонное депо для пассажирских вагонов (обычно совмещается с ПТС)
ЛНП — начальник (механик-бригадир) поезда
ЛОВД — это линейной отделение внутренних дел
ЛОВДТ — линейное отделение внутренних дел на транспорте
ЛП — пассажирский поезд (на некоторых дорогах)
ЛПМ — линейный пункт транспортной милиции
ЛРК — ревизор-контролер пассажирских поездов
ЛРКИ — ревизор-инструктор
ЛСПП — дежурный по парку
ЛСЦ — маневровый диспетчер
М — мягкие вагоны; служба грузовой и коммерческой работы
МАЛС — Маневровая Автоматическая Локомотивная Сигнализация
МВПС — моторвагонный подвижной состав
МВР — ревизор по весовому хозяйству
МДП — моторная платформа
МЖС — поездная межстанционная связь
МК — мягко-купейные вагоны (МИКСТ)
МКР — участковый коммерческий ревизор отделения дороги
МКРС — старший коммерческий ревизор отделения дороги
МКУ — Маршрутно-контрольные устройства (при ручных стрелках)
МП — мостовой поезд
МПРС — комплекс для выправки, шлифовки и подбивки стыков
МПТ — мотовоз путейский транспортный
МПЦ — микропроцессорная централизация стрелок и сигналов
МР — дорожный коммерческий ревизор; мелкий ремонт вагона
МРЦ — маршрутно-релейная централизация стрелок и сигналов
МСП — машина для смены стрелочных переводов
МХ — сектор хладотранспорта в службе М
МХП — хладотехник (практически упразднены)
МХР — ревизор по хладотранспорту (практически упразднены)
МЧ — механизированная дистанция погрузочно-разгрузочных работ
МЧК — МЧ с выполнением коммерческих операций (в СПб — Дирекция грузовой и коммерческой работы)
МЭЦ — электрическая централизация маневровых районов
МЮ — актово-претензионный сектор службы М
Н — управление дороги, начальник дороги
НБТ — дорожная служба охраны труда
НВП — начальник восстановительного поезда
НГ — главный инженер дороги
НГЧ — дистанция гражданских сооружений
НЗ — заместитель начальника дороги
НОД — начальник отделения дороги
НОДА — общий отдел отделения дороги
НОДБТ — начальник отдела охраны труда
НОДВ — отдел вагонного хозяйства (подвижного состава) отделения дороги
НОДВИС — инспектор по контролю за сохранностью вагонного парка
НОДГ — главный инженер отделения дороги
НОДЗ — отдел труда и зарплаты отделения дороги
НОДИС — инспектор по контролю за исполнением поручений НОДа
НОДК — начальник отдела управления персоналом отделения дороги
НОДЛ — начальник пассажирского отдела (там же)
НОДМ — начальник отдела грузовой и коммерческой работы отделения дороги
НОДН — начальник отдела перевозок отделения дороги
НОДО — первый отдел отделения дороги
НОДП — отдел пути отделения дороги
НОДР — второй (режимный) отдел отделения дороги
НОДТ — локомотивный отдел отделения дороги
НОДУ — отдел статистического учета и анализа отделения дороги
НОДФ — финансовый отдел отделения дороги
НОДХ — отдел материально-технического снабжения отделения дороги
НОДШ — отделение сигнализации и связи
НОДЮ — юридический отдел отделения дороги
НОК — дорожная служба управления персоналом
НОР — управление военизированной охраны
НОРВ — отдел военизированной охраны
НФ — финансовая служба дороги
НФКР — участковый финансовый ревизор
НФКРС — старший финансовый ревизор
НХ — дорожная служба материально-технического снабжения
НХГ — главный материальный склад Дороги
НХГУ — участок ГМС
НХО — отдел МТС (2-е подчинение = НОД+НХ)
НХОУ — участок отдела
НЧУ — дорожная служба статистического учета и анализа
НЮ — юридическая служба управления дороги
О — вагоны с общими местами
ОБЛ — вагоны областного типа
ОБЛБ — вагоны областного типа с буфетом
ОДБ — отдельное дорожное бюро (учет, распределение и использование мест)
ОК — купейные вагоны с общими местами
ОМ — мягкие вагоны с общими местами
ОПМС — опытная путевая машинная станция
ОПМСГ — главный инженер ОПМС
ОПЦ — оператор поста централизации стрелочных переводов
П — почтовые вагоны; служба пути
ПАБ — полуавтоматическая блокировка
ПБ — планировщик балласта
ПГС — перегонная связь
ПДК — погрузочный кран
ПДМ — дорожная ремонтно-механическая мастерская
ПД — дорожный мастер
ПДБ — бригадир пути
ПДС — старший дорожный мастер
ПДС — поездная диспетчерская связь
ПИТ — Путевой Источник Тока (применяется в системе защиты от электрокоррозии)
ПКО — пункт коммерческого осмотра вагонов
ПКТО — пункт контрольно-технического обслуживания
ПЛ — плацкартные вагоны
ПМГ — путевой моторный гайковерт
ПМС — путевая машинная станция
ПМСГ — главный инженер ПМС
ПОНАБ — Прибор Обнаружения Нагретых Аварийно Букс
ПОТ — пункт опробования тормозов
ПП — пожарный поезд
ППВ — пункт подготовки вагонов к перевозкам
ППЖТ — промышленное предприятие железнодорожного транспорта
ПРБ — путерихтовочная машина Балашенко
ПРЛ — путеремонтная летучка
ПРМЗ — путевой ремонтно-механический завод
ПРСМ — передвижная рельсосварочная машина
ПС — начальник вагона-путеизмерителя
ПСКС — пост секционирования контактной сети
ПТО — пункт технического обслуживания вагонов
ПТОЛ — пункт технического обслуживания локомотивов
ПТП — пункт технической передачи вагонов на подъездные пути ППЖТ
ПТС — Пассажирская Техническая Станция
ПТЭ — Правила технической эксплуатации железных дорог
ПЧ — дистанция пути, начальник дистанции пути
ПЧМех — дистанционная мастерская
ПЧЗ — зам. начальника дистанции пути (он же ЗамПЧ)
ПЧЛ — дистанция защитных лесонасаждений
ПЧП — балластный карьер
ПЧУ — начальник участка пути
ПШ — шпалопропиточный завод
ПЭМ — поездной электромеханик
Р1 — вагоны габарита «РИЦ» I класса
Р2 — вагоны габарита»РИЦ» I и II класса
РБ — дорожный ревизор по безопасности движения поездов и автотранспорта
РБА — дорожный ревизор автомобильной службы
РБВ — дорожный ревизор вагонной службы
РБД — дорожный ревизор службы движения
РБМ — дорожный ревизор службы грузовой
РБП — дорожный ревизор службы пути
РБТ — дорожный ревизор локомотивной службы
РБЧС (РБО) — дорожный ревизор аппарата РБ по чрезвычайным ситуациям (опасным грузам)
РБШЭ — дорожный ревизор службы сигнализации, связи и электроснабжения
РВЦ — региональный вычислительный центр
РЖДС — Росжелдорснаб - филиал ОАО «РЖД»
РКП — редукторно-карданный привод вагонного генератора (бывает от торца оси или от середины оси КП)
РМН — реле максимального напряжения генератора (служит для защиты потребителей эл. энергии вагона от перенапряжения)
РПБ — то же, что и ПАБ (системы РПБ ГТСС, РПБ КБ ЦШ)
РПН — реле пониженного напряжения; защита аккумулятора вагона от глубокого разряда
РПЦ — релейно-процессорная централизация
РСП — рельсосварочный поезд
РЦ — рельсовая цепь; релейная централизация
РЦС — региональный центр связи
РШ, РШС — релейный шкаф сигнальной точки
РШП — рельсошлифовальный поезд
САВПЭ — Система Автоматического Ведения Поезда и Экономии Электроэнергии
САИД «Пальма» — Система Автоматической Идентификации, главным образом предназначающаяся для службы перевозок (Движения)
САУТ — Система Автоматического Управления Тормозами
СВ — мягкие вагоны с 2-местными купе с верхними и нижними полками
СВН — то же с нижними полками
СВМ — то же с 2-местными и 4-местными купе
СДС — служебная диспетчерская связь
СИРИУС — сетевая итнегрированная система российская информационно-управляющая система
СКНБ — система контроля нагрева букс в пассажирских вагонах
СМП — строительно-монтажный поезд
СПД ЛП — система передачи данных с линейного пункта
СПС — специальный подвижной состав
СР — средний ремонт
ССПС — самоходный СПС
СТП — станционная тяговая подстанция
СТЦ — станционный технологический центр
СУРСТ — система управления работой станции
СЦБ — сигнализация, централизация, блокировка
Т — локомотивная служба
Т1-2 — вагоны габарита «РИЦ» I и II класса
ТГЛ — Телеуправление Горочным Локомотивом
ТГНЛ — телеграмма-натурный лист грузового поезда
ТехПД — технологический центр по обработке перевозочных документов (не более одного на каждое отделение дороги)
ТКП — текстропно-карданный привод вагонного генератора
ТМО — тоннельно-мостовой отряд
ТНЦ — локомотивный диспетчер
ТНЦС — старший локомотивный диспетчер
ТП — тяговая подстанция
ТР — текущий ремонт
ТРЦ — тональные рельсовые цепи
ТРКП — текстропно-редукторно-карданный привод вагонного генератора
ТСКБМ — Телемеханическая Система Контроля Бдительности Машиниста
ТЧ — тяговая часть (локомотивное депо); начальник депо
ТЧЗр — заместитель начальника локомотивного депо по ремонту
ТЧЗэ — заместитель начальника локомотивного депо по эксплуатации
ТЧЗк — заместитель начальника локомотивного депо по кадрам
ТЧЗт — заместитель начальника локомотивного депо по топливу
ТЧЗс — заместитель начальника локомотивного депо по снабжению
ТЧГ — главный инженер депо
ТЧГТ — главный технолог депо
ТЧИ — инженер из депо
ТЧМ — машинист
ТЧМИ — машинист-инструктор
ТЧМП — помощник машиниста
ТЧПЛ — приемщик локомотивов (принимает локомотивы из ремонта)
ТЭУ — тягово-энергетическая установка
УГР — уровень головки рельса
УЗП — Устройство Заграждения Переезда
УК — путеукладочный кран
УК25СП — путеукладочный кран для смены стрелочных переводов
УКБМ — Устройство Контроля Бдительности Машиниста (лампочки системы Рема Лобовкина)
УКСПС — Устройство для Контроля Схода Подвижного Состава
УРБ — отделенческий ревизор по безопасности движения поездов и автотранспорта
УРБВ — отделенческий ревизор аппарата УРБ по вагонной службе
УРБД — отделенческий ревизор аппарата УРБ по службе движения
УРБП — отделенческий ревизор аппарата УРБ по службе пути
УРБТ — отделенческий ревизор аппарата УРБ по локомотивной службе
УРБА — отделенческий ревизор аппарата УРБ по автомобильной службе
УРБМ — отделенческий ревизор аппарата УРБ по грузовой службе
УРБЧС (УРБО) — отделенческий ревизор аппарата УРБ по чрезвычайным ситуациям (опасным грузам)
УРБШЭ — отделенческий ревизор службы сигнализации, связи и энергоснабжения
УСАБ — усовершенствованная АБ
УСАБ-Ц — УСАБ с централизованным размещением аппаратуры
УСАВП — Усовершенствованая Система Автоматического Ведения Поезда
УТС — упор тормозной стационарный; устройство торможения состава
УКП СО — устройство контроля свбодности перегона методом счёта осей подвижного состава
УКРУП — устройство контроля усилия перевода
УУ АПС СО — устройство управления автоматической перездной сигнализацией с применением аппаратуры счёта осей подвижного состава
УЭЗ — управление экономической защиты МПС
Ц — президент ОАО «РЖД»
ЦАБ — Централизованная Автоматическая Блокировка
ЦБТ — управление охраны труда РЖД
ЦВ — департамент вагонного хозяйства РЖД
ЦД — департамент управления перевозками РЖД
ЦДВ — отдел негабаритных и специальных перевозок ЦД
ЦДГР — главный ревизор ЦД
ЦЗ — заместитель президента ОАО «РЖД»
ЦИ — управление внешних связей РЖД
ЦИС — департамент информатизации и связи РЖД
ЦКАДР -департамент кадров и учебных заведений РЖД
ЦЛ -департамент пассажирских сообщений РЖД
ЦМ — департамент грузовой и коммерческой работы РЖД
ЦМГВ — цельнометаллический грузовой вагон.
ЦМКО — отдел по организации и условиям перевозок ЦМ
ЦМКЮ — отдел по профилактике сохранности перевозимых грузов ЦМ
ЦМР — главный коммерческий ревизор РЖД
ЦМХ — отдел скоропортящихся грузов ЦМ
ЦН — управление делами РЖД
ЦНИИТЭИ — Московский филиал ВНИИУП (ранее — центральный научно-исследовательский институт технико-экономических исследований на железнодорожном транспорте)
ЦП — департамент пути и сооружений РЖД
ЦРБ — аппарат главного ревизора по безопасности движения поездов и автотранспорта, главный ревизор по безопасности движения поездов и автотранспорта.
ЦРБ — департамент безопасности движения и экологии РЖД
ЦРБА — главный ревизор аппарата ЦРБ по автомобильной службе
ЦРБВ — главный ревизор аппарата ЦРБ по вагонной службе
ЦРБД — главный ревизор аппарата ЦРБ по службе движения
ЦРБМ — главный ревизор аппарата ЦРБ по грузовой службе
ЦРБТ — главный ревизор аппарата ЦРБ по локомотивной службе
ЦРБП — главный ревизор аппарата ЦРБ по службе пути
ЦРБЧС (ЦРБО) — главный ревизор аппарата ЦРБ по чрезвычайным ситуациям (опасным грузам)
ЦРБШЭ — ревизор по службе сигнализации, связи и энергоснабжения
ЦРЖ — департамент реформирования железнодорожного транспорта РЖД
ЦСЖТ — совет по железнодорожному транспорту государств-участников Содружества Независимых Государств, Литовской Республики, Латвийской Республики, Эстонской Республики
ЦТ — департамент локомотивного хозяйства РЖД
ЦТЕХ — департамент технической политики РЖД
ЦТВР — Главное управление по ремонту подвижного состава и производству запасных частей
ЦУВС — департамент здравоохранения РЖД
ЦУКС — департамент капитального строительства и эксплуатации объектов железнодорожного транспорта РЖД
ЦУО — управление военизированной охраны РЖД
ЦУП -центр управления перевозками РЖД
ЦУШ — управление имущества и реестра предприятий РЖД
ЦФ -департамент финансов РЖД
ЦФТО — центр фирменного транспортного обслуживания РЖД
ЦЧУ — управление статистики РЖД
ЦШ — департамент сигнализации, централизации и блокировки РЖД
ЦЭ — департамент электрификации и энергоснабжения РЖД
ЦЭУ — департамент экономики РЖД
ЦЮ — юридическое управление РЖД
ЧДК — частотный диспетчерский контроль
Ш — служба сигнализации и связи
ШМ — электромонтёр
ШН — электромеханик СЦБ или связи
ШНС — старший электромеханик СЦБ или связи
ШНЦ — механик СЦБ
ШНЦС — старший механик СЦБ
ШРМ — шпалоремонтная мастерская
ШЦМ — электромонтер СЦБ или связи
ШЧ — дистанция сигнализации, централизации и блокировки (быв. дистанция сигнализации и связи, быв. Шнуровая Часть либо Шиллингова Часть)
ШЧГ — главный инженер ШЧ
ШЧД — диспетчер дистанции или дежурный инженер дистанции
ШЧЗ — зам. ШЧ (обычно их двое: по связи и по СЦБ)
ШЧИС — старший инженер ШЧ
ШЧУ — начальник производственного участка СЦБ или связи
ЩОМ — щебнеочистительная машина
Э — дорожная служба электрификации и энергоснабжения
ЭДС — энергодиспетчерская связь
ЭЖС — электрожезловая система
ЭМС — электромеханическая служба
ЭПТ — ЭлектроПневматический Тормоз
ЭС — служба энергоснабжения
ЭССО — электронная система счета осей
ЭТРАН — электронная транспортная накладная
ЭЦ — электрическая централизация стрелок и сигналов
ЭЦ-Е, ЭЦ-ЕМ — электрическая централизация единого ряда (микроэлектронная, она же микропроцессорная)
ЭЦ-И — электрическая централизация с индустриальной системой монтажа
ЭЦ-МПК — электрическая централизация на базе микроЭВМ и программируемых контроллеров
ЭЧ — дистанция электроснабжения, начальник дистанции электроснабжения
ЭЧГ — главный инженер дистанции электроснабжения.
ЭЧЗК — заместитель начальника дистанции электроснабжения по контактной сети
ЭЧЗП — заместитель начальника дистанции электроснабжения по тяговым подстанциям
ЭЧК — район контактной сети дистанции электроснабжения, начальник района контактной сети
ЭЧКМ — мастер ЭЧК
ЭЧП — начальник тяговой подстанции
ЭЧС — сетевой район дистанции электроснабжения, начальник сетевого района
ЭЧЦ — энергодиспетчер дистанции электроснабжения
ЭЧЦС — старший ЭЧЦ
ЭЧЭ — тяговая подстанция
"""


@component
class DictTranslator:
    def __init__(self):
        self.system_prompt = """
          You are an assistant that expands abbreviations in sentences.

          *Instructions:*
          - When given a sentence containing abbreviations, you should return the sentence with all abbreviations expanded.
          - *Use only* the abbreviations and their expansions provided within the `<context>` tags.
          - *Do not invent* or use any abbreviations not present in the provided context.
          - The output must be in *Russian*.
          - Do not provide explanations or additional information—only the expanded sentence.

          *Dictionary of Abbreviations:*
          <context>
          {{context}}
          </context>


          <examples>
          1. *Input:* "ДСП и ДНЦ встретились на ПЧ, чтобы обсудить ПМС."
            *Output:* "Дежурный по станции и поездной диспетчер встретились на дистанции пути, чтобы обсудить путевую машинную станцию."

          2. *Input:* "НГЧ забыл АБ на АЛСН и срочно побежал в ТЧ."
            *Output:* "Начальник дистанции гражданских сооружений забыл автоблокировку на автоматической локомотивной сигнализации и срочно побежал в локомотивное депо."

          3. *Input:* "ТЧМП опоздал на ПТОЛ, и НГ отправил его на АРМ ТВК."
            *Output:* "Помощник машиниста опоздал на пункт технического обслуживания локомотивов, и начальник дороги отправил его за автоматизированное рабочее место товарного кассира."

          4. *Input:* "ПЧЗ забыл КПД в АСУ КП, и теперь его ищет ДСПП."
            *Output:* "Заместитель начальника дистанции пути забыл контроль параметров движения в автоматизированной системе управления контейнерным пунктом, и теперь его ищет дежурный по парку."

          5. *Input:* "РБЧС вызвал СЦБ, чтобы проверить, как работает ЭЧ на ПЧ."
            *Output:* "Ревизор по чрезвычайным ситуациям вызвал сигнализацию, централизацию и блокировку, чтобы проверить, как работает дистанция электроснабжения на дистанции пути."
          </examples>
          """

        self.user_prompt_template = """
                                *Input:* "{{query}}"
                                *Output:*
                                """

        builder = ChatPromptBuilder(variables=["query", "context"])
        llm = OpenAIChatGenerator(
            api_key=Secret.from_token("VLLM-PLACEHOLDER-API-KEY"),
            model=MODEL_NAME,
            api_base_url=MODEL_URL,
            generation_kwargs={
                "max_tokens": 2048,
                "temperature": 0.3},
            timeout=600
        )
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=builder)
        self.pipeline.add_component(name="llm", instance=llm)
        self.pipeline.connect("builder", "llm")

    @component.output_types(queries=str)
    def run(self, query: str, context: str = rzd_dict):
        messages = [
            ChatMessage.from_system(self.system_prompt),
            ChatMessage.from_user(self.user_prompt_template)
        ]

        result = self.pipeline.run({
            'builder': {
                'template': messages,
                'query': query,
                'context': context
            }
        },
            include_outputs_from={"builder"})
        response_text = result['llm']['replies'][0].content.strip()

        return ModelResponse(answer=response_text)


my_translator = DictTranslator()


def get_translation(question: str) -> str:
    return my_translator.run(query=question).answer
