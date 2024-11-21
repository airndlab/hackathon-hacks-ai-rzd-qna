import gc
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from haystack import Pipeline, Document, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import PyPDFToDocument, DOCXToDocument
from haystack.components.converters.csv import CSVToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import ComponentDevice, Device, Secret
from pydantic import BaseModel

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = ComponentDevice.from_single(Device.gpu(id=0))
else:
    device = ComponentDevice.from_single(Device.cpu())

VLLM_URL = os.getenv("VLLM_URL")

MAIN_DOCS_DIR = os.getenv('MAIN_DOCS_DIR')

# PROMPT CONFIG
PROMPTS_CONFIG_PATH = os.getenv("PROMPTS_CONFIG_PATH")
with open(PROMPTS_CONFIG_PATH, 'r', encoding='utf-8') as file:
    prompt_config = yaml.safe_load(file)

# DICTS CONFIG
DICTS_CONFIG_PATH = os.getenv("DICTS_CONFIG_PATH")
with open(DICTS_CONFIG_PATH, 'r') as file:
    dicts_config = yaml.safe_load(file)

ENG_TO_RUS = dicts_config.get("ENG_TO_RUS")
RUS_GENITIVE = dicts_config.get("RUS_GENITIVE")
RUS_INSTRUMENTAL = dicts_config.get("RUS_INSTRUMENTAL")

# RAG CONFIG
RAG_CONFIG_PATH = os.getenv("RAG_CONFIG_PATH")
with open(RAG_CONFIG_PATH, "r") as file:
    rag_config = yaml.safe_load(file)

model = rag_config.get("model")
embedding_model = rag_config.get("embedding_model")

split_function_config = rag_config.get("split_function", {})
max_chunk_size = split_function_config.get("max_chunk_size", 300)
overlap = split_function_config.get("overlap", 0)

rag_gen_kwargs = rag_config.get("rag_gen_kwargs", {})
json_gen_kwargs = rag_config.get("json_gen_kwargs", {})


def custom_split_function(
        text: str,
        max_chunk_size: int = max_chunk_size,
        overlap: int = overlap
) -> List[str]:
    """
    Разбивает текст на чанки. Если текст содержит пункты (например, 7.2, 7.2.1 и т.д.),
    разбивает по ним, учитывая перекрытие. Если пункт слишком длинный, разбивает его
    на части с перекрытием и добавляет номер пункта в начало каждого чанка.
    Если в тексте нет пунктов, просто разбивает текст на чанки по количеству слов
    с заданным перекрытием.

    :param text: Входной текст для разбиения.
    :param max_chunk_size: Максимальное количество слов в чанке.
    :param overlap: Количество перекрывающихся слов между чанками.
    :return: Список чанков текста.
    """

    def word_count(text: str) -> int:
        """Возвращает количество слов в тексте."""
        return len(text.split())

    def split_large_text(
            text: str,
            max_chunk_size: int,
            overlap: int) -> List[str]:
        """
        Разбивает большой текст без пунктов на чанки заданного размера с перекрытием.

        :param text: Текст для разбиения.
        :param max_chunk_size: Максимальное количество слов в чанке.
        :param overlap: Количество перекрывающихся слов между чанками.
        :return: Список чанков.
        """
        words = text.split()
        chunks = []
        idx = 0
        while idx < len(words):
            end_idx = idx + max_chunk_size
            chunk_words = words[idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            if end_idx >= len(words):
                break
            idx += max_chunk_size - overlap  # Перекрытие в k слов
        return chunks

    def split_large_point(
            point_text: str,
            point_number: str,
            max_chunk_size: int,
            overlap: int) -> List[str]:
        """
        Разбивает большой пункт на чанки с перекрытием, добавляя номер пункта в начало каждого чанка.

        :param point_text: Текст пункта.
        :param point_number: Номер пункта (например, "7.2").
        :param max_chunk_size: Максимальное количество слов в чанке.
        :param overlap: Количество перекрывающихся слов между чанками.
        :return: Список чанков.
        """
        words = point_text.split()
        chunks = []
        idx = 0
        total_words = len(words)

        while idx < total_words:
            end_idx = idx + max_chunk_size
            chunk_words = words[idx:end_idx]
            # Добавляем номер пункта в начало чанка, если это не первый чанк
            if idx != 0:
                chunk_text = f"{point_number} " + ' '.join(chunk_words)
            else:
                chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            if end_idx >= total_words:
                break
            idx += max_chunk_size - overlap
        return chunks

    # Регулярное выражение для поиска номеров пунктов (например, 7.2, 7.2.1 и т.д.)
    point_pattern = r"(?:(?:\n|^)(\d+(?:\.\d+)+))"

    # Находим все позиции пунктов в тексте
    points = [(m.start(1), m.group(1)) for m in re.finditer(point_pattern, text)]

    chunks = []

    if not points:
        # Если в тексте нет пунктов, просто разбиваем текст на чанки по словам
        chunks = split_large_text(text, max_chunk_size, overlap)
        return chunks

    # Если текст начинается без пунктов, обрабатываем этот префикс
    if points and points[0][0] != 0:
        prefix_text = text[:points[0][0]].strip()
        if prefix_text:
            prefix_chunks = split_large_text(prefix_text, max_chunk_size, overlap)
            chunks.extend(prefix_chunks)

    # Обрабатываем каждый пункт
    for idx, (pos, point_number) in enumerate(points):
        # Начало пункта
        start_pos = pos
        # Конец пункта - это начало следующего пункта или конец текста
        end_pos = points[idx + 1][0] if idx + 1 < len(points) else len(text)
        point_text = text[start_pos:end_pos].strip()

        # Проверяем, вписывается ли пункт в максимальный размер чанка
        if word_count(point_text) <= max_chunk_size:
            chunks.append(point_text)
        else:
            # Разбиваем пункт на части с перекрытием
            split_point_chunks = split_large_point(point_text, point_number, max_chunk_size, overlap)
            chunks.extend(split_point_chunks)

    # Обработка текста после последнего пункта, если есть
    last_point_end = points[-1][0] + len(points[-1][1])
    suffix_text = text[last_point_end:].strip()
    if suffix_text:
        suffix_chunks = split_large_text(suffix_text, max_chunk_size, overlap)
        chunks.extend(suffix_chunks)

    return chunks


def create_in_memory_document_store():
    document_store = InMemoryDocumentStore()

    return document_store


def create_indexing_pipeline(
        document_store,
):
    file_type_router = FileTypeRouter(mime_types=[
        "application/pdf",
        "text/csv",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"])

    pdf_converter = PyPDFToDocument()
    docx_converter = DOCXToDocument()
    csv_converter = CSVToDocument()

    document_joiner = DocumentJoiner()
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="function", splitting_function=custom_split_function, split_length=1)
    document_writer = DocumentWriter(document_store)
    document_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model, device=device)

    indexing_pipeline = Pipeline()

    indexing_pipeline.add_component(instance=file_type_router, name="file_type_router")
    indexing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
    indexing_pipeline.add_component(instance=docx_converter, name="docx_converter")
    indexing_pipeline.add_component(instance=csv_converter, name="csv_converter")
    indexing_pipeline.add_component(instance=document_joiner, name="document_joiner")
    indexing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    indexing_pipeline.add_component(instance=document_splitter, name="document_splitter")
    indexing_pipeline.add_component(instance=document_embedder, name="document_embedder")
    indexing_pipeline.add_component(instance=document_writer, name="document_writer")

    indexing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
    indexing_pipeline.connect(
        "file_type_router.application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "docx_converter.sources")
    indexing_pipeline.connect("file_type_router.text/csv", "csv_converter.sources")

    indexing_pipeline.connect("pypdf_converter", "document_joiner")
    indexing_pipeline.connect("docx_converter", "document_joiner")
    indexing_pipeline.connect("csv_converter", "document_joiner")

    indexing_pipeline.connect("document_joiner", "document_splitter")
    indexing_pipeline.connect("document_splitter", "document_cleaner")
    indexing_pipeline.connect("document_cleaner", "document_embedder")
    indexing_pipeline.connect("document_embedder", "document_writer")

    return indexing_pipeline


document_store = create_in_memory_document_store()

indexing_pipeline = create_indexing_pipeline(
    document_store=document_store,
)


def do_reindex():
    orgs_dirs = Path(MAIN_DOCS_DIR)
    for org_dir in orgs_dirs.iterdir():
        if org_dir.is_dir():
            source_docs = list(org_dir.glob("**/*"))

            indexing_pipeline.run({
                "file_type_router": {"sources": source_docs},
                "pypdf_converter": {"meta": {"organization": org_dir.name}},
                "csv_converter": {"meta": {"organization": org_dir.name}},
                "docx_converter": {"meta": {"organization": org_dir.name}},
            })


do_reindex()


def create_generator(gen_kwargs=None):
    return OpenAIChatGenerator(
        api_key=Secret.from_token("VLLM-PLACEHOLDER-API-KEY"),
        model=model,
        api_base_url=VLLM_URL,
        generation_kwargs=gen_kwargs,
        timeout=600,
    )


@component
class QueryExpander:
    def __init__(
            self,
            system_prompt: str,
            user_prompt_template: str,
            json_gen_kwargs,
    ):
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

        builder = ChatPromptBuilder(variables=["query", "user_info"])
        llm = create_generator(json_gen_kwargs)

        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=builder)
        self.pipeline.add_component(name="llm", instance=llm)
        self.pipeline.connect("builder", "llm")

    @component.output_types(queries=List[str])
    def run(
            self,
            query: str,
            user_info: str,
    ):
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

            except Exception as e:
                logger.warning(e)
                return {"queries": [query]}

        return {"queries": expanded_queries}


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
    def __init__(self, retriever: InMemoryEmbeddingRetriever, filters=None, top_k: int = 2,
                 score_threshold: float = 0.3):

        self.retriever = retriever
        self.results = []
        self.ids = set()
        self.top_k = top_k
        self.filters = filters
        self.score_threshold = score_threshold

    def add_document(self, document: Document):
        if (document.id not in self.ids) and (document.score > self.score_threshold):
            self.results.append(document)
            self.ids.add(document.id)

    @component.output_types(documents=List[Document])
    def run(self, emdeddings: List[List[str]], top_k=2, filters=None):
        self.results = []
        self.ids = set()

        for emdedding in emdeddings:
            result = self.retriever.run(query_embedding=emdedding['embedding'], filters=filters, top_k=top_k)
            for doc in result['documents']:
                self.add_document(doc)

        self.results.sort(key=lambda x: x.score, reverse=True)

        return {"documents": self.results}


def create_rag_pipeline(
    document_store,
    ) -> Pipeline:
  expander = QueryExpander(
      system_prompt = prompt_config['query_expander_system_prompt'],
      user_prompt_template = prompt_config['query_expander_user_prompt_template'],
      json_gen_kwargs = json_gen_kwargs,
  )
  text_embedder = MultiQueryTextEmbedder(SentenceTransformersTextEmbedder(model=embedding_model, device=device))
  retriever = MultiQueryInMemoryRetriever(InMemoryEmbeddingRetriever(document_store))
  generator = create_generator(rag_gen_kwargs)

  chat_prompt_builder = ChatPromptBuilder(variables=["documents", "question", "user_name", "user_info", "changes"])

  rag_pipeline = Pipeline()

  rag_pipeline.add_component("expander", expander)
  rag_pipeline.add_component("text_embedder", text_embedder)
  rag_pipeline.add_component("retriever", retriever)
  rag_pipeline.add_component("prompt_builder", chat_prompt_builder)
  rag_pipeline.add_component("llm", generator)

  rag_pipeline.connect("expander.queries", "text_embedder.queries")
  rag_pipeline.connect("text_embedder.embeddings", "retriever")
  rag_pipeline.connect("retriever", "prompt_builder.documents")
  rag_pipeline.connect("prompt_builder", "llm")

  return rag_pipeline


chat_system_message = ChatMessage.from_system(prompt_config['chat_system_prompt'])
chat_user_message = ChatMessage.from_user(prompt_config['chat_user_prompt_template'])
chat_messages = [chat_system_message, chat_user_message]

changes_system_message = ChatMessage.from_system(prompt_config['changes_system_prompt'])
changes_user_message = ChatMessage.from_user(prompt_config['changes_user_prompt_template'])
changes_messages = [changes_system_message, changes_user_message]

rag_pipeline = create_rag_pipeline(
    document_store=document_store,
)


class Reference(BaseModel):
    document: str
    paragraph: str


class ModelResponse(BaseModel):
    answer: str
    references: Optional[List[Reference]] = None


def get_chat_response(
    question: str,
    user_name: str="",
    user_info: str="",
    user_org: str = "",
    ) -> ModelResponse:
    response = rag_pipeline.run({
    "expander": {
        "query": question,
        "user_info": user_info
        },
    "prompt_builder": {"question": question,
                       "template": chat_messages,
                       "user_name": user_name,
                       "user_info": user_info},
    "retriever":{
      "filters": {
          "operator": "OR",
          "conditions":[
              {"field": "meta.organization", "operator": "==", "value": "RZD"},
              {"field": "meta.organization", "operator": "==", "value": user_org},
          ]
      },
      }
    },
    include_outputs_from={"expander", "retriever","prompt_builder"})

    if len(response['retriever']['documents']) == 0:
      return ModelResponse(
          answer = "Я не знаю ответа на ваш вопрос.",
          references = None
      )
    print(response['expander'])
    print(len(response['retriever']['documents']))

    response_text = response["llm"]["replies"][0].content

    references = None
    try:
        # Находим индекс последнего символа ']'
        last_bracket_idx = response_text.rfind(']')
        if last_bracket_idx != -1:
            # Находим индекс соответствующего символа '[' перед ']'
            first_bracket_idx = response_text.rfind('[', 0, last_bracket_idx)
            if first_bracket_idx != -1:
                json_str = response_text[first_bracket_idx:last_bracket_idx+1]

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


def get_benefits_response(
    changes_dict: dict,
    user_name: str = '',
    user_info: str = '',
    user_org: str = '',
    top_k = 3
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
    print(prompt_builder_question)

   # Запускаем RAG пайплайн
    response = rag_pipeline.run({
        "expander": {
            "query": retriever_question,
            "user_info": user_info
        },
        "prompt_builder": {
            "question": prompt_builder_question,
            "template": changes_messages,
            "user_name": user_name,
            "user_info": user_info,
            "changes": russian_changes_base
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
    include_outputs_from={"retriever","prompt_builder"})

    response_text = response["llm"]["replies"][0].content

    print(len(response['retriever']['documents']))

    # Извлечение JSON-объекта по символам [ и ] с конца строки
    references = None
    try:
        # Находим индекс последнего символа ']'
        last_bracket_idx = response_text.rfind(']')
        if last_bracket_idx != -1:
            # Находим индекс соответствующего символа '[' перед ']'
            first_bracket_idx = response_text.rfind('[', 0, last_bracket_idx)
            if first_bracket_idx != -1:
                json_str = response_text[first_bracket_idx:last_bracket_idx+1]

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


@component
class DictTranslator:
    def __init__(
            self,
            system_prompt: str,
            user_prompt_template: str,
            context: str,
            gen_kwargs: dict = None,
    ):
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.context = context

        builder = ChatPromptBuilder(variables=["query", "context"])
        llm = create_generator(gen_kwargs)
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=builder)
        self.pipeline.add_component(name="llm", instance=llm)
        self.pipeline.connect("builder", "llm")

    @component.output_types(queries=str)
    def run(
            self,
            query: str,
    ):
        messages = [
            ChatMessage.from_system(self.system_prompt),
            ChatMessage.from_user(self.user_prompt_template)
        ]

        result = self.pipeline.run({
            'builder': {
                'template': messages,
                'query': query,
                'context': self.context
            }
        },
            include_outputs_from={"builder"})

        logger.info(result['builder'])
        response_text = result['llm']['replies'][0].content.strip()

        return ModelResponse(answer=response_text)


my_translator = DictTranslator(
    system_prompt=prompt_config['dict_translator_system_prompt'],
    user_prompt_template=prompt_config['dict_translator_user_prompt_template'],
    context=prompt_config['rzd_dict'],
    gen_kwargs={
        "max_tokens": 2048,
        "temperature": 0.3},
)


def get_translation(query):
    return my_translator.run(query).answer
