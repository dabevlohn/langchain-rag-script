import pprint
import re
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.utilities import SearxSearchWrapper
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_core.documents import Document

class RAGArticleGenerator:
    def __init__(self, model_name: str = "gpt-oss:20b", searx_host: str = "http://127.0.0.1:8080"):
        self.model = OllamaLLM(model=model_name)
        self.search = SearxSearchWrapper(searx_host=searx_host)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        self.sources_metadata = {}  # Хранение метаданных источников
        self.source_counter = 1

        # Расширенный шаблон промпта с инструкциями по форматированию
        self.template = """You are an expert writer creating a comprehensive article based on provided context documents.

CONTEXT DOCUMENTS:
{context_with_sources}

TASK: Write a detailed, well-structured article about: {question}

FORMATTING REQUIREMENTS:
1. Create a compelling article with clear sections and subsections
2. Use information from the provided context documents
3. When using information from a specific source, add a citation in square brackets [1], [2], etc.
4. Each citation number corresponds to a source document
5. Write in a journalistic, engaging style
6. Include specific details, examples, and technical information from the sources
7. Ensure logical flow between paragraphs and sections

STRUCTURE:
- Introduction (engaging opening)
- Main content (multiple sections with subheadings)
- Conclusion (summary and insights)
- Do NOT include a references section - this will be added automatically

Begin writing the article now:"""

        self.prompt = ChatPromptTemplate.from_template(self.template)

    def search_and_collect_urls(self, query: str, num_results: int = 10) -> List[str]:
        """Поиск и сбор URL"""
        results = self.search.results(
            query,
            num_results=num_results,
            time_range="year",
        )

        urls = []
        for result in results:
            url = result.get('link')
            if url and url not in urls:
                urls.append(url)
                # Сохраняем метаданные источника
                self.sources_metadata[self.source_counter] = {
                    'url': url,
                    'title': result.get('title', 'Untitled'),
                    'snippet': result.get('snippet', '')
                }
                self.source_counter += 1

        print(f"Найдено {len(urls)} уникальных URL:")
        pprint.pp(urls)
        return urls

    def load_and_process_documents(self, urls: List[str]) -> List[Document]:
        """Загрузка и обработка документов с добавлением метаданных источников"""
        all_documents = []

        for i, url in enumerate(urls, 1):
            try:
                print(f"Загрузка документа {i}/{len(urls)}: {url}")

                loader = RecursiveUrlLoader(
                    url,
                    max_depth=2,
                    timeout=15,
                    check_response_status=True,
                    continue_on_failure=True,
                    prevent_outside=True,
                )

                documents = loader.load()

                if documents:
                    # Добавляем метаданные источника к каждому документу
                    for doc in documents:
                        doc.metadata['source_number'] = i
                        doc.metadata['source_url'] = url
                        doc.metadata['source_title'] = self.sources_metadata.get(i, {}).get('title', 'Untitled')

                    all_documents.extend(documents)
                    print(f"Загружено {len(documents)} документов с {url}")
                else:
                    print(f"Не удалось загрузить документы с {url}")

            except Exception as e:
                print(f"Ошибка при загрузке {url}: {str(e)}")
                continue

        print(f"Всего загружено {len(all_documents)} документов")
        return all_documents

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Создание векторного хранилища"""
        if not documents:
            raise ValueError("Нет документов для создания векторного хранилища")

        vector_store = Chroma(
            collection_name="article_sources",
            embedding_function=self.embeddings,
        )

        print("Добавление документов в векторное хранилище...")
        vector_store.add_documents(documents)
        print("Векторное хранилище создано")

        return vector_store

    def prepare_context_with_sources(self, retrieved_docs: List[Document]) -> str:
        """Подготовка контекста с нумерацией источников"""
        context_parts = []

        for doc in retrieved_docs:
            source_num = doc.metadata.get('source_number', 'Unknown')
            source_title = doc.metadata.get('source_title', 'Untitled')

            context_parts.append(f"=== ИСТОЧНИК {source_num}: {source_title} ===\n{doc.page_content}\n")

        return "\n".join(context_parts)

    def generate_article(self, query: str, max_retrieved_docs: int = 8) -> str:
        """Основная функция генерации статьи"""
        # 1. Поиск URL
        urls = self.search_and_collect_urls(query)

        if not urls:
            return "Не найдено источников для создания статьи."

        # 2. Загрузка документов
        documents = self.load_and_process_documents(urls)

        if not documents:
            return "Не удалось загрузить документы из найденных источников."

        # 3. Создание векторного хранилища
        vector_store = self.create_vector_store(documents)

        # 4. Создание ретривера с большим количеством документов
        retriever = vector_store.as_retriever(
            search_kwargs={"k": max_retrieved_docs}
        )

        # 5. Получение релевантных документов
        retrieved_docs = retriever.get_relevant_documents(query)

        # 6. Подготовка контекста с источниками
        context_with_sources = self.prepare_context_with_sources(retrieved_docs)

        # 7. Генерация статьи
        print("Генерация статьи...")
        chain = self.prompt | self.model

        response = chain.invoke({
            "question": query,
            "context_with_sources": context_with_sources
        })

        # 8. Добавление списка источников
        article_with_sources = self.add_sources_list(response)

        return article_with_sources

    def add_sources_list(self, article_text: str) -> str:
        """Добавление нумерованного списка источников в конец статьи"""
        sources_list = "\n\n## Источники\n\n"

        for source_num, metadata in self.sources_metadata.items():
            sources_list += f"{source_num}. {metadata['title']}\n   URL: {metadata['url']}\n\n"

        return article_text + sources_list

# Основная логика выполнения
if __name__ == "__main__":
    # Инициализация генератора
    generator = RAGArticleGenerator(
        model_name="qwen3:30b",  # Можно изменить на другую модель
        searx_host="http://127.0.0.1:8080"
    )

    # Запрос для поиска
    search_query = "Langchain Ollama RAG SearXNG integration tutorial"

    # Генерация статьи
    try:
        article = generator.generate_article(
            query="Write a comprehensive article about integrating LangChain, Ollama, and SearXNG for building RAG applications with local LLMs",
            max_retrieved_docs=10
        )

        print("\n" + "="*80)
        print("СГЕНЕРИРОВАННАЯ СТАТЬЯ:")
        print("="*80 + "\n")
        print(article)

        # Сохранение статьи в файл
        with open('generated_article.md', 'w', encoding='utf-8') as f:
            f.write(article)
        print(f"\nСтатья сохранена в файл: generated_article.md")

    except Exception as e:
        print(f"Ошибка при генерации статьи: {str(e)}")

# Альтернативное использование с другими параметрами:
"""
# Для других моделей:
generator = RAGArticleGenerator(model_name="llama3.1:8b")
generator = RAGArticleGenerator(model_name="qwen3:30b")
generator = RAGArticleGenerator(model_name="deepseek-r1:32b")

# Для других тем:
article = generator.generate_article(
    query="Write about the latest developments in open-source LLMs and their applications",
    max_retrieved_docs=15
)
"""
