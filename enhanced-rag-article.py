import pprint
import re
from typing import List, Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_core.documents import Document
import html2text
from urllib.parse import urlparse

class EnhancedRAGArticleGenerator:
    def __init__(self, model_name: str = "qwen3:30b", searx_host: str = "http://127.0.0.1:8080"):
        self.model = OllamaLLM(model=model_name)
        self.search = SearxSearchWrapper(searx_host=searx_host)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        self.sources_metadata = {}
        self.source_counter = 1

        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.body_width = 0
        self.html_converter.unicode_snob = True

        self.template = """You are an expert academic writer creating a comprehensive research article based on provided context documents.

AVAILABLE SOURCE DOCUMENTS:
{context_with_sources}

TASK: Write a detailed, well-structured article about: {question}

CRITICAL CITATION REQUIREMENTS:
1. When referencing specific information, data, quotes, or concepts from a source, immediately follow with a citation [X] where X is the source number
2. Citations must be semantically meaningful - each citation should guide readers to sources containing MORE DETAILED information about that specific topic
3. Place citations at the end of sentences or paragraphs that contain information from that source
4. Use multiple citations [1][2] when information is supported by multiple sources
5. Ensure each citation logically connects the content to the source that elaborates on that topic

WRITING STYLE:
- Academic but accessible tone
- Clear section headings (use ##, ###)
- Logical progression of ideas
- Incorporate specific technical details, examples, and methodologies from sources
- Each paragraph should develop a distinct aspect of the topic

STRUCTURE REQUIREMENTS:
- Engaging introduction with context and importance
- Multiple main sections covering different aspects
- Technical implementation details where relevant
- Real-world applications and use cases
- Critical analysis and limitations
- Future directions and conclusions
- DO NOT add a references section (will be added automatically)

CONTENT DEPTH:
- Minimum 1500 words
- Include technical specifications, code examples, and implementation details from sources
- Discuss advantages, limitations, and comparison with alternatives
- Provide practical insights and recommendations

Begin writing the comprehensive article now:"""

        self.prompt = ChatPromptTemplate.from_template(self.template)

    def search_and_collect_urls(self, query: str, num_results: int = 15) -> List[str]:
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
                domain = urlparse(url).netloc
                self.sources_metadata[self.source_counter] = {
                    "url": url,
                    "title": result.get('title', 'Untitled'),
                    "snippet": result.get('snippet', ''),
                    "domain": domain,
                    "content_summary": '',
                    "topics_covered": ''
                }
                self.source_counter += 1

        print(f"{len(urls)} URLs found:")
        pprint.pp(urls)
        return urls

    def convert_html_to_markdown(self, html_content: str) -> str:
        try:
            markdown_content = self.html_converter.handle(html_content)
            markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)
            markdown_content = re.sub(r'^\s*\*\s*(Home|Menu|Navigation|Skip to|Back to top).*$', '',
                                    markdown_content, flags=re.MULTILINE | re.IGNORECASE)
            lines = markdown_content.split('\n')
            cleaned_lines = []
            for line in lines:
                stripped = line.strip()
                if len(stripped) > 10 or stripped.startswith('#'):
                    cleaned_lines.append(line)
            return '\n'.join(cleaned_lines)
        except Exception as e:
            print(f"Error by HTML to Markdown conversion: {str(e)}")
            return html_content

    def extract_content_summary(self, markdown_content: str) -> Tuple[str, List[str]]:
        headers = re.findall(r'^#+\s+(.+)$', markdown_content, re.MULTILINE)
        summary = markdown_content[:300].replace('\n', ' ').strip()
        if len(markdown_content) > 300:
            summary += "..."
        topics = [header.strip() for header in headers[:5]]
        return summary, topics

    def load_and_process_documents(self, urls: List[str]) -> List[Document]:
        all_documents = []

        for i, url in enumerate(urls, 1):
            try:
                print(f"Loading {i}/{len(urls)}: {url}")

                loader = RecursiveUrlLoader(
                    url,
                    max_depth=1,
                    timeout=20,
                    check_response_status=True,
                    continue_on_failure=True,
                    prevent_outside=True,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (compatible; RAG-Bot/1.0)'
                    }
                )

                documents = loader.load()

                if documents:
                    processed_docs = []
                    for doc in documents:
                        markdown_content = self.convert_html_to_markdown(doc.page_content)
                        processed_doc = Document(
                            page_content=markdown_content,
                            metadata={
                                "source_number": i,
                                "source_url": url,
                                "source_title": str(self.sources_metadata.get(i, {}).get('title', 'Untitled')),
                                "source_domain": str(self.sources_metadata.get(i, {}).get('domain', '')),
                                "original_metadata": str(doc.metadata)
                            }
                        )
                        processed_docs.append(processed_doc)

                    if processed_docs:
                        combined_content = '\n'.join([doc.page_content for doc in processed_docs])
                        summary, topics = self.extract_content_summary(combined_content)
                        self.sources_metadata[i]['content_summary'] = str(summary)
                        self.sources_metadata[i]['topics_covered'] = str(topics)

                    all_documents.extend(processed_docs)
                    print(f"{len(processed_docs)} docs from {url} loaded")
                else:
                    print(f"can't load {url}")

            except Exception as e:
                print(f"error loading {url}: {str(e)}")
                continue

        print(f"all documents loaded: {len(all_documents)} in Markdown")
        return all_documents

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        if not documents:
            raise ValueError("no docs for vec")

        filtered_docs = [doc for doc in documents if len(doc.page_content.strip()) > 100]

        if not filtered_docs:
            raise ValueError("no long docs")

        print(f"Creating vector store from {len(filtered_docs)} docs...")

        vector_store = Chroma(
            collection_name="article_sources_md",
            embedding_function=self.embeddings,
        )

        # vector_store.add_documents(filter_complex_metadata(filtered_docs))
        batch_size = 5
        for i in range(0, len(filtered_docs), batch_size):
            batch = filtered_docs[i:i+batch_size]
            vector_store.add_documents(batch)
            print(f"{min(i+batch_size, len(filtered_docs))}/{len(filtered_docs)} in batches")

        print("Successfully created")
        return vector_store

    def prepare_context_with_sources(self, retrieved_docs: List[Document]) -> str:
        context_parts = []

        for doc in retrieved_docs:
            source_num = doc.metadata.get('source_number', 'Unknown')
            source_title = doc.metadata.get('source_title', 'Untitled')
            source_domain = doc.metadata.get('source_domain', '')

            source_metadata = self.sources_metadata.get(source_num, {})
            topics = source_metadata.get('topics_covered', [])
            topics_str = ', '.join(topics[:3]) if topics else 'General information'

            context_parts.append(f"""
=== ИСТОЧНИК {source_num}: {source_title} ===
Домен: {source_domain}
Основные темы: {topics_str}
Контент:
{doc.page_content}
""")

        return "\n".join(context_parts)

    def generate_article(self, query: str, max_retrieved_docs: int = 12) -> str:
        print("AI works...")

        urls = self.search_and_collect_urls(query, num_results=15)

        if not urls:
            return "No docs for article"
        documents = self.load_and_process_documents(urls)
        if not documents:
            return "No docs loaded"
        vector_store = self.create_vector_store(documents)
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": max_retrieved_docs, "fetch_k": max_retrieved_docs * 2}
        )
        print("Get relevant docs...")
        retrieved_docs = retriever.get_relevant_documents(query)
        print(f"{len(retrieved_docs)} found")

        context_with_sources = self.prepare_context_with_sources(retrieved_docs)
        print("LLM works...")
        chain = self.prompt | self.model
        response = chain.invoke({
            "question": query,
            "context_with_sources": context_with_sources
        })
        article_with_sources = self.add_enhanced_sources_list(response)
        return article_with_sources

    def add_enhanced_sources_list(self, article_text: str) -> str:
        sources_list = "\n\n## References\n\n"

        for source_num, metadata in self.sources_metadata.items():
            if source_num <= len(self.sources_metadata):
                topics_str = ', '.join(metadata.get('topics_covered', [])[:3])

                sources_list += f"{source_num}. **{metadata['title']}**\n"
                sources_list += f"   - URL: {metadata['url']}\n"
                sources_list += f"   - Домен: {metadata.get('domain', '')}\n"

                # if topics_str:
                #    sources_list += f"   - Основные темы: {topics_str}\n"

                # if metadata.get('content_summary'):
                #    sources_list += f"   - Краткое содержание: {metadata['content_summary']}\n"

                sources_list += "\n"

        return article_text + sources_list

if __name__ == "__main__":
    generator = EnhancedRAGArticleGenerator(
        model_name="qwen3:30b",
        searx_host="http://127.0.0.1:8080"
    )
    article_query = "Write a comprehensive article about integrating LangChain, Ollama, and SearXNG for building RAG applications with local LLMs. Include technical implementation details, advantages, challenges, and real-world applications."
    try:
        article = generator.generate_article(
            query=article_query,
            max_retrieved_docs=15
        )

        print("\n" + "="*80)
        print("="*80 + "\n")
        print(article)

        with open('enhanced_article.md', 'w', encoding='utf-8') as f:
            f.write(article)
        print(f"\nSaved to enhanced_article.md")

    except Exception as e:
        print(f"Error generating article: {str(e)}")
        import traceback
        traceback.print_exc()
