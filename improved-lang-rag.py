import pprint
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.utilities import SearxSearchWrapper
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader

template = """Context: {context}

Question: {question}

Answer: Based on the context above, let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

# model = OllamaLLM(model="llama3.1:8b")
model = OllamaLLM(model="owl/t-lite:instruct")
# model = OllamaLLM(model="qwen3:30b")
# model = OllamaLLM(model="deepseek-r1:32b")

search = SearxSearchWrapper(searx_host="http://127.0.0.1:8080")

results = search.results(
    "Zen Buddhism",
    num_results=250,
    categories="science",
    language="en"
)
# time_range="year",

urls = []
for result in results:
    ll = result['link']
    if ll not in urls and ll is not None and "budd" in ll:
        urls.append(ll)

print("URLs:")
pprint.pp(urls)

if urls:
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    vector_store = Chroma(
        collection_name="my_collection",
        embedding_function=embeddings,
    )

    for url in urls:
        loader = RecursiveUrlLoader(
            url,
            max_depth=2,
            timeout=10,
            check_response_status=True,
            continue_on_failure=True,
            prevent_outside=True,
        )
        data = loader.load()
        if data is not None and len(data) > 0:
            vector_store.add_documents(data)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever
    )
    response = qa_chain.invoke({"query": "resume the context and write an abstract about 7000 characters long and include a reference to the paper you are writing about"})
    print(response['result'])
else:
    print("No URLs")
