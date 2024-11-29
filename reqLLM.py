import requests
import json
import openai
import faiss
import numpy as np
import os
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class DataAnalysisSystem:
    def __init__(self, key: str):
        """
        데이터 분석 시스템 초기화
        """
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        index = faiss.IndexFlatIP(768)

        """
        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        """

    def fetch_data(self, url: str, param: str):
        """
        API에서 데이터 가져오기
        :param url: API URL
        :return: JSON 데이터
        """
        response = requests.post(url, param)
        if response.status_code == 200:
            data = response.json()
            return data.get('dataList', [])
        else:
            raise Exception(f"API 요청 실패: {response.status_code}")

    def prepare_text_data(self, data):
        documents = []
        for item in data:
            text = (
                f"Region: {item['admin_regn1_name']}, "
                f"Total Applications: {item['tot']}, "
                f"Date: {item['res_date']}."
            )
            documents.append(text)
        return documents

def main():
    # 시스템 초기화
    system = DataAnalysisSystem("")
    
    try:
        # 데이터 가져오기
        url = "https://data.iros.go.kr/cr/rs/selectRgsCsTab.do"
        payload = {
            "rdata_seq": "0000000015",
            "search_real_cls": "",
            "search_sql": "m01",
            "search_type": "02",
            "search_start_date": "202406",
            "search_end_date": "202408",
            "search_regn_cls": "01",
            "search_tab": "grid",
            "search_regn_name": ""
        }

        data = system.fetch_data(url, payload)
        texts = system.prepare_text_data(data)

        # OpenAI Embeddings 초기화
        embedding = OpenAIEmbeddings()

        # Step 3: FAISS Vector Store 생성
        vector_store = FAISS.from_texts(texts, embedding)

        # Step 4: LLM 연결 및 RAG 체인 생성
        #llm = ChatOpenAI(model="gpt-4", temperature=0)
        llm = ChatOpenAI()

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

        template = """Ansert the question based only on the follwing context:
        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI() 

        retrival_chain = {
            {"context": retriever, "quetstion" : RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        }

        result = retrival_chain.invoke("해당 데이터 분석해줘")
        print(result)
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()