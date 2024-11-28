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

class DataAnalysisSystem:
    def __init__(self, api_key: str):
        """
        데이터 분석 시스템 초기화
        :param api_key: OpenAI API 키
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
        openai.api_key = api_key

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
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # Step 5: 질의 수행
        #query = "2024년 8월에 신청 건수가 가장 많은 지역은 어디인가요?"
        query = "이 데이터는 소유권이전등기 지역별 신청현황 데이터인데 이 데이터에 대해 분석해줘"

        result = qa_chain.invoke(query)

        print("질의:", query)
        print("응답:", result)
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()