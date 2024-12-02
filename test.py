import requests
import faiss
import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

class NewsArticleRAG:
    def __init__(self, openai_api_key):
        """
        Initialize the NewsArticleRAG system
        
        :param openai_api_key: OpenAI API 키
        """
        # OpenAI 임베딩 및 언어 모델 초기화
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.llm = ChatOpenAI(
            api_key=openai_api_key, 
            model='gpt-3.5-turbo', 
            temperature=0.7
        )

    def fetch_json_data(self, url, param):
        """
        HTTP 요청으로 JSON 데이터 가져오기
        
        :param url: JSON 데이터를 제공하는 API 엔드포인트
        :return: JSON 데이터
        """
        response = requests.post(url, param)
        if response.status_code == 200:
            data = response.json()
            return data.get('dataList', [])
        else:
            raise Exception(f"API 요청 실패: {response.status_code}")

    def prepare_faiss_database(self, json_data):
        """
        JSON 데이터를 FAISS 데이터베이스에 저장
        
        :param json_data: 처리할 JSON 데이터
        :return: FAISS 벡터 스토어
        """
        # JSON 데이터 전처리 및 텍스트 추출
#        texts = [
#            f"{item.get('res_date', '')} {item.get('tot', '')} {item.get('admin_regn1_name', '')}" 
#            for item in json_data
#        ]

        texts = []
        for item in json_data:
#            text = (
#                f"Region: {item['admin_regn1_name']}, "
#                f"Total Applications: {item['tot']}, "
#                f"Date: {item['res_date']}."
#            )
            text_content = json.dumps(item, ensure_ascii=False)
            texts.append(text_content)

        # FAISS 벡터 스토어 생성
        vector_store = FAISS.from_texts(texts, self.embeddings)
        return vector_store

    def generate_news_article(self, vector_store, query):
        """
        RAG를 활용하여 기사 생성
        
        :param vector_store: FAISS 벡터 스토어
        :param query: 기사 주제 쿼리
        :return: 생성된 기사
        """
        # 관련 문서 검색
        relevant_docs = vector_store.similarity_search(query, k=3)
        
        # 기사 작성 프롬프트 템플릿
        prompt_template = PromptTemplate(
            input_variables=['context', 'query'],
            template="""
            당신은 전문 뉴스 기자입니다. 다음 맥락을 바탕으로 {query} 주제에 대한 심층적이고 객관적인 기사를 작성하세요.

            맥락:
            {context}

            기사는 다음 구조를 따르세요:
            1. 매력적인 헤드라인
            2. 리드(소개) 문단
            3. 주요 내용 (3-4개 문단)
            4. 결론 및 전망

            기사 스타일: 객관적이고 전문적이며, 사실에 기반한 보도
            """
        )

        # 컨텍스트 준비
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # LLM 체인 생성 및 기사 생성
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        article = chain.run(context=context, query=query)

        return article

    def run_news_generation_pipeline(self, url, payload, query):
        """
        전체 뉴스 생성 파이프라인 실행
        
        :param url: JSON 데이터 URL
        :param query: 기사 주제 쿼리
        """
        # JSON 데이터 가져오기
        json_data = self.fetch_json_data(url, payload)
        if not json_data:
            print("데이터를 가져오지 못했습니다.")
            return

        # FAISS 데이터베이스 준비
        vector_store = self.prepare_faiss_database(json_data)

        # 기사 생성
        news_article = self.generate_news_article(vector_store, query)

        # 기사 출력
        print("📰 생성된 뉴스 기사:\n")
        print(news_article)

# 사용 예시
if __name__ == "__main__":
    # OpenAI API 키 설정 필요
    
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    # 예시 URL과 쿼리
    url = "https://data.iros.go.kr/cr/rs/selectRgsCsTab.do"
    payload = {
        "rdata_seq": "0000000015",
        "search_real_cls": "",
        "search_sql": "m01",
        "search_type": "02",
        "search_start_date": "202406",
        "search_end_date": "202411",
        "search_regn_cls": "01",
        "search_tab": "grid",
        "search_regn_name": ""
    }
    query = '지역별 소유권 이전등기 신청 현황'

    news_rag = NewsArticleRAG(openai_api_key)
    news_rag.run_news_generation_pipeline(url, payload, query)