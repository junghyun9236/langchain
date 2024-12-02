import requests
import json
import textwrap
import os
from dotenv import load_dotenv
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

class EnhancedNewsArticleRAG:
    def __init__(self, openai_api_key, model='gpt-4-turbo'):
        """
        고급 뉴스 기사 생성 시스템 초기화
        
        :param openai_api_key: OpenAI API 키
        :param model: 사용할 언어 모델
        """
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.llm = ChatOpenAI(
            api_key=openai_api_key, 
            model=model, 
            temperature=0.6  # 창의성과 사실성 사이 균형
        )

    def preprocess_json_data(self, json_data: List[Dict]) -> List[str]:
        """
        JSON 데이터 전처리 및 고급 텍스트 추출
        
        :param json_data: 원본 JSON 데이터
        :return: 전처리된 텍스트 리스트
        """
        processed_texts = []
        for item in json_data:
            # 다양한 키에서 정보 추출
            text_parts = []
            
            # 주요 정보 추출 (예시 - 실제 JSON 구조에 맞게 조정 필요)
            if 'res_date' in item:
                text_parts.append(f"날짜: {item['res_date']}")
            
            if 'admin_regn1_name' in item:
                text_parts.append(f"행정구역명: {item['admin_regn1_name']}")
            
            if 'tot' in item:
                text_parts.append(f"총 신청건수: {item['tot']}")
            
            processed_text = " ".join(text_parts)
            processed_texts.append(processed_text)
            print(processed_texts)
        return processed_texts

    def prepare_faiss_database(self, json_data: List[Dict]):
        """
        FAISS 데이터베이스 고급 준비
        
        :param json_data: 처리할 JSON 데이터
        :return: FAISS 벡터 스토어
        """
        processed_texts = self.preprocess_json_data(json_data)
        vector_store = FAISS.from_texts(processed_texts, self.embeddings)
        return vector_store

    def generate_news_article(self, vector_store, query: str):
        """
        고급 RAG 기사 생성
        
        :param vector_store: FAISS 벡터 스토어
        :param query: 기사 주제 쿼리
        :return: 생성된 기사
        """
        # 관련 문서 더 많이 검색
        relevant_docs = vector_store.similarity_search(query, k=5)
        
        # 고급 프롬프트 템플릿
        advanced_prompt_template = PromptTemplate(
            input_variables=['context', 'query'],
            template=textwrap.dedent("""
            당신은 세계적인 수준의 전문 저널리스트입니다. 다음 맥락을 바탕으로 {query} 주제에 대한 심도 있고 균형 잡힌 기사를 작성하세요.

            핵심 가이드라인:
            1. 객관적이고 중립적인 시각 유지
            2. 다양한 관점 고려
            3. 사실 기반 보도
            4. 깊이 있는 분석 제공

            제공된 맥락:
            {context}

            기사 구조:
            1. 강력하고 매력적인 헤드라인
            2. 상황의 핵심을 요약하는 리드 문단
            3. 심층 분석 (3-4개 문단)
               - 배경 설명
               - 현재 상황 분석
               - 잠재적 영향 및 미래 전망
            4. 균형 잡힌 결론
            5. 추가 맥락이나 전문가 인용 포함

            스타일 노트:
            - 전문적이고 학술적인 톤
            - 명확하고 간결한 언어 사용
            - 복잡한 개념을 이해하기 쉽게 설명
            - 가능한 경우 통계, 연구, 전문가 의견 인용
            """)
        )

        # 컨텍스트 준비
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # 기사 생성 체인
        article_chain = LLMChain(
            llm=self.llm, 
            prompt=advanced_prompt_template
        )

        # 기사 생성
        article = article_chain.run(context=context, query=query)

        return article

    def run_news_generation_pipeline(self, url: str, paylaod: str, query: str):
        """
        뉴스 생성 파이프라인 실행
        
        :param url: JSON 데이터 URL
        :param query: 기사 주제 쿼리
        """
        # JSON 데이터 가져오기
        try: 
            response = requests.post(url, payload)
            data = response.json()
            json_data = data.get('dataList', [])
        except requests.RequestException as e:
            print(f"데이터 fetching 오류: {e}")
            return

        # FAISS 데이터베이스 준비
        vector_store = self.prepare_faiss_database(json_data)

        # 고급 기사 생성
        news_article = self.generate_news_article(vector_store, query)

        # 기사 출력 및 포매팅
        print("📰 생성된 심층 뉴스 기사:\n")
        print(news_article)

# 사용 예시
if __name__ == "__main__":
    
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    # 실제 사용 시 적절한 URL로 대체
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
    article = '지역별 소유권 이전등기 신청 현황'

    news_rag = EnhancedNewsArticleRAG(openai_api_key)
    news_rag.run_news_generation_pipeline(url, payload, article)