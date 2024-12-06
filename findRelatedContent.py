import requests
import faiss
import json
import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class FindRelatedContents:
    def __init__(self, embedding_dim=1536):
        load_dotenv()
        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.chat_model = ChatOpenAI(model="gpt-4-turbo")

        # FAISS 인덱스 초기화
        self.index = faiss.IndexFlatL2(embedding_dim)

        # 메타데이터 및 매핑 관리를 위한 리스트
        self.contents_metadata = []
        self.embeddings_list = []
        self.region_arr = ["서울특별시", "경기도", "부산광역시", "인천광역시", "대전광역시", "대구광역시"]

    def add_content_metadata(self, content_meta):

        # 임베딩 생성
        embedding = np.array(self.embeddings.embed_query(f"{content_meta['rdata_name']} {content_meta['rdata_dsc']}"), dtype='float32').reshape(1, -1)

        # 메타데이터 저장
        metadata = {
            "rdata_seq": content_meta['rdata_seq'],
            "rdata_name": content_meta['rdata_name'],
            "rdata_dsc": content_meta['rdata_dsc'],
            "sbjt_per_ctg1_name": content_meta['sbjt_per_ctg1_name'],
            "sbjt_per_ctg2_name": content_meta['sbjt_per_ctg2_name'],
            "sbjt_per_ctg3_name": content_meta['sbjt_per_ctg3_name'],
            "sbjt_per_ctg4_name": content_meta['sbjt_per_ctg4_name']
        }
        
        # FAISS 인덱스에 추가
        self.index.add(embedding)
        
        # 메타데이터 리스트에 저장
        self.contents_metadata.append(metadata)
        self.embeddings_list.append(embedding)


    def fetch_contentList(self, url, payload):
        response = requests.post(url, payload)
        if response.status_code == 200:
            data = response.json()
            json_data = data.get('dataList', [])
            return json_data
        else:
            raise Exception(f"API 요청 실패: {response.status_code}")

    def find_related_contents(self, query, top_k=3):
        # 쿼리 임베딩 생성
        query_embedding = np.array(self.embeddings.embed_query(query), dtype='float32').reshape(1, -1)
        
        # FAISS 검색 수행
        distances, indices = self.index.search(query_embedding, top_k)

        # 결과 처리
        context = ""
        for dist, idx in zip(distances[0], indices[0]):
            metadata = self.contents_metadata[idx]
            stats = self.search_statistics_data(metadata['rdata_seq'])
            stat_text = ""
            for stat in stats:
                parse_text = ""
                if(stat['admin_regn1_name'] in self.region_arr):
                    for key,value in stat.items():
                        if(not "_cd" in key):
                            parse_text += f"{value} "
                    if(parse_text != "" ): 
                        stat_text += parse_text + "\n"

                #stat_text += json.dumps(stat, ensure_ascii=False) + "\n"
            if(stat_text != ""):
                context += f"컨텐츠명: {metadata['rdata_name']}, 설명: {metadata['rdata_dsc']}, stat: {stat_text}\n"
        return context

    def create_rag_chain(self):
        """
        Langchain RAG 파이프라인 생성
        """
        template = """
        다음 컨텍스트를 기반으로 사용자의 쿼리에 대해 심층 분석을 제공하세요:

        컨텍스트:
        {context}

        사용자 쿼리: {question}

        분석 가이드라인:
        1. 제공된 컨텍스트를 종합적으로 분석하세요.
        2. 각 컨텐츠의 핵심 내용과 연관성을 설명하세요.
        3. 전문적이고 심층적인 분석 결과를 제시하세요.
        4. 제공된 컨텐츠들을 이용해서 새로운 컨텐츠를 생성하세요.
        """

        prompt = ChatPromptTemplate.from_template(template)

        # RAG 체인 생성
        rag_chain = (
            {"context": RunnablePassthrough(), 
             "question": RunnablePassthrough(), 
             "statistics_summary": RunnablePassthrough()}
            | prompt
            | self.chat_model
            | StrOutputParser()
        )

        return rag_chain

    def search_statistics_data(self, rdata_seq):  
        try:
            # 데이터 가져오기
            url = "https://data.iros.go.kr/cr/rs/selectRgsCsTab.do"
            payload = {
                "rdata_seq": rdata_seq,
                "search_real_cls": "",
                "search_sql": "m01",
                "search_type": "02",
                "search_start_date": "202409",
                "search_end_date": "202410",
                "search_regn_cls": "01",
                "search_tab": "grid",
                "search_regn_name": ""
            }
            response = requests.post(url, payload)
            if response.status_code == 200:
                data = response.json()
                return data.get('dataList', [])
        except requests.RequestException as e:
            print(f"데이터 fetching 오류: {e}")
            return

    def analyze_related_contents(self, query):
        """
        관련 컨텐츠 분석 및 RAG 기반 통합 인사이트 생성
        """
        # 관련 컨텐츠 통계 정보조회
        context = self.find_related_contents(query)
        
        # RAG 체인 생성
        rag_chain = self.create_rag_chain()
        
        # 통합 분석 수행
        analysis_result = rag_chain.invoke({
            "context": context, 
            "question": query, 
        })
        
        return analysis_result

if __name__ == "__main__":
    compare = FindRelatedContents()
    
    # 실제 사용 시 적절한 URL로 대체
    url = "https://data.iros.go.kr/cr/rs/selectRgsCsList.do"
    payload = {
        "pageIndex": "1",
        "ris_menu_seq": "0000000022",
        "rdata_seq": "",
        "list_click": "",
        "show_button": "open",
        "sbjt_per_ctg1_cd": "",
        "sbjt_per_ctg2_cd": "",
        "sbjt_per_ctg3_cd": "",
        "sbjt_per_ctg4_cd": "",
        "itrs_info_cls_cd": "026000",
        "search_tab": "",
        "search_datagubun": "A",
        "search_type01_list": "All",
        "search_order_init": "Y",
        "search_word": "",
        "search_type01": "",
        "search_type02": "",
        "search_type03": "",
        "search_type04": "",
        "search_order": "031003",
        "pagePerCount": "300"
    }

    content_metalist = compare.fetch_contentList(url, payload)
    for content_meta in content_metalist:
        compare.add_content_metadata(content_meta)
    query = '외국인 부동산 소유현황'
    #query = '상법법인 파산현황'
    result = compare.analyze_related_contents(query)

    # 결과 출력
    print("\n통합 분석 결과:")
    print(result)
