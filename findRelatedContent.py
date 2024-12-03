import requests
import faiss
import json
import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

class FindRelatedContents:
    def __init__(self, openai_api_key, model='gpt-4-turbo'):
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.llm = ChatOpenAI(
            api_key=openai_api_key, 
            model=model, 
            temperature=0.6  # 창의성과 사실성 사이 균형
        )

        # FAISS 인덱스 초기화
        self.index = faiss.IndexFlatL2(embedding_dim)

        # 메타데이터 및 매핑 관리를 위한 리스트
        self.contents_metadata = []
        self.embeddings_list = []

    def add_content_metadata(self, json_meta):
        metadata = json_meta.loads(json_meta)

        # 임베딩 생성
        embedding = np.array(self.embeddings.embed_query(f"{metadata['rdata_name']} {metadata['rdata_dsc']}"), dtype='float32').reshape(1, -1)


    def fetch_contentList(self, url, payload):
        response = requests.post(url, payload)
        if response.status_code == 200:
            data = response.json()
            json_data = data.get('dataList', [])
            texts = []
            for item in json_data:
                text = (
                    f"컨텐츠 번호: {item['rdata_seq']}, "
                    f"컨텐츠 명: {item['rdata_name']}, "
                    f"컨텐츠 설명: {item['rdata_dsc']}, "
                    f"카테고리1: {item['sbjt_per_ctg1_name']}, "
                    f"카테고리2: {item['sbjt_per_ctg2_name']}, "
                    f"카테고리3: {item['sbjt_per_ctg3_name']}, "
                    f"카테고리4: {item['sbjt_per_ctg4_name']}."
                )
                texts.append(text)
            # FAISS 벡터 스토어 생성
            vector_store = FAISS.from_texts(texts, self.embeddings)
            return vector_store
        else:
            raise Exception(f"API 요청 실패: {response.status_code}")

if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    compare = FindRelatedContents(openai_api_key)
    
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

    vector_store = compare.fetch_contentList(url, payload)
    query = '지역별 부동산 소유현황'
    relevant_contents = vector_store.similarity_search(query, k=3)

    print(len(vector_store.index_to_docstore_id))
    for content in relevant_contents:
        print(content)


    

