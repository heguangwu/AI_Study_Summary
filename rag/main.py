import os
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from openai import OpenAI


class RAG:
    def __init__(self):
        # embedding 和 rerank 模型
        self.embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')
        self.rerank_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        # 使用 智谱BigModel开放平台 大模型
        self.llm_model = OpenAI(api_key=RAG.get_api_key(),base_url="https://open.bigmodel.cn/api/paas/v4/")
        # 创建一个内存向量数据库
        self.chromadb_ = chromadb.EphemeralClient().get_or_create_collection(name='default')

    # 读取文件分割成多个块计算块的向量值并保存到向量数据库
    def load_file_to_chromadb(self, file: str) -> None:
        chunks = self.__split_chunks(file)
        embeddings = [self.__embed_chunk(chunk) for chunk in chunks]
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.chromadb_.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[str(i)]
            )

    def query_and_answer(self, query: str) -> str:
        retrieve_content = self.__retrieve(query)
        rerank_content = self.__rerank(query, retrieve_content)
        return self.__generate(query, rerank_content)

    def __retrieve(self, query:str, topk:int=5) -> List[str]:
        query_embedding = self.__embed_chunk(query)
        results = self.chromadb_.query(
            query_embeddings=[query_embedding],
            n_results=topk
        )['documents']
        if results:
            return results[0]
        else:
            raise ValueError(f"未找到 {query} 相似的数据")

    def __embed_chunk(self, chunk: str) -> List[float]:
        return self.embedding_model.encode(chunk, normalize_embeddings=True).tolist()

    def __rerank(self, query: str, retrieve_chunks: List[str], topk:int=3) -> List[str]:
        pairs = [(query, chunk) for chunk in retrieve_chunks]
        scores = self.rerank_model.predict(pairs)
        scored_chunks = list(zip(retrieve_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored_chunks][:topk]

    def __generate(self, query: str, chunks: List[str]) -> str:
        prompt = f"""用户问题：{query}

        相关片段：{"\n\n".join(chunks)}

        请基于上述内容作答，不要编造信息。
        """
        response = self.llm_model.chat.completions.create(
            model="glm-4-flash",  # 模型名称（智谱的 GLM-4.5-Flash 对应此名称）
            messages=[
                {"role": "system", "content": "你是一位电力行业的专家，请根据用户的问题和下列片段生成准确的问答。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 控制输出随机性（0-1，值越高越随机）
            max_tokens=1024  # 最大生成 tokens 数
        )
        # 提取生成的内容
        if response.choices[0].message.content:
            return response.choices[0].message.content 
        return '模型返回错误，暂时无法提供服务'

    # 读取文件并使用换行符分割成多个块
    def __split_chunks(self, file: str) -> List[str]:
        with open(file, encoding="utf-8") as f:
            content = f.read()
        return [chunk for chunk in content.split("\n\n")]

    @staticmethod
    def get_api_key() -> str:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("未找到 OPENAI_API_KEY 环境变量，请在 .env 文件中配置")
        return api_key


def main():
    rag = RAG()
    rag.load_file_to_chromadb('./rag_doc.md')
    query = '跨步电压是什么'
    result = rag.query_and_answer(query)
    print(f"问题：{query}")
    print(f'{result}')

if __name__ == "__main__":
    main()
