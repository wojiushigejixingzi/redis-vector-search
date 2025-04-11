import redis
from redis.commands.search.field import VectorField
from redis.commands.search.query import Query
import numpy as np

class RedisManager:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.vector_dim = 300  # Word2Vec默认维度
        
        # 创建向量索引
        self._create_index()
    
    def _create_index(self):
        """创建向量索引"""
        try:
            # 创建包含向量字段的索引
            schema = (
                VectorField("embedding", "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": self.vector_dim,
                    "DISTANCE_METRIC": "COSINE"
                }),
            )
            
            self.redis_client.ft("danmaku_idx").create_index(schema)
        except Exception as e:
            print(f"Index might already exist: {e}")
    
    def store_danmaku(self, danmaku_id, text, embedding):
        """存储弹幕和其embedding"""
        if embedding is None:
            return False
            
        # 存储弹幕文本
        self.redis_client.hset(f"danmaku:{danmaku_id}", 
                             mapping={
                                 "text": text,
                                 "embedding": embedding.astype(np.float32).tobytes()
                             })
        return True
    
    def search_similar(self, query_embedding, top_k=5):
        """搜索相似弹幕"""
        query = (
            Query(f"*=>[KNN {top_k} @embedding $query_vec AS score]")
            .sort_by("score")
            .return_fields("text", "score")
            .dialect(2)
        )
        
        query_params = {
            "query_vec": query_embedding.astype(np.float32).tobytes()
        }
        
        # 执行搜索
        results = self.redis_client.ft("danmaku_idx").search(query, query_params)
        
        return [(doc.text, 1 - float(doc.score)) for doc in results.docs] 