import redis
import numpy as np
import json
from typing import List, Tuple, Optional
import logging
from datetime import datetime

class DanmakuRedisManager:
    def __init__(self, host='localhost', port=6379, db=0):
        """初始化Redis连接"""
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True  # 自动解码响应
        )
        self.vector_key_prefix = "danmaku:vector:"
        self.metadata_key_prefix = "danmaku:metadata:"
        
        # 配置日志
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=logging.INFO
        )

    def store_vector(self, text: str, vector: np.ndarray, metadata: dict = None) -> bool:
        """
        存储文本及其向量到Redis
        :param text: 原始文本
        :param vector: 向量表示
        :param metadata: 额外的元数据（如时间戳、用户ID等）
        """
        try:
            # 生成唯一key
            vector_key = f"{self.vector_key_prefix}{text}"
            metadata_key = f"{self.metadata_key_prefix}{text}"
            
            # 准备元数据
            metadata = metadata or {}
            metadata.update({
                'text': text,
                'timestamp': datetime.now().isoformat(),
                'vector_key': vector_key
            })
            
            # 存储向量和元数据
            self.redis_client.set(
                vector_key,
                json.dumps(vector.tolist())  # numpy数组转换为JSON
            )
            self.redis_client.set(
                metadata_key,
                json.dumps(metadata)
            )
            
            logging.info(f"成功存储文本: {text} vector_key: {vector_key}")
            return True
            
        except Exception as e:
            logging.error(f"存储向量时出错: {str(e)}")
            return False

    def get_vector(self, text: str) -> Optional[np.ndarray]:
        """从Redis获取文本的向量表示"""
        try:
            vector_key = f"{self.vector_key_prefix}{text}"
            vector_data = self.redis_client.get(vector_key)
            
            if vector_data:
                return np.array(json.loads(vector_data))
            return None
            
        except Exception as e:
            logging.error(f"获取向量时出错: {str(e)}")
            return None

    def find_similar_texts(self, query_vector: np.ndarray, 
                          similarity_threshold: float = 0.5,
                          max_results: int = 10) -> List[Tuple[str, float]]:
        """
        查找与给定向量相似的文本
        :param query_vector: 查询向量
        :param similarity_threshold: 相似度阈值
        :param max_results: 最大返回结果数
        :return: [(text, similarity_score), ...]
        """
        try:
            results = []
            # 获取所有向量keys
            vector_keys = self.redis_client.keys(f"{self.vector_key_prefix}*")
            
            for key in vector_keys:
                # 获取向量数据
                vector_data = self.redis_client.get(key)
                if vector_data:
                    vector = np.array(json.loads(vector_data))
                    
                    # 计算余弦相似度
                    similarity = np.dot(query_vector, vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(vector)
                    )
                    
                    # 如果相似度超过阈值，添加到结果中
                    if similarity >= similarity_threshold:
                        text = key.replace(self.vector_key_prefix, '')
                        results.append((text, float(similarity)))
            
            # 按相似度降序排序并限制结果数量
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:max_results]
            
        except Exception as e:
            logging.error(f"查找相似文本时出错: {str(e)}")
            return []

    def clear_all_vectors(self):
        """清除所有存储的向量和元数据"""
        try:
            vector_keys = self.redis_client.keys(f"{self.vector_key_prefix}*")
            metadata_keys = self.redis_client.keys(f"{self.metadata_key_prefix}*")
            
            if vector_keys:
                self.redis_client.delete(*vector_keys)
            if metadata_keys:
                self.redis_client.delete(*metadata_keys)
                
            logging.info("已清除所有向量和元数据")
            
        except Exception as e:
            logging.error(f"清除数据时出错: {str(e)}") 