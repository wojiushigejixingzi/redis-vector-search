from danmaku_redis_manager import DanmakuRedisManager
from use_word2vec_model import DanmakuVectorMatcher
import logging

class DanmakuVectorService:
    def __init__(self, model_path, redis_host='localhost', redis_port=6379):
        """
        初始化服务
        :param model_path: Word2Vec模型路径
        :param redis_host: Redis服务器地址
        :param redis_port: Redis服务器端口
        """
        self.matcher = DanmakuVectorMatcher(model_path)
        self.redis_manager = DanmakuRedisManager(
            host=redis_host,
            port=redis_port
        )
        
    def process_new_text(self, text: str, metadata: dict = None) -> bool:
        """
        处理新文本：生成向量并存储到Redis
        """
        # 生成文本的向量表示
        vector = self.matcher.get_text_vector(text)
        
        # 存储到Redis
        return self.redis_manager.store_vector(text, vector, metadata)
        
    def find_similar_texts(self, query_text: str, 
                          similarity_threshold: float = 0.5,
                          max_results: int = 10):
        """
        查找与输入文本相似的已存储文本
        """
        # 生成查询文本的向量
        query_vector = self.matcher.get_text_vector(query_text)
        
        # 在Redis中查找相似文本
        return self.redis_manager.find_similar_texts(
            query_vector,
            similarity_threshold,
            max_results
        )

def main():
    # 初始化服务
    service = DanmakuVectorService(
        model_path="models/tencent_chinese_word2vec.bin",
        redis_host="localhost",
        redis_port=6379
    )
    
    # 示例：存储一些测试文本
    test_texts = [
        "主播真厉害",
        "主播操作太秀了",
        "这个游戏好难",
        "主播玩得真棒",
        "今天天气真好"
    ]

    # 存储测试文本
    for text in test_texts:
        service.process_new_text(text)
        
    # 测试相似文本查找
    query = "主播好厉害"
    similar_texts = service.find_similar_texts(
        query,
        similarity_threshold=0.8,
        max_results=10
    )
    
    print(f"\n与'{query}'相似的文本:")
    print(f"\n 相似文本长度{len(similar_texts)}")
    for text, score in similar_texts:
        print(f"  {text}: {score:.4f}")

if __name__ == "__main__":
    main() 