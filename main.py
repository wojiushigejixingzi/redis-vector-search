from danmaku_processor import DanmakuProcessor
from redis_manager import RedisManager
import time

class DanmakuSearchSystem:
    def __init__(self, model_path):
        self.processor = DanmakuProcessor(model_path)
        self.redis_manager = RedisManager()
        self.danmaku_counter = 0
    
    def add_danmaku(self, text):
        """添加新弹幕"""
        embedding = self.processor.get_embedding(text)
        if embedding is not None:
            self.danmaku_counter += 1
            success = self.redis_manager.store_danmaku(
                self.danmaku_counter, text, embedding
            )
            return success
        return False
    
    def search_similar_danmaku(self, query_text, top_k=5):
        """搜索相似弹幕"""
        query_embedding = self.processor.get_embedding(query_text)
        if query_embedding is None:
            return []
        
        return self.redis_manager.search_similar(query_embedding, top_k)

def main():
    # 初始化系统
    system = DanmakuSearchSystem("path_to_your_word2vec_model")
    
    # 添加一些示例弹幕
    sample_danmaku = [
        "这个主播太搞笑了",
        "主播玩得真厉害",
        "这波操作秀啊",
        "主播好厉害",
        "这个游戏真好玩"
    ]
    
    for danmaku in sample_danmaku:
        system.add_danmaku(danmaku)
        print(f"添加弹幕: {danmaku}")
    
    # 测试搜索
    query = "主播真厉害"
    print(f"\n搜索与'{query}'相似的弹幕:")
    results = system.search_similar_danmaku(query)
    for text, score in results:
        print(f"弹幕: {text}, 相似度: {score:.4f}")

if __name__ == "__main__":
    main() 