from gensim.models import KeyedVectors
import jieba
import numpy as np
from pathlib import Path
import logging

class DanmakuVectorMatcher:
    def __init__(self, model_path):
        """
        初始化匹配器
        :param model_path: 训练好的模型路径
        """
        # 加载模型
        self.model = KeyedVectors.load_word2vec_format(str(model_path), binary=True)
        self.vector_size = self.model.vector_size
        
        # 配置日志
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=logging.INFO
        )
        
    def get_text_vector(self, text):
        """
        获取文本的向量表示（词向量的平均值）
        """
        words = jieba.lcut(text)
        vectors = []
        
        for word in words:
            if word in self.model:
                vectors.append(self.model[word])
                
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.vector_size)
    
    def find_similar_words(self, word, topn=10):
        """
        查找与给定词最相似的词
        """
        if word not in self.model:
            logging.warning(f"词语 '{word}' 不在词汇表中")
            return []
            
        return self.model.most_similar(word, topn=topn)
    
    def calculate_similarity(self, text1, text2):
        """
        计算两段文本的相似度
        """
        vec1 = self.get_text_vector(text1)
        vec2 = self.get_text_vector(text2)
        
        # 计算余弦相似度
        if np.any(vec1) and np.any(vec2):
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
        return 0.0
    
    def find_similar_texts(self, query_text, texts, topn=5):
        """
        在文本列表中找出与查询文本最相似的几个
        """
        query_vector = self.get_text_vector(query_text)
        
        # 计算所有文本的向量
        text_vectors = [(text, self.get_text_vector(text)) for text in texts]
        
        # 计算相似度并排序
        similarities = []
        for text, vector in text_vectors:
            if np.any(vector):
                sim = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                similarities.append((text, float(sim)))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

def main():
    # 使用下载好的模型
    model_path = "models/tencent_chinese_word2vec.bin"
    matcher = DanmakuVectorMatcher(model_path)
    
    # 示例1：查找相似词
    test_words = ['盘子', '真棒', '跳舞']
    print("\n=== 相似词查找 ===")
    for word in test_words:
        print(f"\n与 '{word}' 相似的词:")
        similar_words = matcher.find_similar_words(word, topn=5)
        for similar_word, score in similar_words:
            print(f"  {similar_word}: {score:.4f}")
    
    # 示例2：计算文本相似度
    print("\n=== 文本相似度计算 ===")
    text1 = "主播玩得真好"
    text2 = "主播操作太厉害了"
    text3 = "这游戏好难"
    
    sim1 = matcher.calculate_similarity(text1, text2)
    sim2 = matcher.calculate_similarity(text1, text3)
    
    print(f"文本1: {text1}")
    print(f"文本2: {text2}")
    print(f"文本3: {text3}")
    print(f"文本1和文本2的相似度: {sim1:.4f}")
    print(f"文本1和文本3的相似度: {sim2:.4f}")
    
    # 示例3：在文本列表中查找相似文本
    print("\n=== 相似文本查找 ===")
    query = "外国"
    texts = [
        "主播操作太秀了",
        "这个游戏好难",
        "主播玩得真棒",
        "今天天气真好",
        "主播太强了",
        "你好"
    ]
    
    similar_texts = matcher.find_similar_texts(query, texts, topn=3)
    print(f"\n与 '{query}' 最相似的文本:")
    for text, score in similar_texts:
        print(f"  {text}: {score:.4f}")

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('lili666/text2vec-word2vec-tencent-chinese')

if __name__ == "__main__":
    main() 
    