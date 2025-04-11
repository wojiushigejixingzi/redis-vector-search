import jieba
import re
from gensim.models import Word2Vec
import numpy as np

class DanmakuProcessor:
    def __init__(self, model_path=None):
        # 加载预训练模型或训练新模型
        self.model = Word2Vec.load(model_path) if model_path else None
        # 停用词列表
        self.stop_words = self._load_stop_words()
    
    def _load_stop_words(self):
        # 这里可以从文件加载停用词，这里简单演示几个
        return {'的', '了', '是', '在', '吗', '啊', '呢', '吧'}
    
    def clean_text(self, text):
        """清理文本"""
        # 去除标点符号和特殊字符
        text = re.sub(r'[^\w\s]', '', text)
        # 分词
        words = jieba.cut(text)
        # 去除停用词
        words = [w for w in words if w not in self.stop_words]
        return words
    
    def get_embedding(self, text):
        """获取文本的embedding"""
        words = self.clean_text(text)
        word_vectors = []
        for word in words:
            try:
                vector = self.model.wv[word]
                word_vectors.append(vector)
            except KeyError:
                continue
        
        if not word_vectors:
            return None
        
        # 取平均值作为整句的向量表示
        return np.mean(word_vectors, axis=0) 