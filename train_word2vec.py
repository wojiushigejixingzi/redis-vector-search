import jieba
import re
from gensim.models import Word2Vec
from pathlib import Path
import logging
import json

class DanmakuWord2VecTrainer:
    def __init__(self, vector_size=300, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.stop_words = self._load_stop_words()
        
        # 配置日志
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=logging.INFO
        )
    
    def _load_stop_words(self):
        """加载停用词"""
        # 这里可以扩展为从文件加载更多停用词
        return {'的', '了', '是', '在', '吗', '啊', '呢', '吧', '哦', '啦', 
                '呀', '哈', '嗯', '这', '那', '就', '都', '和', '与', '也'}
    
    def clean_text(self, text):
        """清理文本"""
        # 去除标点符号和特殊字符
        text = re.sub(r'[^\w\s]', '', text)
        # 分词
        words = jieba.cut(text)
        # 去除停用词
        words = [w for w in words if w not in self.stop_words]
        return words
    
    def prepare_sentences(self, danmaku_file):
        """准备训练数据"""
        sentences = []
        
        # 从JSON文件加载弹幕数据
        with open(danmaku_file, 'r', encoding='utf-8') as f:
            danmaku_data = json.load(f)
        
        # 处理每条弹幕
        for danmaku in danmaku_data:
            if isinstance(danmaku, str):
                text = danmaku
            elif isinstance(danmaku, dict):
                text = danmaku.get('text', '')  # 假设JSON中弹幕文本的键为'text'
            else:
                continue
                
            words = self.clean_text(text)
            if words:  # 只添加非空的分词结果
                sentences.append(words)
        
        return sentences
    
    def train(self, danmaku_file, output_path):
        """训练Word2Vec模型"""
        # 准备训练数据
        sentences = self.prepare_sentences(danmaku_file)
        
        if not sentences:
            raise ValueError("没有可用的训练数据！")
        
        logging.info(f"开始训练Word2Vec模型，共有{len(sentences)}条训练数据")
        
        # 训练模型
        model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,  # 使用4个线程训练
            sg=1,      # 使用skip-gram模型
            epochs=10  # 训练轮数
        )
        
        # 保存模型
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output_path))
        
        logging.info(f"模型训练完成，已保存到: {output_path}")
        
        return model

def main():
    # 示例弹幕数据，实际使用时替换为你的数据文件路径
    sample_danmaku = [
        "这个主播太搞笑了",
        "主播玩得真厉害",
        "这波操作秀啊",
        "主播好厉害",
        "这个游戏真好玩",
        "666666",
        "主播带带我",
        "这游戏太难了",
        "主播操作真细腻",
        "这波走位很帅"
    ]
    
    # 保存示例数据到JSON文件
    
    with open('sample_danmaku.json', 'w', encoding='utf-8') as f:
        json.dump(sample_danmaku, f, ensure_ascii=False, indent=2)
    
    # 训练模型
    trainer = DanmakuWord2VecTrainer(
        vector_size=300,  # 词向量维度
        window=5,         # 上下文窗口大小
        min_count=1       # 最小词频
    )
    
    model = trainer.train(
        danmaku_file='sample_danmaku.json',
        output_path='models/danmaku_word2vec.model'
    )
    
    # 测试模型
    test_words = ['主播', '游戏', '操作']
    for word in test_words:
        if word in model.wv:
            print(f"\n与'{word}'最相似的词:")
            similar_words = model.wv.most_similar(word)
            for similar_word, score in similar_words[:5]:
                print(f"{similar_word}: {score:.4f}")

if __name__ == "__main__":
    main() 