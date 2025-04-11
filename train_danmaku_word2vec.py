import jieba
import logging
import json
import re
from pathlib import Path
from gensim.models import Word2Vec
from tqdm import tqdm
import psutil
import multiprocessing
from datetime import datetime

class DanmakuProcessor:
    def __init__(self):
        self.stop_words = self._load_stop_words()
        jieba.enable_parallel(multiprocessing.cpu_count())
        
    def _load_stop_words(self):
        """加载停用词"""
        return {'的', '了', '是', '在', '吗', '啊', '呢', '吧', '哦', '啦', 
                '呀', '哈', '嗯', '这', '那', '就', '都', '和', '与', '也',
                '666', '233', '？', '！', '。', '，', '1', '2', '3', '4',
                '你', '我', '他', '她', '它', '有', '好', '来', '去', '说'}

    def clean_text(self, text):
        """清理文本"""
        if not isinstance(text, str) or not text.strip():
            return []
        
        # 去除@用户名
        text = re.sub(r'@[\w\u4e00-\u9fa5]+\s*', '', text)
        
        # 去除特殊字符和表情符号
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        
        # 分词并过滤
        words = [w for w in jieba.cut(text) if w.strip() 
                and w not in self.stop_words 
                and len(w) > 1]  # 只保留长度大于1的词
        
        return words

class DanmakuWord2VecTrainer:
    def __init__(self, processor=None):
        self.processor = processor or DanmakuProcessor()
        self.process = psutil.Process()
        
        # 配置日志
        log_file = f'logs/training_{datetime.now():%Y%m%d_%H%M%S}.log'
        Path('logs').mkdir(exist_ok=True)
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def log_memory_usage(self):
        memory_info = self.process.memory_info()
        logging.info(f"当前内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")

    def prepare_sentences(self, danmaku_file):
        """处理弹幕数据"""
        processed_count = 0
        valid_count = 0
        sentences = []
        
        logging.info("开始读取弹幕数据...")
        try:
            with open(danmaku_file, 'r', encoding='utf-8') as f:
                danmaku_list = json.load(f)  # 直接读取整个JSON数组
                
            logging.info(f"成功读取数据，共 {len(danmaku_list)} 条弹幕")
            
            # 处理每条弹幕
            for danmaku in tqdm(danmaku_list, desc="处理弹幕"):
                processed_count += 1
                try:
                    text = danmaku.get('content', '')
                    words = self.processor.clean_text(text)
                    
                    if words:
                        valid_count += 1
                        sentences.append(words)
                    
                    if processed_count % 100000 == 0:
                        logging.info(f"已处理 {processed_count} 条弹幕，有效数据 {valid_count} 条")
                        self.log_memory_usage()
                        
                except Exception as e:
                    if processed_count % 100000 == 0:
                        logging.warning(f"处理弹幕出错: {str(e)}")
                    continue
                    
        except Exception as e:
            logging.error(f"读取文件出错: {str(e)}")
            raise

        logging.info(f"数据处理完成，共处理 {processed_count} 条弹幕，有效数据 {valid_count} 条")
        return sentences

    def train(self, danmaku_file, output_path):
        """训练Word2Vec模型"""
        logging.info(f"开始训练，使用数据文件: {danmaku_file}")
        self.log_memory_usage()

        # 准备训练数据
        sentences = self.prepare_sentences(danmaku_file)
        logging.info(f"训练数据准备完成，共有 {len(sentences)} 条有效句子")
        
        if not sentences:
            raise ValueError("没有找到有效的训练数据！")

        # 训练模型
        model = Word2Vec(
            sentences=sentences,
            vector_size=200,     # 词向量维度
            window=5,            # 上下文窗口大小
            min_count=10,        # 最小词频
            workers=multiprocessing.cpu_count(),  # 使用所有CPU核心
            sg=1,                # 使用skip-gram模型
            epochs=5,            # 训练轮数
            batch_words=10000    # 批处理大小
        )

        # 保存模型
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output_path))
        
        logging.info(f"模型训练完成，已保存到: {output_path}")
        self.log_memory_usage()
        
        return model

def main():
    # 设置文件路径
    data_file = 'corpus/pepper_chat_2019_chat_20241231.json'
    model_file = f'models/danmaku_word2vec_{datetime.now():%Y%m%d}.model'
    
    # 创建训练器并开始训练
    trainer = DanmakuWord2VecTrainer()
    model = trainer.train(data_file, model_file)
    
    # 测试模型效果
    test_words = ['主播', '游戏', '666', '厉害']
    logging.info("\n模型测试结果:")
    for word in test_words:
        if word in model.wv:
            logging.info(f"\n与'{word}'最相似的词:")
            similar_words = model.wv.most_similar(word)
            for similar_word, score in similar_words[:5]:
                logging.info(f"{similar_word}: {score:.4f}")

if __name__ == "__main__":
    main() 