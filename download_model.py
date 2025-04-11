from modelscope import snapshot_download
import shutil
from pathlib import Path
import logging

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )

def download_and_save_model():
    # 设置日志
    setup_logging()
    logging.info("开始下载模型...")

    try:
        # 创建models目录
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # 从ModelScope下载模型
        model_dir = snapshot_download('lili666/text2vec-word2vec-tencent-chinese')
        logging.info(f"模型下载完成，临时保存在: {model_dir}")
        
        # 源文件和目标文件路径
        source_file = Path(model_dir) / 'light_Tencent_AILab_ChineseEmbedding.bin'
        target_path = models_dir / 'tencent_chinese_word2vec.bin'
        
        # 检查源文件是否存在
        if not source_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {source_file}")
        
        # 复制模型文件到models目录
        shutil.copy2(source_file, target_path)
        logging.info(f"模型已保存到: {target_path}")
        
        return str(target_path)
        
    except Exception as e:
        logging.error(f"下载或保存模型时出错: {str(e)}")
        raise

if __name__ == "__main__":
    model_path = download_and_save_model()
    print(f"\n模型已成功下载并保存到: {model_path}") 