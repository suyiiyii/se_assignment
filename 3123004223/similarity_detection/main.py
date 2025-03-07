import os
import sys
import warnings
from pathlib import Path

import jieba
import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 忽略警告信息
warnings.filterwarnings("ignore")

# 全局变量用于存储word2vec模型
word_vectors = None

def load_word_vectors():
    """
    加载预训练的中文词向量模型
    """
    global word_vectors
    
    # 模型路径（需要先下载模型）
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sgns.literature.bigram-char")
    
    if os.path.exists(model_path):
        try:
            print("正在加载中文词向量模型...")
            word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=False)
            print("中文词向量模型加载完成")
            return True
        except Exception as e:
            print(f"加载词向量模型出错: {e}")
    else:
        print(f"中文词向量模型不存在，请下载并保存到: {model_path}")
    
    return False

def calculate_similarity_with_embeddings(original_text: str, plagiarized_text: str) -> float:
    """
    使用词向量计算文本相似度
    """
    # 使用jieba进行分词
    original_words = list(jieba.cut(original_text))
    plagiarized_words = list(jieba.cut(plagiarized_text))
    
    # 获取每个词的词向量
    original_vectors = [word_vectors[word] for word in original_words if word in word_vectors]
    plagiarized_vectors = [word_vectors[word] for word in plagiarized_words if word in word_vectors]
    
    # 如果没有有效的向量，返回0
    if not original_vectors or not plagiarized_vectors:
        return 0.0
    
    # 计算文档向量（词向量的平均值）
    original_doc_vector = np.mean(original_vectors, axis=0).reshape(1, -1)
    plagiarized_doc_vector = np.mean(plagiarized_vectors, axis=0).reshape(1, -1)
    
    # 计算余弦相似度
    similarity = cosine_similarity(original_doc_vector, plagiarized_doc_vector)
    
    return similarity[0][0] * 100  # 转换为百分比

def calculate_similarity_tfidf(original_text: str, plagiarized_text: str) -> float:
    """
    使用TF-IDF向量化和余弦相似度计算相似度（改进版，使用jieba分词）
    """
    # 使用jieba进行分词，提高中文处理能力
    original_words = " ".join(jieba.cut(original_text))
    plagiarized_words = " ".join(jieba.cut(plagiarized_text))
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original_words, plagiarized_words])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return similarity[0][0] * 100  # 转换为百分比

def calculate_similarity(original_text: str, plagiarized_text: str) -> float:
    """
    计算原文与抄袭版论文的相似度
    如果可用，使用词向量；否则回退到TF-IDF方法
    """
    global word_vectors
    
    # 如果词向量模型未加载，尝试加载
    if word_vectors is None:
        load_word_vectors()
    
    # 如果词向量模型已加载，使用词向量方法
    if word_vectors is not None:
        return calculate_similarity_with_embeddings(original_text, plagiarized_text)
    # 否则回退到改进的TF-IDF方法
    else:
        return calculate_similarity_tfidf(original_text, plagiarized_text)


def batch_test():
    """
    批量测试相似度计算函数
    """
    test_case_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_case"))
    # 读取原文和抄袭版论文的内容
    ori = (test_case_path / "orig.txt").read_text(encoding="utf-8")

    # 遍历目录，读取其他文件内容并计算相似度
    for file in test_case_path.glob("*.txt"):
        if file.name == "orig.txt":
            continue
        
        plagiarized = file.read_text(encoding="utf-8")
        similarity = calculate_similarity(ori, plagiarized)
        print(f"{file.name}: {similarity:.2f}%")


if __name__ == "__main__":
    # 检查是否有命令行参数
    if len(sys.argv) == 4:  # 包含脚本名称在内共4个参数
        original_file = sys.argv[1]
        plagiarized_file = sys.argv[2]
        answer_file = sys.argv[3]
        
        try:
            # 读取文件内容
            with open(original_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            with open(plagiarized_file, 'r', encoding='utf-8') as f:
                plagiarized_content = f.read()
            
            # 计算相似度
            similarity = calculate_similarity(original_content, plagiarized_content)
            
            # 将结果写入答案文件
            with open(answer_file, 'w', encoding='utf-8') as f:
                f.write(f"{similarity:.2f}")
            
            print(f"重复率: {similarity:.2f}%")
            print(f"结果已写入: {answer_file}")
            
        except Exception as e:
            print(f"处理文件时出错: {e}")
    else:
        # 如果没有传入命令行参数，执行批量测试
        batch_test()


