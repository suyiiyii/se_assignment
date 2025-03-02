from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(original_text: str, plagiarized_text: str) -> float:
    """
    计算原文与抄袭版论文的相似度，基于 TF-IDF 向量化和余弦相似度。
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original_text, plagiarized_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return similarity[0][0] * 100  # 转换为百分比

if __name__ == "__main__":
    original = "今天是星期天，天气晴，今天晚上我要去看电影。"
    plagiarized = "今天是周天，天气晴朗，我晚上要去看电影。"
    
    similarity = calculate_similarity(original, plagiarized)
    print(f"重复率: {similarity:.2f}%")
