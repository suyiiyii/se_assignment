from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from main import (
    calculate_similarity,
    calculate_similarity_tfidf,
    calculate_similarity_with_embeddings,
    load_word_vectors,
)


class TestWordVectorLoader:
    @patch('os.path.exists')
    @patch('gensim.models.KeyedVectors.load_word2vec_format')
    def test_load_word_vectors_success(self, mock_load, mock_exists):
        # Setup mocks
        mock_exists.return_value = True
        mock_load.return_value = MagicMock()
        
        # Test function
        result = load_word_vectors()
        
        # Assertions
        assert result is True
        mock_load.assert_called_once()
    
    @patch('os.path.exists')
    def test_load_word_vectors_no_model(self, mock_exists):
        # Setup mocks
        mock_exists.return_value = False
        
        # Test function
        result = load_word_vectors()
        
        # Assertions
        assert result is False
    
    @patch('os.path.exists')
    @patch('gensim.models.KeyedVectors.load_word2vec_format')
    def test_load_word_vectors_error(self, mock_load, mock_exists):
        # Setup mocks
        mock_exists.return_value = True
        mock_load.side_effect = Exception("Test exception")
        
        # Test function
        result = load_word_vectors()
        
        # Assertions
        assert result is False

class TestSimilarityFunctions:
    def test_calculate_similarity_tfidf(self):
        original = "今天是星期天，天气晴，今天晚上我要去看电影。"
        plagiarized = "今天是周天，天气晴朗，我晚上要去看电影。"
        
        similarity = calculate_similarity_tfidf(original, plagiarized)
        
        # Test that similarity is a positive number between 0 and 100
        assert 0 <= similarity <= 100
        assert similarity > 40  # Should be relatively similar
    
    @patch('main.word_vectors')
    def test_calculate_similarity_with_embeddings(self, mock_word_vectors):
        # Setup mock for word vectors
        mock_word_vectors.__contains__.side_effect = lambda word: word in ["今天", "是", "星期天", "天气", "晴", "晚上", "我", "要", "去", "看", "电影"]
        mock_word_vectors.__getitem__.side_effect = lambda word: np.random.rand(100)  # Return random vector
        
        original = "今天是星期天，天气晴，今天晚上我要去看电影。"
        plagiarized = "今天是周天，天气晴朗，我晚上要去看电影。"
        
        similarity = calculate_similarity_with_embeddings(original, plagiarized)
        
        # Test that similarity is a positive number between 0 and 100
        assert 0 <= similarity <= 100
    
    def test_calculate_similarity_with_embeddings_no_vectors(self):
        # Test when no word vectors are found in the text
        with patch('main.word_vectors') as mock_word_vectors:
            mock_word_vectors.__contains__.return_value = False
            
            original = "AAA BBB CCC"
            plagiarized = "DDD EEE FFF"
            
            similarity = calculate_similarity_with_embeddings(original, plagiarized)
            assert similarity == 0.0
    
    @patch('main.load_word_vectors')
    @patch('main.word_vectors')
    @patch('main.calculate_similarity_with_embeddings')
    @patch('main.calculate_similarity_tfidf')
    def test_calculate_similarity_with_word_vectors(self, mock_tfidf, mock_embeddings, mock_vectors, mock_load):
        # Setup mocks
        mock_vectors.__bool__.return_value = True
        mock_embeddings.return_value = 75.0
        
        original = "今天是星期天，天气晴，今天晚上我要去看电影。"
        plagiarized = "今天是周天，天气晴朗，我晚上要去看电影。"
        
        similarity = calculate_similarity(original, plagiarized)
        
        # Assertions
        assert similarity == 75.0
        mock_embeddings.assert_called_once_with(original, plagiarized)
        mock_tfidf.assert_not_called()
    
    @patch('main.load_word_vectors')
    @patch('main.word_vectors', None)
    @patch('main.calculate_similarity_tfidf')
    def test_calculate_similarity_with_tfidf_fallback(self, mock_tfidf, mock_load):
        # Setup mocks
        mock_load.return_value = False
        mock_tfidf.return_value = 60.0
        
        original = "今天是星期天，天气晴，今天晚上我要去看电影。"
        plagiarized = "今天是周天，天气晴朗，我晚上要去看电影。"
        
        similarity = calculate_similarity(original, plagiarized)
        
        # Assertions
        assert similarity == 60.0
        mock_tfidf.assert_called_once_with(original, plagiarized)
    
    def test_identical_texts_tfidf(self):
        """Test that identical texts have 100% similarity with TFIDF method"""
        text = "这是一个测试文本，用于测试相同文本的相似度。"
        similarity = calculate_similarity_tfidf(text, text)
        assert pytest.approx(similarity, 0.1) == 100.0

