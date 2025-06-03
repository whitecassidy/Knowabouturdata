import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import json

class TestDataInsights(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'age': [25, 30, None, 40, 35],
            'category': ['A', 'B', 'A', None, 'B'],
            'price': [100.0, None, 300.0, 400.0, 500.0]
        })
        
        # Sample valid LLM response
        self.valid_llm_response = {
            "insights": "This dataset appears to be a customer or product dataset with age, category, and price information. Possible analysis questions: 1) What is the average price by category? 2) Is there a correlation between age and price? 3) How are missing values distributed across columns?",
            "preprocessing_suggestions": {
                "impute_age": "Impute missing values in 'age' with mean (missing: 20.0%).",
                "impute_price": "Impute missing values in 'price' with mean (missing: 20.0%).",
                "encode_category": "Encode categorical column 'category' for analysis.",
                "explanation": "Handling missing values and encoding categorical data will prepare the dataset for analysis."
            },
            "visualization_suggestions": [
                {"query": "Show histogram of age", "desc": "Histogram showing age distribution"},
                {"query": "Show bar chart of category counts", "desc": "Bar chart showing distribution of categories"},
                {"query": "Show scatter plot of age vs price", "desc": "Scatter plot to explore age-price relationship"}
            ],
            "model_recommendations": "For price prediction: 1) Linear Regression - good for numeric prediction, 2) Random Forest - handles both numeric and categorical features well."
        }

    @patch('openai.OpenAI')
    def test_combined_analysis_agent_success(self, mock_openai):
        # Mock the OpenAI client response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(self.valid_llm_response)))]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        # Import the function here to avoid circular imports
        from new import CombinedAnalysisAgent

        # Run the analysis
        result = CombinedAnalysisAgent(self.df)

        # Verify the structure and content of the result
        self.assertIsInstance(result, dict)
        self.assertIn("insights", result)
        self.assertIn("preprocessing_suggestions", result)
        self.assertIn("visualization_suggestions", result)
        self.assertIn("model_recommendations", result)

        # Verify preprocessing suggestions
        self.assertIsInstance(result["preprocessing_suggestions"], dict)
        self.assertIn("explanation", result["preprocessing_suggestions"])

        # Verify visualization suggestions
        self.assertIsInstance(result["visualization_suggestions"], list)
        self.assertTrue(len(result["visualization_suggestions"]) > 0)
        for suggestion in result["visualization_suggestions"]:
            self.assertIn("query", suggestion)
            self.assertIn("desc", suggestion)

    @patch('openai.OpenAI')
    def test_combined_analysis_agent_invalid_json(self, mock_openai):
        # Mock an invalid JSON response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Invalid JSON response"))]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        from new import CombinedAnalysisAgent

        # Run the analysis
        result = CombinedAnalysisAgent(self.df)

        # Verify error handling
        self.assertIsInstance(result, dict)
        self.assertIn("Error:", result["insights"])
        self.assertEqual(result["preprocessing_suggestions"], {})
        self.assertEqual(result["visualization_suggestions"], [])
        self.assertIn("Error:", result["model_recommendations"])

    @patch('openai.OpenAI')
    def test_combined_analysis_agent_missing_keys(self, mock_openai):
        # Mock a response with missing required keys
        invalid_response = {"insights": "Some insights"}  # Missing other required keys
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(invalid_response)))]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        from new import CombinedAnalysisAgent

        # Run the analysis
        result = CombinedAnalysisAgent(self.df)

        # Verify error handling for missing keys
        self.assertIsInstance(result, dict)
        self.assertIn("Error:", result["insights"])
        self.assertEqual(result["preprocessing_suggestions"], {})
        self.assertEqual(result["visualization_suggestions"], [])
        self.assertIn("Error:", result["model_recommendations"])

if __name__ == '__main__':
    unittest.main() 