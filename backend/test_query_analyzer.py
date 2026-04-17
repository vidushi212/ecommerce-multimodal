import unittest

from utils.query_analyzer import analyze_query, analyze_query_details


class QueryAnalyzerTests(unittest.TestCase):
    def test_color_and_material_balanced(self):
        alpha = analyze_query("blue silk saree", has_image=True)
        self.assertEqual(alpha, 0.65)

    def test_size_terms_push_text_weight(self):
        alpha = analyze_query("blue ethnic kurta size m", has_image=True)
        self.assertEqual(alpha, 0.70)

    def test_image_only_query_uses_high_alpha_profile(self):
        alpha = analyze_query("", has_image=True)
        self.assertEqual(alpha, 0.85)

    def test_details_include_detected_attributes(self):
        details = analyze_query_details("black cotton formal shirt xl", has_image=True)
        self.assertIn("black", details["detected"]["colors"])
        self.assertIn("cotton", details["detected"]["materials"])
        self.assertIn("formal", details["detected"]["occasions"])
        self.assertIn("xl", details["detected"]["sizes_or_fit"])


if __name__ == "__main__":
    unittest.main()
