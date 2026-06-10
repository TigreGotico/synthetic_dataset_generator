"""Unit tests for pure-Python logic in adversarial_samples.py.

Only covers functionality that does not require audio files, LLM calls,
or GPU resources.
"""
import os
import textwrap
import unittest
from unittest.mock import patch

from adversarial_samples import GraphemeAugmenter, resolve_output_mode


class TestGraphemeAugmenterInit(unittest.TestCase):
    def test_default_init(self):
        aug = GraphemeAugmenter()
        self.assertEqual(aug.max_edits, 1)
        self.assertEqual(aug.min_edits, 1)
        self.assertIn('a', aug.VOWELS)
        self.assertIn('b', aug.CONSONANTS)

    def test_custom_edits(self):
        aug = GraphemeAugmenter(max_edits=3, min_edits=2)
        self.assertEqual(aug.max_edits, 3)
        self.assertEqual(aug.min_edits, 2)

    def test_invalid_max_edits(self):
        with self.assertRaises(ValueError):
            GraphemeAugmenter(max_edits=0)

    def test_invalid_min_edits_negative(self):
        with self.assertRaises(ValueError):
            GraphemeAugmenter(min_edits=-1)

    def test_invalid_min_greater_than_max(self):
        with self.assertRaises(ValueError):
            GraphemeAugmenter(max_edits=1, min_edits=2)

    def test_extra_vowels_merged(self):
        aug = GraphemeAugmenter(language_vowels={'é', 'ã'})
        self.assertIn('é', aug.VOWELS)
        self.assertIn('ã', aug.VOWELS)

    def test_all_graphemes_is_union(self):
        aug = GraphemeAugmenter()
        self.assertEqual(aug.ALL_GRAPHEMES, aug.VOWELS | aug.CONSONANTS)


class TestGraphemeAugmenterGetCharClass(unittest.TestCase):
    def setUp(self):
        self.aug = GraphemeAugmenter()

    def test_vowel_lower(self):
        self.assertEqual(self.aug._get_char_class('a'), 'vowel')

    def test_vowel_upper(self):
        self.assertEqual(self.aug._get_char_class('E'), 'vowel')

    def test_consonant_lower(self):
        self.assertEqual(self.aug._get_char_class('b'), 'consonant')

    def test_unknown_char(self):
        self.assertIsNone(self.aug._get_char_class('1'))


class TestLevenshteinDistance(unittest.TestCase):
    def setUp(self):
        self.aug = GraphemeAugmenter()

    def test_equal_strings(self):
        self.assertEqual(self.aug._levenshtein_distance('abc', 'abc'), 0)

    def test_empty_vs_word(self):
        self.assertEqual(self.aug._levenshtein_distance('', 'abc'), 3)

    def test_one_edit(self):
        self.assertEqual(self.aug._levenshtein_distance('cat', 'bat'), 1)

    def test_symmetric(self):
        self.assertEqual(
            self.aug._levenshtein_distance('kitten', 'sitting'),
            self.aug._levenshtein_distance('sitting', 'kitten'),
        )


class TestGenerateConfusables(unittest.TestCase):
    def setUp(self):
        self.aug = GraphemeAugmenter(max_edits=1, min_edits=1)

    def test_zero_samples_returns_empty(self):
        result = self.aug.generate_confusables('hey', n_samples=0)
        self.assertEqual(result, [])

    def test_generates_requested_count(self):
        result = self.aug.generate_confusables('hey', n_samples=5)
        self.assertLessEqual(len(result), 5)
        # All results must differ from the original keyword by exactly 1 edit.
        for candidate in result:
            dist = self.aug._levenshtein_distance('hey', candidate)
            self.assertEqual(dist, 1, msg=f"'{candidate}' has distance {dist} from 'hey'")

    def test_no_duplicates(self):
        result = self.aug.generate_confusables('computer', n_samples=20)
        self.assertEqual(len(result), len(set(result)))

    def test_original_not_in_result(self):
        result = self.aug.generate_confusables('hey', n_samples=10)
        self.assertNotIn('hey', result)

    def test_exhaustive_mode_raises(self):
        with self.assertRaises(NotImplementedError):
            self.aug.generate_confusables('hey', n_samples=None)

    def test_result_is_sorted(self):
        result = self.aug.generate_confusables('wake', n_samples=10)
        self.assertEqual(result, sorted(result))


class TestResolveOutputMode(unittest.TestCase):
    def test_new_file_returns_w(self, tmp_path=None):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'out.txt')
            mode = resolve_output_mode(path, 'error')
            self.assertEqual(mode, 'w')

    def test_existing_file_error_raises(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'out.txt')
            open(path, 'w').close()
            with self.assertRaises(Exception):
                resolve_output_mode(path, 'error')

    def test_existing_file_overwrite_returns_w(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'out.txt')
            open(path, 'w').close()
            mode = resolve_output_mode(path, 'overwrite')
            self.assertEqual(mode, 'w')

    def test_existing_file_append_returns_a(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'out.txt')
            open(path, 'w').close()
            mode = resolve_output_mode(path, 'append')
            self.assertEqual(mode, 'a')
