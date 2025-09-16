import sqlite3
import re
from collections import Counter

STOP_WORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now"
])


class VocabularyManager:
    def __init__(self, db_path='vocabulary.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vocabulary (
                word TEXT PRIMARY KEY,
                frequency INTEGER NOT NULL
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS speech_patterns (
                pattern_type TEXT,
                pattern_value TEXT,
                frequency INTEGER,
                PRIMARY KEY (pattern_type, pattern_value)
            )
        ''')
        self.conn.commit()

    def _clean_and_tokenize(self, text: str) -> list:
        text = re.sub(r'[^\w\s]', '', text).lower()
        tokens = text.split()
        return [word for word in tokens if word not in STOP_WORDS and len(word) > 1]

    def _log_pattern(self, pattern_type: str, pattern_value: str):
        self.cursor.execute('''
            INSERT INTO speech_patterns (pattern_type, pattern_value, frequency)
            VALUES (?, ?, 1)
            ON CONFLICT(pattern_type, pattern_value)
            DO UPDATE SET frequency = frequency + 1;
        ''', (pattern_type, pattern_value))

    def log_phrase(self, text: str):
        words = self._clean_and_tokenize(text)
        word_counts = Counter(words)

        for word, count in word_counts.items():
            self.cursor.execute('''
                INSERT INTO vocabulary (word, frequency) VALUES (?, ?)
                ON CONFLICT(word) DO UPDATE SET frequency = frequency + ?;
            ''', (word, count, count))

        if len(words) > 3:
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                self._log_pattern('common_phrase', phrase)

            if len(words) >= 2:
                starter = f"{words[0]} {words[1]}"
                self._log_pattern('sentence_starter', starter)

        self.conn.commit()

    def get_user_speech_patterns(self) -> dict:
        patterns = {
            'common_words': [],
            'common_phrases': [],
            'sentence_starters': []
        }

        self.cursor.execute('''
            SELECT word FROM vocabulary
            WHERE frequency >= 3
            ORDER BY frequency DESC LIMIT 10
        ''')
        patterns['common_words'] = [row[0] for row in self.cursor.fetchall()]

        self.cursor.execute('''
            SELECT pattern_value FROM speech_patterns
            WHERE pattern_type = 'common_phrase' AND frequency >= 2
            ORDER BY frequency DESC LIMIT 5
        ''')
        patterns['common_phrases'] = [row[0] for row in self.cursor.fetchall()]

        self.cursor.execute('''
            SELECT pattern_value FROM speech_patterns
            WHERE pattern_type = 'sentence_starter' AND frequency >= 2
            ORDER BY frequency DESC LIMIT 3
        ''')
        patterns['sentence_starters'] = [row[0] for row in self.cursor.fetchall()]

        return patterns

    def get_random_learned_word(self, min_frequency: int = 2) -> str | None:
        self.cursor.execute('''
            SELECT word FROM vocabulary WHERE frequency >= ? ORDER BY RANDOM() LIMIT 1
        ''', (min_frequency,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def get_word_count(self, word: str) -> int:
        self.cursor.execute('SELECT frequency FROM vocabulary WHERE word = ?', (word,))
        result = self.cursor.fetchone()
        return result[0] if result else 0

    def close(self):
        self.conn.close()