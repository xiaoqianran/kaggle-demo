"""Tiny built-in NLP corpora — no external download for classical stages."""
from __future__ import annotations

# Sentiment / topic classification (English, deliberately tiny & clear)
CLS_TRAIN = [
    ("I love this movie, amazing acting and plot", "pos"),
    ("Fantastic film, highly recommend watching", "pos"),
    ("Wonderful experience, best cinema ever", "pos"),
    ("Great story and superb performances", "pos"),
    ("Enjoyed every minute of this masterpiece", "pos"),
    ("Brilliant direction and beautiful score", "pos"),
    ("This movie is terrible and boring", "neg"),
    ("Awful film, waste of time and money", "neg"),
    ("Horrible acting and ridiculous plot", "neg"),
    ("I hate this movie, complete disaster", "neg"),
    ("Boring and poorly written script", "neg"),
    ("Worst cinema experience of my life", "neg"),
    ("The stock market rose after good earnings", "business"),
    ("Investors buy shares amid strong growth", "business"),
    ("Company profits beat analyst estimates", "business"),
    ("Federal bank raises interest rates today", "business"),
    ("Goal scored in extra time seals victory", "sports"),
    ("The team won the championship final", "sports"),
    ("Athlete breaks world record in sprint", "sports"),
    ("Coach praises defense after clean sheet", "sports"),
]

CLS_TEST = [
    ("I really love the amazing plot", "pos"),
    ("Terrible boring waste of money", "neg"),
    ("Shares climb after strong profits", "business"),
    ("Team wins the final championship", "sports"),
    ("Superb and wonderful film experience", "pos"),
    ("Horrible disaster of a movie", "neg"),
]

# Parallel mini corpus for translation (en→zh-style romanized / simple en→fr)
MT_PAIRS = [
    ("hello", "bonjour"),
    ("goodbye", "au revoir"),
    ("thank you", "merci"),
    ("good morning", "bonjour"),
    ("I love cats", "j aime les chats"),
    ("I love dogs", "j aime les chiens"),
    ("the cat sleeps", "le chat dort"),
    ("the dog runs", "le chien court"),
    ("she reads a book", "elle lit un livre"),
    ("he writes a letter", "il ecrit une lettre"),
    ("where is the station", "ou est la gare"),
    ("how are you", "comment allez vous"),
    ("I am fine", "je vais bien"),
    ("see you tomorrow", "a demain"),
    ("please help me", "aidez moi s il vous plait"),
    ("the weather is nice", "il fait beau"),
]

# Documents for ranking / QA / summarization
DOCS = [
    {
        "id": "d1",
        "title": "Photosynthesis",
        "text": (
            "Photosynthesis is the process by which green plants convert sunlight into chemical energy. "
            "Chlorophyll in leaves absorbs light. Carbon dioxide and water become glucose and oxygen. "
            "This process sustains most life on Earth by producing oxygen and food."
        ),
    },
    {
        "id": "d2",
        "title": "Newton Laws",
        "text": (
            "Newton's first law states that an object stays at rest or in uniform motion unless acted upon by a force. "
            "The second law says force equals mass times acceleration. "
            "The third law says every action has an equal and opposite reaction."
        ),
    },
    {
        "id": "d3",
        "title": "Machine Learning",
        "text": (
            "Machine learning algorithms learn patterns from data without being explicitly programmed for every rule. "
            "Supervised learning uses labeled examples. Unsupervised learning finds structure in unlabeled data. "
            "Deep learning uses multi-layer neural networks for vision and language tasks."
        ),
    },
    {
        "id": "d4",
        "title": "French Revolution",
        "text": (
            "The French Revolution began in 1789 and reshaped European politics. "
            "The storming of the Bastille became a symbol of popular uprising. "
            "Ideals of liberty, equality, and fraternity influenced later democracies."
        ),
    },
]

QA_PAIRS = [
    ("What do plants produce during photosynthesis?", "d1", "oxygen"),
    ("What does Newton's second law say?", "d2", "force equals mass times acceleration"),
    ("What does supervised learning use?", "d3", "labeled examples"),
    ("When did the French Revolution begin?", "d4", "1789"),
]

# NER-style BIO sentences (token, tag)
NER_TRAIN = [
    (["John", "lives", "in", "Paris"], ["B-PER", "O", "O", "B-LOC"]),
    (["Mary", "works", "at", "Google"], ["B-PER", "O", "O", "B-ORG"]),
    (["Paris", "is", "in", "France"], ["B-LOC", "O", "O", "B-LOC"]),
    (["Apple", "hired", "Tim", "Cook"], ["B-ORG", "O", "B-PER", "I-PER"]),
    (["Berlin", "hosts", "many", "startups"], ["B-LOC", "O", "O", "O"]),
    (["Alice", "visited", "London", "yesterday"], ["B-PER", "O", "B-LOC", "O"]),
    (["Microsoft", "is", "based", "in", "Seattle"], ["B-ORG", "O", "O", "O", "B-LOC"]),
    (["Bob", "met", "Carol", "in", "Tokyo"], ["B-PER", "O", "B-PER", "O", "B-LOC"]),
]

NER_TEST = [
    (["John", "visited", "Berlin"], ["B-PER", "O", "B-LOC"]),
    (["Google", "hired", "Alice"], ["B-ORG", "O", "B-PER"]),
]

# Table for table-QA
TABLE = {
    "columns": ["City", "Country", "Population_M", "Language"],
    "rows": [
        ["Tokyo", "Japan", 37, "Japanese"],
        ["Paris", "France", 11, "French"],
        ["Cairo", "Egypt", 22, "Arabic"],
        ["Sao Paulo", "Brazil", 22, "Portuguese"],
        ["New York", "USA", 19, "English"],
    ],
}

TABLE_QA = [
    ("Which city is in Japan?", "Tokyo"),
    ("What language is spoken in Paris?", "French"),
    ("Which city has population 37 million?", "Tokyo"),
    ("What is the country of Cairo?", "Egypt"),
]
