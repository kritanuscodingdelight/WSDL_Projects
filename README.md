# WSDL 2026 — Project Solutions

**Winter School on Deep Learning 2026**
*From Amateur to Connoisseur: A Journey Transforming Perceptrons to Agents*

Six problem sets covering gradient descent, sentence embeddings, document retrieval, speech processing, transformer fine-tuning, and YouTube API agents.

---

## Repository Structure

```
.
├── PS1_Gradient_Descent_Problem.ipynb
├── PS2_Similarity_Search_using_Sentence_Embeddings.ipynb
├── PS3_Building_a_Document_Retrieval_System_using_Embeddings_WSDL2026.ipynb
├── PS4_Speech_Separation_Transcription.ipynb
├── PS5_Fine_Tuning_a_Transformer_for_Multi_Label_Emotion_Classification_WSDL2026.ipynb
├── PS6_Final.ipynb
└── README.md
```

---

## PS1 — Gradient Descent for Function Minimization

**Goal:** Implement and visualize gradient descent on the non-convex function f(x) = x⁴ − 3x³ + 2.

### What was done
- Plotted f(x) over x ∈ [−2, 4] and identified critical points
- Computed the derivative f′(x) = 4x³ − 9x² analytically and verified against PyTorch autograd
- Implemented gradient descent from x₀ = −1.0 over 100 iterations
- Compared three learning rates

### Results

**Critical points identified:** x = 0 (saddle point), x = 2.25 (local minimum)

**Analytical vs autograd derivative at x = 2.0:** both return f′(2.0) = −4.0000

**Learning rate comparison (starting from x₀ = −1.0, 100 iterations):**

| Learning Rate | Final x | Final f(x) | Behavior |
|---|---|---|---|
| η = 0.01 | −0.0896 | 2.0022 | Slow and stable, converges near saddle region |
| η = 0.05 | −0.0200 | 2.0000 | Faster and stable, still near saddle |
| η = 0.10 | 2.3368 | −6.4627 | Escapes saddle and reaches lower minimum |

**Key finding:** η = 0.10 is the only rate that escapes the saddle region and reaches the true local minimum near x = 2.25, f(x) ≈ −6.46.

---

## PS2 — Similarity Search using Sentence Embeddings

**Goal:** Build a semantic similarity search system over 2,000 AG News headlines using sentence embeddings.

### What was done
- Loaded first 2,000 samples from the AG News training set
- Encoded all texts with `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings)
- Computed a 2000×2000 cosine similarity matrix
- Implemented a reusable `search(query, texts, embeddings, model, top_k=5)` function
- Tested on multiple queries

### Results

**Embedding shape:** (2000, 384)

**Sample query results:**

*Query: "new technology for artificial intelligence"*
```
Score: 0.4742 — NASA Develops Robust Artificial Intelligence for Planetary Rovers
NASA is planning to add a strong dose of artificial intelligence (AI) to planetary
rovers to make them much more self-reliant...
```

*Query: "stock market performance"*
```
Score: 0.5507 — Stocks Higher on Economic Data, Earnings
U.S. stocks traded higher on Tuesday, despite oil prices hitting a new high before
falling back, as U.S. economic reports showed an easing of inflationary pressure...
```

### Libraries
`datasets`, `sentence-transformers`, `torch`, `sklearn`

---

## PS3 — Document Retrieval System using Embeddings

**Goal:** Build a Wikipedia document retrieval system using semantic similarity — illustrating core RAG and vector database concepts.

### What was done
- Loaded first 2,000 articles from Simple English Wikipedia (`20220301.simple`)
- Encoded all documents with `sentence-transformers/all-MiniLM-L6-v2`
- Computed 2000×2000 cosine similarity matrix
- Implemented `retrieve(query, documents, embeddings, model, top_k=5)`
- Tested on multiple queries with qualitative analysis

### Results

**Dataset:** 2,000 Wikipedia articles loaded (sample: "April — April (Apr.) is the fourth month of the year...")

**Sample retrieval:**

*Query: "artificial intelligence"*
```
Rank 1 | Score: 0.5378 | [Artificial intelligence]
Artificial intelligence (AI) is the ability of a computer program or a machine
to think and learn. It is a field of study which tries to make computers "smart"...

Rank 2 | Score: 0.3678 | [Creativity]
Creativity is the ability of a person or group to make something new and useful...
```

### Analysis
Retrieved documents are topically relevant for clear queries. Similarity scores drop sharply after rank 1-2, indicating the embedding space is well-separated. Limitation: short Wikipedia articles with generic language can produce off-topic results at lower similarity scores.

---

## PS4 — Speech Separation and Transcription

**Goal:** Build an audio pipeline that extracts, separates, and transcribes speech from a YouTube video.

### What was done
- Downloaded a 3-minute YouTube video (multi-speaker, instrumental background) using `yt-dlp`
- Extracted audio as WAV using `moviepy`
- Separated speech from background music using **Demucs** (`htdemucs` model) — output stored in `/content/separated/htdemucs`
- Transcribed speech using **Whisper**
- Bonus: Applied a speech enhancer and compared transcription quality across three conditions

### Results

**Video used:** YouTube video with multiple speakers discussing travel/locations (London, Australia, Macedonia, USA, Italy)

**Transcription comparison (first ~100 words):**

| Pipeline | Sample Output |
|---|---|
| **Original audio** | "Hi, I'm Antonio and this is EasyLong with you from London. Where are you from? I'm from Australia. I'm originally from Macedonia..." |
| **Demucs separated** | "Hi, I'm Antonio and this is diesel and windiers from London. Where are you from? I'm from Australia..." |
| **Speech enhanced** | "Hi, I'm Antonio and this is EasyLand, which is from London. Where are you from? I'm from Australia..." |

**Finding:** The original audio produced the most accurate transcription. Demucs separation introduced minor artifacts ("diesel and windiers" vs "EasyLong"). The speech enhancer slightly improved clarity over the separated version but still underperformed the original, suggesting the original audio was already clean enough that separation was unnecessary for this video.

### Libraries
`yt-dlp`, `moviepy`, `demucs`, `openai-whisper`

---

## PS5 — Fine-Tuning DistilBERT for Multi-Label Emotion Classification

**Goal:** Fine-tune DistilBERT on the GoEmotions dataset for multi-label emotion detection across 28 emotion categories.

### What was done
- Loaded GoEmotions dataset (43,410 train / 5,426 val / 5,427 test)
- Preprocessed text: lowercased, removed special symbols, punctuation, and tokens
- Fine-tuned `distilbert-base-uncased` (66.97M parameters) for 3 epochs
- Implemented 4 strategies to improve performance on rare emotions (Bonus)

### Dataset

| Split | Samples |
|---|---|
| Train | 43,410 |
| Validation | 5,426 |
| Test | 5,427 |

**28 emotion classes:** admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

**Class imbalance:**

| Most common | Count | Rarest | Count |
|---|---|---|---|
| neutral | 14,219 | grief | 77 |
| admiration | 4,130 | pride | 111 |
| approval | 2,939 | relief | 153 |
| gratitude | 2,662 | nervousness | 164 |

### Results

**Baseline (standard fine-tuning, 3 epochs):**

| Metric | Score |
|---|---|
| eval_loss | 0.0877 |
| F1 micro | 0.5444 |
| F1 macro | 0.3399 |
| F1 weighted | 0.4936 |

**Sample predictions:**
```
"I can't believe how amazing this is! I'm so thrilled!"
→ admiration (0.58)

"This is really frustrating and makes me angry."
→ anger (0.64), annoyance (0.47)

"I feel so lonely and sad today."
→ sadness (0.79)

"The results were surprising but I expected something different."
→ surprise (0.73)
```

### Bonus: Rare Emotion Improvement Strategies

| Strategy | F1 Micro | F1 Macro | Notes |
|---|---|---|---|
| Baseline | 0.5444 | 0.3399 | Standard BCE loss |
| Class-weighted loss | 0.4581 | 0.4046 | Inverse-frequency weights (grief weight=562.8) |
| Per-class threshold tuning | **0.5589** | **0.4774** | Best threshold per class on val set |
| Augmentation + weighted loss | 0.3911 | 0.3370 | 1,606 synonym-augmented rare samples |
| Focal loss (γ=2.0) | 0.4938 | 0.4366 | Down-weights easy examples |

**Best overall:** Per-class threshold tuning (F1 micro=0.5589, F1 macro=0.4774, F1 weighted=0.5687). Rare emotions like grief and embarrassment get lower thresholds (0.85-0.90) to improve recall.

**Training config:** 3 epochs, batch size 32, lr=2e-5, GPU (CUDA)

---

## PS6 — YouTube Search Agent

**Goal:** Build a simple AI agent that accepts natural language queries and retrieves relevant YouTube videos using the YouTube Data API v3.

### What was done
- Set up Google Cloud project with YouTube Data API v3
- Implemented `search_youtube(query, max_results=5)` returning title, channel, and URL
- Built `youtube_agent()` accepting user queries and displaying structured results
- Tested on multiple queries

### Results

**Sample query: "data science"**
```
Result 1:
  Title:   What is Data Science?
  Channel: IBM Technology
  URL:     https://youtube.com/watch?v=RBSUwFGa6Fk

Result 2:
  Title:   The Complete Data Science Roadmap
  Channel: Programming with Mosh
  URL:     https://youtube.com/watch?v=9R3X0JoCLyU

Result 3:
  Title:   Data Analytics vs Data Science
  Channel: IBM Technology
  URL:     https://youtube.com/watch?v=dcXqhMqhZUo

Result 4:
  Title:   What REALLY is Data Science? Told by a Data Scientist
  Channel: Joma Tech
  URL:     https://youtube.com/watch?v=xC-c7E5PK0Y

Result 5:
  Title:   Intro to Data Science: What is Data Science?
  Channel: Steve Brunton
  URL:     https://youtube.com/watch?v=...
```

**Queries tested:** "gradient descent tutorial", "transformers neural networks", "python programming lecture", "speech recognition deep learning", "data science"

### Analysis
Retrieved videos are highly relevant -- all top results for "data science" are from credible educational channels (IBM Technology, Programming with Mosh, Joma Tech). The agent correctly calls the YouTube Data API, parses the JSON response, and formats results. Limitation: API has a daily quota limit (10,000 units/day by default) and results reflect YouTube's ranking algorithm, not pure semantic relevance.

### Libraries
`google-api-python-client`, `google-auth`

---

## Environment

All notebooks run on Google Colab with GPU acceleration (CUDA).

| Library | Version |
|---|---|
| transformers | 5.0.0 |
| datasets | 4.0.0 |
| sentence-transformers | 5.2.3 |
| torch | 2.10.0+cu128 |
| yt-dlp | 2026.2.21 |
| google-api-python-client | 2.190.0 |

---

## Author

**Kritanu Chattopadhyay**
B.Tech Mechanical Engineering, NIT Durgapur (Class of 2027)
Research affiliations: IIT Patna (AI-NLP-ML Lab) · IIT Bombay (MInDS Lab)
