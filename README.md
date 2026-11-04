# 🎮 Steam Game Recommender

A content-based game recommender built on ~20,000 Steam games. Pick a game and get similar ones, or just describe what you want ("a scary survival game") and get matches. Served through a Gradio UI.

This README focuses on two things: **how the data was preprocessed** and **how the text-to-vector "encoder" pipeline works**.

---

## Dataset

- Source: [Steam Games Dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) (`games_march2025_cleaned.csv`), downloaded with `kagglehub`
- First 20,000 rows loaded, 47 columns originally

---

## 1. Data Preprocessing

Goal: turn each game into **one clean text string** that describes it, so it can be encoded into a vector.

### Step 1: Keep only useful columns
From 47 columns, kept 10:
`appid, name, price, detailed_description, about_the_game, short_description, header_image, developers, publishers, genres`

### Step 2: Drop missing values
`dropna()` removed rows with missing text → **19,916 games** left. One extra game was removed manually (`Furry Arena [18+]`), leaving **19,915**.

### Step 3: Convert everything to lists of tokens
| Column | What was done |
|---|---|
| `price` | Cast to string, wrapped in a list (e.g. `['3.99']`) so price becomes a token the model can match on |
| `detailed_description`, `about_the_game`, `short_description` | Split into word lists with `.split()` |
| `genres`, `developers`, `publishers` | Stored as strings like `"['Action', 'Free To Play']"`, parsed into real lists with `ast.literal_eval` |

### Step 4: Keep multi-word genres as one token
Spaces inside genres are removed so `Free To Play` becomes `FreeToPlay` and `Massively Multiplayer` becomes `MassivelyMultiplayer`. Without this, each word would be counted as a separate feature and genres would lose their meaning.

### Step 5: Merge into a single `tags` column
```python
df['tags'] = df['price'] + df['detailed_description'] + df['about_the_game'] \
           + df['short_description'] + df['genres']
```
Because every column is a list, `+` simply concatenates them into one big token list per game.

### Step 6: Normalize the text
1. **Join** the token list into one string
2. **Lowercase** everything
3. **Stem** every word with NLTK's `PorterStemmer` (`running`, `runs` → `run`) so different forms of the same word count as one feature

```python
def stem(text):
    return " ".join(ps.stem(w) for w in text.split())
```

Result: a 3-column dataframe `appid | name | tags`, where `tags` is clean, lowercase, stemmed text.

---

## 2. Mimicking an Encoder

An encoder takes variable-length text and maps it to a **fixed-length vector** in a shared space, where similar things end up close together. This project builds that idea with classical NLP instead of a neural network.

### The encoder: `CountVectorizer`
```python
cv = CountVectorizer(stop_words='english', max_features=5000)
vectors = cv.fit_transform(new_df['tags'])   # shape: (19915, 5000)
```
- **`fit`** learns the vocabulary: the 5,000 most frequent non-stopword terms
- **`transform`** encodes any text into a 5,000-dimensional count vector
- Every game becomes a 5,000-dim vector (sparse, ~2.09M non-zero entries in total)

### The "similarity space"
```python
similarity = cosine_similarity(vectors)      # shape: (19915, 19915)
```
Cosine similarity measures the angle between two vectors, so it compares *what a game is about* rather than how long its description is.

### Mode 1: Similar games
Look up the chosen game's row in the similarity matrix, sort, and return the top 5 (excluding itself).

```
recommend('Terraria')
→ Creativerse, RPG World - Action RPG Maker, Garage: Bad Dream Adventure, Omega Crafter, Teravit
```

### Mode 2: Describe what you want (the encoder applied to a query)
A free-text query is encoded by the **same fitted encoder** as the games. Same preprocessing, same vocabulary, same vector space, so the query can be compared directly against every game.

```
User text ──► LLM expands into preferences/genres
          ──► stem() both the original text and the expansion
          ──► cv.transform()  (same vocabulary as the games)
          ──► add the two vectors together
          ──► cosine similarity vs. all 19,915 game vectors
          ──► top 5
```

The LLM step (`Llama-3.2-3B-Instruct`) rewrites short queries into richer keywords, which helps because a few words from the user would otherwise produce a very sparse vector:

```
"I want a scary survival game"
→ "scary, survival, horror, intense action, exploration, stealth, dark atmosphere"
→ Blame Him, Pitty Meaty, Uncanny Valley, Outlast, The Long Dark
```

The key rule: **a query must go through exactly the same `stem` + `cv.transform` pipeline used on the games**, otherwise the vectors live in different spaces and the scores mean nothing.

---

## 3. App (Gradio)

- **Similar games**: search box with substring + fuzzy matching (`difflib`), genre filter, result cards with header images
- **Recommend by description**: free-text box that runs the query-encoding pipeline above

---

## Setup

```bash
pip install kagglehub gradio pandas numpy scikit-learn nltk transformers torch
```

- Run `project.ipynb` top to bottom (built on Google Colab)
- The LLM step uses `meta-llama/Llama-3.2-3B-Instruct`, which requires accepting the license on Hugging Face and a GPU (`device="cuda"`)

## Limitations & Ideas

- Bag-of-words counts words, not meaning, so synonyms are only matched via the LLM expansion
- Only the first 20,000 games are used
- Next steps: TF-IDF weighting, or sentence embeddings (e.g. `sentence-transformers`) as a real neural encoder

## Tech Stack
Python · pandas · NumPy · scikit-learn · NLTK · Hugging Face Transformers · Gradio
