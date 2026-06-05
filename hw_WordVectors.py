import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
import gensim.downloader as api

# =============================================
# 1. 词表
# =============================================
words = [
    # 城市
    "Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Chengdu", "Wuhan", 
    "NewYork", "London", "Paris", "Tokyo", "Berlin", "Sydney", 
    "Moscow", "Washington", "Rome", "Bangkok",
    # 水果
    "apple", "orange", "banana", "grape", "strawberry", "watermelon", 
    "mango", "pear", "peach", "cherry",
    # 动物
    "cat", "dog", "tiger", "lion", "elephant", "monkey", 
    "rabbit", "snake", "bird", "fish",
    # 职业
    "doctor", "teacher", "engineer", "lawyer", "accountant", 
    "journalist", "actor", "director", "writer", "scientist",
    # 国家
    "China", "USA", "UK", "France", "Germany", "Japan", 
    "Korea", "India", "Russia", "Australia",
    # 情感
    "happy", "sad", "angry", "fear", "surprise", "disgust", 
    "love", "hate", "jealousy", "sympathy",
    # 交通工具
    "car", "train", "plane", "ship", "bicycle", "motorcycle", 
    "subway", "bus", "taxi", "truck",
    # 亲属
    "father", "mother", "brother", "sister", "son", "daughter", 
    "grandfather", "grandmother", "uncle", "aunt",
    # 食物
    "rice", "noodle", "bread", "cake", "beef", "pork", 
    "chicken", "fish", "shrimp", "crab",
    # 学科
    "math", "physics", "chemistry", "biology", "history", 
    "geography", "english", "politics", "economics", "psychology",
    "king", "queen", "man", "woman", "airport", "station"
]

print(f"一共 {len(words)} 个词\n")

# =============================================
# 2. 加载模型
# =============================================

# --- 方法一：GloVe ---
print("下载/加载 GloVe...")
glove_model = api.load("glove-wiki-gigaword-50")
glove_vecs = {w: glove_model[w] for w in words if w in glove_model.key_to_index}
print(f"✅ 方法一完成：GloVe (50维)，有效词 {len(glove_vecs)} 个\n")

# --- 方法二：Word2Vec ---
print("下载/加载 Word2Vec...")
w2v_model = api.load("word2vec-google-news-300")
w2v_vecs = {w: w2v_model[w] for w in words if w in w2v_model.key_to_index}
print(f"✅ 方法二完成：Word2Vec (300维)，有效词 {len(w2v_vecs)} 个\n")

# --- 方法三：FastText ---
print("下载/加载 FastText...")
ft_model = api.load("fasttext-wiki-news-subwords-300")
ft_vecs = {w: ft_model[w] for w in words if w in ft_model.key_to_index}
print(f"✅ 方法三完成：FastText (300维)，有效词 {len(ft_vecs)} 个\n")

# =============================================
# 3. 任务一：t-SNE 可视化
# =============================================
print("进行 t-SNE 降维和可视化（使用 Word2Vec 向量）...")

valid_words = [w for w in words if w in w2v_vecs]
valid_vecs = np.array([w2v_vecs[w] for w in valid_words])

tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
coords = tsne.fit_transform(valid_vecs)

plt.figure(figsize=(14, 14))
plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
for i, word in enumerate(valid_words):
    plt.annotate(word, xy=(coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.8)
plt.title('100 Word Vectors t-SNE Visualization (Word2Vec)')
plt.tight_layout()
plt.savefig('word_vectors_tsne_English.png')
print("✅ 可视化图片已保存为 word_vectors_tsne_English.png\n")

# =============================================
# 4. 任务二：余弦相似度计算
# =============================================
def cosine_sim(v1, v2):
    return 1 - cosine(v1, v2)

print("余弦相似度结果（使用 Word2Vec 向量）：")
if 'Beijing' in w2v_vecs and 'Tokyo' in w2v_vecs:
    print(f"Beijing vs Tokyo : {cosine_sim(w2v_vecs['Beijing'], w2v_vecs['Tokyo']):.4f}")
else:
    print("Beijing 或 Tokyo 不在词表中")

if 'Beijing' in w2v_vecs and 'apple' in w2v_vecs:
    print(f"Beijing vs apple : {cosine_sim(w2v_vecs['Beijing'], w2v_vecs['apple']):.4f}")
else:
    print("Beijing 或 apple 不在词表中")

if 'orange' in w2v_vecs and 'apple' in w2v_vecs:
    print(f"orange vs apple : {cosine_sim(w2v_vecs['orange'], w2v_vecs['apple']):.4f}")
else:
    print("orange 或 apple 不在词表中")
print()

# =============================================
# 5. 任务三：词向量类比
# =============================================
def analogy(pos_words, neg_words, vec_dict):
    # 检查词是否存在
    for w in pos_words + neg_words:
        if w not in vec_dict:
            return f"❌ 词 '{w}' 不在词表中"
    
    target = np.zeros_like(next(iter(vec_dict.values())))
    for w in pos_words:
        target += vec_dict[w]
    for w in neg_words:
        target -= vec_dict[w]
    
    best_word = None
    best_score = -1
    for w, v in vec_dict.items():
        if w in pos_words + neg_words:
            continue
        score = cosine_sim(target, v)
        if score > best_score:
            best_score = score
            best_word = w
    return f"{best_word} (相似度: {best_score:.4f})"

print("词向量类比结果（使用 Word2Vec 向量）：")
print(f"king - man + woman = {analogy(['king', 'woman'], ['man'], w2v_vecs)}")
print(f"China - Beijing + Washington = {analogy(['China', 'Washington'], ['Beijing'], w2v_vecs)}")
print(f"airport - plane + train = {analogy(['airport', 'train'], ['plane'], w2v_vecs)}")

print("\n✅ 所有任务完成！")