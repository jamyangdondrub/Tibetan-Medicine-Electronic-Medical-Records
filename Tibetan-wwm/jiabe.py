import sentencepiece as spm
import  regex as re
# 加载 sentencepiece 模型
spm_model_path = "/home/yang/Desktop/TBERT/bert-tibetan/vocab.model"
sp = spm.SentencePieceProcessor()
sp.load(spm_model_path)

# 获取 sentencepiece 模型的词汇表
vocab_size = sp.get_piece_size()
id_to_piece = {i: sp.id_to_piece(i) for i in range(vocab_size)}
# 定义音节分词函数

def syllable_tokenize(text):
    # 这里实现你的音节分词逻辑，将文本切分成音节
    # 例如，你可以使用正则表达式来进行音节分词
    syllables = []  
    slel =  re.findall(r'[\u0F00-\u0FFF]+|.', text)
    print(slel)
    #slel = re.findall(r'\X+', text)
    jia = [s + '་' for s in '་'.join(slel).split('་')]
    for s in jia:
        syllables.append(s) 
    return syllables

# 将输入音节切分为 token
input_tokens = syllable_tokenize('ཕྲུམ་སྐྲངས་སྦོས་ཆགས་པ')


UNK_SYMBOL = 'UNK'
UNK_ID = 0

# 匹配输入 token 和 sentencepiece 子词，生成对应的 token ID
input_ids = []
for token in input_tokens:
    # 如果输入 token 在 sentencepiece 词汇表中，将其对应的 ID 添加到 input_ids 中
    if token in id_to_piece.values():
        token_id = list(id_to_piece.keys())[list(id_to_piece.values()).index(token)]
        input_ids.append(token_id)
    else:
         #print(f"Token '{token}' 不在 sentencepiece 词汇表中")
         input_ids.append(UNK_ID)  #输入token 不在sentencepiece词汇表中，将其替换为 UNK，并将对应的 token_id 添加到 input_ids 中
print(input_ids)


