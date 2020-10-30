import tokenizers



tokenizer=tokenizers.BertWordPieceTokenizer("./model/RoBERTa/vocab.txt")
tokenizer.enable_padding(length=16)
tokenizer.enable_truncation(max_length=16)



sentence = '河西区海河沿线的新房，均价30000，带装修，看看去吗，优惠点位很大，五一特惠'
encode = tokenizer.encode(sentence, sentence)
print(encode.ids)
print(encode.type_ids)
print(encode.tokens)
print(encode.attention_mask)
