def test_tokenizer_and_descriptions(tokenizer, descriptions):
    texts = [x.split() for x in list(descriptions.values())]
    sequences = tokenizer.texts_to_sequences(texts)
    texts_new = tokenizer.sequences_to_texts(sequences)
    for a in range(len(texts)):
        assert len(texts[a]) == len(sequences[a])
        assert ' '.join(texts[a]) == texts_new[a], 'String 1 is %s, string 2 is %s' % (' '.join(texts[a]), texts_new[a])