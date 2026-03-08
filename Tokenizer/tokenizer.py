#Simple Tokenizer https://youtu.be/zduSFxRajkE?si=crFxAmQWjORgIJdU
#Implemented using byte-pair encoding algorithm: https://en.wikipedia.org/wiki/Byte-pair_encoding


with open('Tokenizer/dante - inferno.txt', encoding='utf-8') as text:
    text1 = text.read()

with open('Tokenizer/dante - paradiso.txt', encoding='utf-8') as text:
    text2 = text.read()


tokens = list(text1.encode(encoding='utf-8'))

def merge(tokens, ab, Z):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == ab:
            new_tokens.append(Z)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def get_stats(tokens):
    count = {}
    for pair in zip(tokens, tokens[1:]):
        count[pair] = count.get(pair, 0) + 1
    return count

vocab_size = 356

def train(size, tokens, verbose=False):
    ids = tokens.copy()
    merges = {}
    for i in range(size-256):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        new_token = 256 + i
        ids = merge(ids, pair, new_token)
        merges[pair] = new_token
        if verbose:
            print(f'{pair} merged into {new_token}')
    if verbose:
        print()
        print('merges: ',merges)
        print()
        print(f'len original: {len(tokens)}; len merged: {len(ids)}; compression: {len(tokens) / len(ids):.2f}x')
        print()
    return merges

merges = train(vocab_size, tokens, verbose=True)

vocab = {}
for pair, idx in merges.items():
    token = []
    for i in pair:
        token.extend([i] if i not in vocab else vocab[i])
    vocab[idx] = token

print('tokens: ', vocab)
print('tokens: ', {idx: bytes(chars).decode('utf-8', errors='replace') for idx, chars in vocab.items()})
print()

def decode(ids):
    unmerged = []
    for id in ids:
        unmerged.extend([id] if id not in vocab else vocab[id])
    return bytes(unmerged).decode('utf-8', errors='replace')

def encode(txt:str):
    tokens = list(txt.encode('utf-8'))
    for pair in merges:
        stats = get_stats(tokens)
        if pair in stats:
            tokens = merge(tokens, pair, merges[pair])
    return tokens

print('text2 length: ', len(list(text2.encode(encoding='utf-8'))))
print('encoded text2 length: ',len(encode(text2)))
print(f'{decode(encode(text2)) == text2 = }')