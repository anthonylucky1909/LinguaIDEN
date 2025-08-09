def ids_to_text(ids, id_to_token):
    tokens = []
    for id in ids:
        if id == 0:  # skip padding
            continue
        tokens.append(id_to_token.get(int(id), '<unk>'))
    return ' '.join(tokens)