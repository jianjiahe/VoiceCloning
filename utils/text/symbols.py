class Symbols:
    pad        = '_'
    eos        = '~'
    characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!\'(),-.:;? '

    symbols = [pad, eos] + list(characters)
    sym2id_dict = {s: i for i, s in enumerate(symbols)}
    id2sym_dict = {i: s for i, s in enumerate(symbols)}