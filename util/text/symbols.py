class Symbols:
    _pad        = '_'
    _eos        = '~'
    _characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!\'(),-.:;? '

    symbols = [_pad, _eos] + list(_characters)
    sym2id_dict = {s: i for i, s in enumerate(symbols)}
    id2sym_dict = {i: s for i, s in enumerate(symbols)}