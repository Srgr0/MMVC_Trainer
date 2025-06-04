"""
Defines the set of symbols used in text input to the model.
"""

# 日本語音素記号
_jamo_leads = ['g', 'gg', 'n', 'd', 'dd', 'r', 'm', 'b', 'bb', 's', 'ss', '', 'j', 'jj', 'ch', 'k', 't', 'p', 'h']
_jamo_vowels = ['a', 'ae', 'ya', 'yae', 'eo', 'e', 'yeo', 'ye', 'o', 'wa', 'wae', 'oe', 'yo', 'u', 'weo', 'we', 'wi', 'yu', 'eu', 'ui', 'i']
_jamo_tails = ['', 'g', 'gg', 'gs', 'n', 'nj', 'nh', 'd', 'r', 'rg', 'rm', 'rb', 'rs', 'rt', 'rp', 'rh', 'm', 'b', 'bs', 's', 'ss', 'ng', 'j', 'ch', 'k', 't', 'p', 'h']

# 日本語音素（pyopenjtalk対応）
_japanese_characters = [
    'A', 'E', 'I', 'N', 'O', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd', 'dy', 'e',
    'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky', 'm', 'my', 'n', 'ny', 'o',
    'p', 'py', 'r', 'ry', 's', 'sh', 'sp', 't', 'ts', 'ty', 'u', 'v', 'w', 'y', 'z', 'zy'
]

# 記号
_symbols = [
    'UNK', '_', '-', '!', "'", '(', ')', ',', '.', ':', ';', '?', ' ', 'sp', 'sil'
] + _japanese_characters

# 音素マッピング
phoneme_list = [
    "N", "a", "a:", "b", "by", "ch", "cl", "d", "dy", "e", "e:", "f", "g", "gy", "h", "hy", "i", "i:", "j", "k", "ky", "m", "my", "n", "ny", "o", "o:", "p", "py", "r", "ry", "s", "sh", "sp", "t", "ts", "ty", "u", "u:", "v", "w", "y", "z", "zy", "sil"
]

# pyopenjtalk から VITS に対応するマッピング
_phoneme_mappings = {
    "a": "a", "i": "i", "u": "u", "e": "e", "o": "o",
    "A": "a", "I": "i", "U": "u", "E": "e", "O": "o",
    "ka": "k a", "ki": "k i", "ku": "k u", "ke": "k e", "ko": "k o",
    "ga": "g a", "gi": "g i", "gu": "g u", "ge": "g e", "go": "g o",
    "sa": "s a", "shi": "sh i", "su": "s u", "se": "s e", "so": "s o",
    "za": "z a", "ji": "zy i", "zu": "z u", "ze": "z e", "zo": "z o",
    "ta": "t a", "chi": "ch i", "tsu": "ts u", "te": "t e", "to": "t o",
    "da": "d a", "di": "dy i", "du": "d u", "de": "d e", "do": "d o",
    "na": "n a", "ni": "n i", "nu": "n u", "ne": "n e", "no": "n o",
    "ha": "h a", "hi": "h i", "fu": "f u", "he": "h e", "ho": "h o",
    "ba": "b a", "bi": "b i", "bu": "b u", "be": "b e", "bo": "b o",
    "pa": "p a", "pi": "p i", "pu": "p u", "pe": "p e", "po": "p o",
    "ma": "m a", "mi": "m i", "mu": "m u", "me": "m e", "mo": "m o",
    "ya": "y a", "yu": "y u", "yo": "y o",
    "ra": "r a", "ri": "r i", "ru": "r u", "re": "r e", "ro": "r o",
    "wa": "w a", "wi": "w i", "we": "w e", "wo": "w o",
    "nn": "N", "n": "n", "kya": "ky a", "kyu": "ky u", "kyo": "ky o",
    "gya": "gy a", "gyu": "gy u", "gyo": "gy o",
    "sha": "sh a", "shu": "sh u", "sho": "sh o",
    "ja": "zy a", "ju": "zy u", "jo": "zy o",
    "cha": "ch a", "chu": "ch u", "cho": "ch o",
    "nya": "ny a", "nyu": "ny u", "nyo": "ny o",
    "hya": "hy a", "hyu": "hy u", "hyo": "hy o",
    "bya": "by a", "byu": "by u", "byo": "by o",
    "pya": "py a", "pyu": "py u", "pyo": "py o",
    "mya": "my a", "myu": "my u", "myo": "my o",
    "rya": "ry a", "ryu": "ry u", "ryo": "ry o",
    "vu": "v u", "va": "v a", "vi": "v i", "ve": "v e", "vo": "v o",
}

# 音素リスト
phonemes = ['sil'] + phoneme_list + ['sp']

# 文字から ID へのマッピング
symbols = _symbols

# 音素から ID へのマッピングを作成
def build_symbol_to_id():
    return {s: i for i, s in enumerate(symbols)}

def build_id_to_symbol():
    return {i: s for i, s in enumerate(symbols)}

# ID マッピング
symbol_to_id = build_symbol_to_id()
id_to_symbol = build_id_to_symbol()

# 長音符処理
_long_vowel_map = {
    'a': 'a:',
    'i': 'i:', 
    'u': 'u:',
    'e': 'e:',
    'o': 'o:'
}

# 音素IDの取得
def get_phoneme_id(phoneme):
    if phoneme in symbol_to_id:
        return symbol_to_id[phoneme]
    else:
        return symbol_to_id['UNK']

def get_phoneme_from_id(id):
    if id < len(symbols):
        return symbols[id]
    else:
        return 'UNK'

# 記号数
SYMBOLS_COUNT = len(symbols)
PAD_ID = symbol_to_id['_']
UNK_ID = symbol_to_id['UNK']
SIL_ID = symbol_to_id['sil']
SP_ID = symbol_to_id['sp']
