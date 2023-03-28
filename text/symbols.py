""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

# --------- Updated mixed FastSpeech2 ---------------
_pad = "_"
_space = " "
_punctuation = "[]§«»¬~!'(),.:;?#"
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_cfrench = 'ÀÂÇÉÊÎÔàâæçèéêëîïôùûü"' # gb: new symbols for turntaking & ldots, [] are for notes, " for new terms.

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

# Export all symbols:
symbols = (
    [_pad]
    +[_space]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + list(_cfrench)
    + _arpabet
    # + _pinyin
)

out_symbols = cmudict.valid_alignments
