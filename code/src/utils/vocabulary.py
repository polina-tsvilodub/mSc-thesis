import re
import os
import pickle 
from collections import Counter
from torchtext.data import get_tokenizer
from pycocotools.coco import COCO
import gensim


class Vocabulary(object):

    def __init__(self,
        vocab_threshold,
        vocab_file='../../data/vocab.pkl',
        start_word="START",
        end_word="END",
        unk_word="UNK",         
        annotations_file="../../data/val/annotation/captions_val2014.json",
        pad_word="PAD",
        vocab_from_file=False,
        from_pretrained=False,
        ):
        """
        Initialize the vocabulary object.
        
        Args:
        -----
            vocab_threshold: Minimum word count threshold.
            vocab_file: File containing the vocabulary.
            start_word: Special word denoting sentence start.
            end_word: Special word denoting sentence end.
            unk_word: Special word denoting unknown words.
            annotations_file: Path for train/val annotation file.
            vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                            If True, load vocab from from existing vocab_file, if it exists
            from_pretrained: If True, use pretrained embeddings. If False, train embeddings from stractch.
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_word = pad_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.from_pretrained = from_pretrained
        if from_pretrained:
            _fasttext_path = "~/gensim-data/fasttext-wiki-news-subwords-300/fasttext-wiki-news-subwords-300.gz"
            self._fasttext = gensim.models.KeyedVectors.load_word2vec_format(_fasttext_path)
        
        self.get_vocab()

    def get_vocab(self):
        """
        Load the vocabulary from file OR build the vocabulary from scratch.
        Assign word2idx and idx2word mappings.
        If vocabulary built from scratch, write out to pickle file.
        """

        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file!')
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)
        
    def build_vocab(self):
        """
        Populate the dictionaries for converting tokens to integers (and vice-versa).
        """
        self.init_vocab()

        # only add caption words if the vocab needs to be built from the dataset  
        if not self.from_pretrained:
            self.add_word(self.start_word)
            self.add_word(self.end_word)
            self.add_word(self.unk_word)
            self.add_word(self.pad_word)
            self.add_captions()

    def init_vocab(self):
        """
        If building vocab from dataset, initialize the dictionaries for converting tokens to integers (and vice-versa).
        If using a pretrained vocabulary, set the mappings to retrieved dictionaries.
        """
        if self.from_pretrained:
            self.word2idx = self._fasttext.key_to_index
            self.idx2word = dict(zip(np.arange(len(self._fasttext.index_to_key)), self._fasttext.index_to_key)) 
        else:
            self.word2idx = {} 
            self.idx2word = {}
            self.idx = 0

    def add_word(self, word):
        """
        Add a token to the vocabulary.

        Args:
        ----
            word: str
                Word to be added to the vocabulary.
        """
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """
        Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold.
        Used if vocab is built from data.
        """
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        tokenizer = get_tokenizer("basic_english")
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            caption = caption.lower().strip()
            caption = re.sub(r"[^a-zA-Z.,!?]+", r" ", caption)
            tokens = tokenizer(caption) 
            counter.update(tokens)

            if i % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        """
        Returns index of word, when calling voab(word) on instantiated Vocabulary object.

        Args:
        -----
            word: str
                Word for which the index is returned.
        Returns: int
            Index of word        
        """
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        """
        Returns number of unique vocabulary tokens.
        """
        return len(self.word2idx)