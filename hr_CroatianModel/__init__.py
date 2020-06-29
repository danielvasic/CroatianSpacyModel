# coding: utf8
from __future__ import unicode_literals

from pathlib import Path
from spacy.util import load_model_from_init_py, get_model_meta
from collections import defaultdict
import os
import pickle
import numpy as np
import cytoolz
import networkx as nx
from spacy.tokens.token import Token
from keras_contrib.layers import CRF
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from collections import defaultdict
from spacy.language import Language

__version__ = get_model_meta(Path(__file__).parent)['version']


def load(**overrides):
    Language.factories["pi"] = lambda nlp, **cfg: PredicateIdentification(nlp, **cfg)
    Language.factories["ac"] = lambda nlp, **cfg: ArgumentClassification(nlp, **cfg)
    return load_model_from_init_py(__file__, **overrides)

class SRLComponent(object):
    @classmethod
    def load(cls, path, nlp):
        json_file = open(os.path.join(
                path, 'bilstm-srl-{}-model.json'.format(cls.name)), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        
        indexes = pickle.load(
                open(os.path.join(path, "indeksi-{}.pkl".format(cls.name)), 'rb'))
        
        model = model_from_json(
            loaded_model_json, custom_objects={'CRF': CRF})
        model.load_weights(os.path.join(
                path, 'bilstm-srl-{}-weights.h5'.format(cls.name)))
        return cls(model, indexes)

    def to_disk(self, path, exclude=None):
        if not os.path.exists(path):
            os.makedirs(path)
        pickle.dump(self._indexes, open(os.path.join(path, "indeksi-{}.pkl".format(self.__class__.name)), 'wb'))
        model_json = self._model.to_json()
        with open(os.path.join(path, "bilstm-srl-{}-model.json".format(self.__class__.name)), "w") as json_file:
            json_file.write(model_json)
        self._model.save_weights(os.path.join(path, "bilstm-srl-{}-weights.h5".format(self.__class__.name)))
                
    
    def from_disk(self, path, exclude=None):
        json_file = open(os.path.join(path, 'bilstm-srl-{}-model.json'.format(self.__class__.name)), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json, custom_objects={'CRF': CRF})
        loaded_model.load_weights(os.path.join(path, "bilstm-srl-{}-weights.h5".format(self.__class__.name)))
        return self

class PredicateIdentification(SRLComponent):
    name = "pi"
    def __init__(self, model, indexes):
        self._model = model
        self._indexes = indexes
        self._score = 0.9
        Token.set_extension('predicate', default="_", force=True)
    
    def __call__(self, doc):
        X = self.get_features([doc])
        y = self._model.predict(X)
        return self.set_predicates(doc, y)
        
    def pipe(self, docs, batch_size=1000, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            X = self.get_features(minibatch)
            y = self._model.predict(X)
            for i, doc in enumerate(minibatch):
                yield self.set_predicates(doc, y[i])
    
    def set_predicates(self, doc, y):
        y_pred = [np.array([1]) if y_prob[0] > self._score else np.array([0]) for y_prob in y[0]]
        for idx, new_y in enumerate(y_pred):
            if new_y[0] == 1:
                doc[idx]._.set('predicate', doc[idx].text)
        return doc
        
    
    def get_features(self, docs):
        X = defaultdict(list)
        for doc in docs:
            words = []
            tags = []
            deps =[]
            letters = []
            for t in doc:
                words.append(self._indexes['rijeci'][t.text] if t.text in self._indexes["rijeci"] else self._indexes["rijeci"]["UNK"])
                tags.append(self._indexes['oznake'][t.tag_] if t.tag_ in self._indexes["oznake"] else 0)
                deps.append(self._indexes['ovisnosti'][t.dep_] if t.dep_ in self._indexes["ovisnosti"] else 0 )
                word_letters =[]
                for letter in t.text:
                    word_letters.append(self._indexes['slova'][letter] if letter in self._indexes['slova'] else 0)
                letters.append(pad_sequences([word_letters], maxlen=25, padding='pre')[0])
            
            X['rijeci'].append(np.array(words))
            X['oznake'].append(np.array(tags))
            X['ovisnosti'].append(np.array(deps))
            X['slova'].append(np.array(letters))
        X['rijeci'] = np.array(X['rijeci'])
        X['oznake'] = np.array(X['oznake'])
        X['ovisnosti'] = np.array(X['ovisnosti'])
        X['slova'] = np.array(X['slova'])
        return X
        
class ArgumentClassification(SRLComponent):
    name = "ac"

    def __init__(self, model, indexes):
        self._model = model
        self._indexes = indexes
        self._reversed_indexes = {
            index: tag for tag, index in self._indexes["srl_tags"].items()
        }
        Token.set_extension('arguments', default="O", force=True)
    
    def __call__(self, doc):
        if len(list(self.predicates(doc))) == 0:
          return doc
        X, predicates = self.get_features([doc])
        y = self._model.predict(X)
        return self.set_arguments(doc, y, predicates[0])

    def set_arguments(self, doc, y, predicates):
      preds = []
      for idx, sequence in enumerate(y.argmax(axis=-1)):
        pred = []
        for item in sequence:
          if item > 0:
            pred.append(self._reversed_indexes[item])
        preds.append(pred)
      for idx, predicate in enumerate(predicates):
        tags = preds[idx]
        if predicate.text == "je":
          doc[predicate.i]._.set("arguments", {
              "AGT": doc[:predicate.i],
              "PAT": doc[predicate.i+1:],
          })
        else:
          arg_start_index = {
            tag[2:]: tags.index(tag) for tag in tags if "B" in tag
          }
          frameset = {}
          for tag, start in arg_start_index.items():
            end = max([i for i, x in enumerate(tags) if x == "I-" + tag] + [start]) + 1
            frameset[tag] = doc[start:end]
          
          doc[predicate.i]._.set("arguments", {
              arg: span for arg, span in frameset.items() if arg != "V"
          })
      return doc
        
    def pipe(self, docs, batch_size=1000, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            X, predicates = self.get_features(minibatch)
            y = self._model.predict(X)
            for i, doc in enumerate(minibatch):
                yield self.set_arguments(doc, y[i], predicates[i])
    
    def predicates(self, doc):
        for token in doc:
          if token._.predicate != "_":
            yield token    
    
    def get_features(self, docs, ctx_size = 3, maxlen=38):
      def path(argument, predicate, doc):
          G = nx.Graph()
          G.add_edges_from([(token.i, token.head.i) for token in doc])
          try:
              path = nx.shortest_path(G, source=argument.i, target=predicate.i)
          except:
              path = []
          return path

      X = defaultdict(list)
      all_predicates = []
      for doc in docs:
        predicates = []
        for predicate in self.predicates(doc):
          predicates.append(predicate)
          char_idxs = []
          tag_idxs = []
          dep_idxs = []
          pred_idxs=[]
          binary_ctx=[]

          pred_idx = predicate.i
          start_pred_ctx = max([pred_idx-ctx_size, 0])
          end_pred_ctx = min([pred_idx+ctx_size, len(doc)])

          for token in doc:
            chars = []
            for char in token.text:
              chars.append(self._indexes["chars"][char])
            char_idxs.append(pad_sequences([chars], maxlen=maxlen)[0])
            
            pred_chars = []
            for char in predicate.text:
              pred_chars.append(self._indexes["chars"][char])
            
            pred_idxs.append(pad_sequences([pred_chars], maxlen=maxlen)[0])
            tag_idxs.append(self._indexes["tags"][token.tag_])
            
            dep_idx = []
            for dep_id in path(token, predicate, doc):
              depToken = doc[dep_id]
              if depToken.dep_.lower() not in self._indexes["deps"]:
                self._indexes["deps"][depToken.dep_.lower()] = len(self._indexes["deps"]) + 1
              dep_idx.append(self._indexes["deps"][depToken.dep_.lower()])
            dep_idxs.append(pad_sequences([dep_idx], maxlen=maxlen)[0])
            
            if token.i >= start_pred_ctx and token.i <= end_pred_ctx:
              binary_ctx.append(1)
            else:
              binary_ctx.append(0)
          
          X["predicate_binary_ctx"].append(np.array(binary_ctx))
          X["chars"].append(np.array(char_idxs))
          X["tags"].append(np.array(tag_idxs))
          X["predicates"].append(np.array(pred_idxs))
          X["dependency"].append(np.array(dep_idxs))
      
      X["chars"] = pad_sequences(X["chars"], maxlen=maxlen, padding="pre")
      X["predicate_binary_ctx"] = pad_sequences(X["predicate_binary_ctx"], maxlen=maxlen, padding="pre")
      X["tags"] = pad_sequences(X["tags"], maxlen=maxlen, padding="pre")
      X["predicates"] = pad_sequences(X["predicates"], maxlen=maxlen, padding="pre")
      X["dependency"] = pad_sequences(X["dependency"], maxlen=maxlen, padding="pre")
      X["predicate_binary_ctx"] = to_categorical(X["predicate_binary_ctx"], 2)
      all_predicates.append(predicates)
      return X, all_predicates