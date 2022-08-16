import spacy
from spacy.language import Language


# Get named entities
def nlp(text):
    ner = spacy.load("en_core_web_sm")
    doc = ner(text)
    span = doc.ents
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(entities)
    return entities


# Get length of doc
@Language.component("length_component")
def length_component_function(doc):
    doc_length_chars = len(doc)
    doc_length_words = len(doc.split(" "))
    print(f"This document is {doc_length_chars} tokens long.")
    length = f"This document is {doc_length_chars} characters long and {doc_length_words} words long."
    return length
