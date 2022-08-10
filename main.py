
# import spaCy
import spacy
import numpy
import sys
sys.path.append("/home/elfouste/.local/lib/python3.8/site-packages")
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

# Create a blank English nlp object
nlp = spacy.blank("en")

# Created by processing a string of text with the nlp object
text = ("I am an A Level Student taking Computing, Maths, Further Maths and Physics. I have many hobbies including learning new languages, writing, music, sports, and modelling, but most of all I love coding. The area of computing that intrigues me most is Artificial Intelligence and machine learning, more specifically surrounding computational finance or computational linguistics. I also have an interest in web development and data science. Recently I have also taken an interest in business and finance and would one day love to combine this with my love of computer science. I hope to study computer science at university as well continuing to pursue my various hobbies in the future. I believe I am hard-working, organised and can adapt and learn quickly to anything I put my mind to. ")

subjects = ["Computing", "Maths", "Further Maths", "Physics", "business", "finance"]
subject_patterns = list(nlp.pipe(subjects))
print("subject_patterns: ", subject_patterns)
matcher = PhraseMatcher(nlp.vocab)
matcher.add("SUBEJCT", subject_patterns)


@Language.component("subject_component")
def subject_component_function(doc):
    matches = matcher(doc)
    spans = [Span(doc, start, end, label="SUBJECT") for match_id, start, end in matches]
    doc.ents = spans
    return doc


nlp.add_pipe("subject_component", after="ner")
print(nlp.pipe_names)

doc = nlp(text)
print([(ent.text, ent.label_) for ent in doc.ents])





