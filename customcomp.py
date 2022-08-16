import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

# Create a blank English nlp object
nlp = spacy.blank("en")


# subject custom component
subjects = ["biology", "chemistry", "physics", "maths", "english",
            "history", "geography", "spanish", "computer science"]
subject_patterns = list(nlp.pipe(subjects))
print("subject_patterns: ", subject_patterns)
matcher = PhraseMatcher(nlp.vocab)
matcher.add("SUBJECT", subject_patterns)


@Language.component("subject_component")
def subject_component_function(doc):
    matches = matcher(doc)
    spans = [Span(doc, start, end, label="SUBJECT") for match_id, start, end in matches]
    doc.ents = spans
    return doc


# animal custom component
animals = ["dog", "cat", "mouse", "horse", "bird",
           "turtle", "rabbit", "elephant", "rat"]
animal_patterns = list(nlp.pipe(subjects))
print("animal_patterns: ", animal_patterns)
matcher = PhraseMatcher(nlp.vocab)
matcher.add("ANIMAL", subject_patterns)


@Language.component("animal_component")
def animal_component_function(doc):
    matches = matcher(doc)
    spans = [Span(doc, start, end, label="ANIMAL") for match_id, start, end in matches]
    doc.ents = spans
    return doc


# Add pipes
nlp.add_pipe("subject_component", after="ner")
print(nlp.pipe_names)
nlp.add_pipe("animal_component", after="ner")
print(nlp.pipe_names)


# Process the doc
def subject_function(text):
    doc = nlp(text)
    print([(ent.text, ent.label_) for ent in doc.ents])


def animal_function(text):
    doc = nlp(text)
    print([(ent.text, ent.label_) for ent in doc.ents])
