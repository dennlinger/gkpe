"""
Quick testing for correct functionality.
"""
from gkpe.unsupervised import sentence_tf_idf

x = """Diese Bibliothek beschäftigt sich speziell mit dem Thema Keyword-Extraction, was oftmals auch als 'Keyphrase Extraction' bezeichnet wird. 
Die Problematik ist, dass sich der Großteil an wisschenschaftlichen Papieren lediglich mit Korpora englischer Sprache beschäftigt, was dazu führt, dass es wenige (bis gar keine) brauchbaren Bibliotheken gibt, die Keywörter im Deutschen extrahieren können. 
Außerdem haben einige der multilingualen Bibliotheken leider stark restriktive Lizenzmodelle gewählt, was ihre Arbeit sehr unattraktiv für kommerzielle Nutzung macht.
Daher versuchen wir, mit einer kleinen kollektion an Algorithmen das Feld für ein größeres Publikum zu öffnen.
"""

if __name__ == "__main__":
    test = sentence_tf_idf()

    print(test.extract_keywords(x, 3, 5, False))