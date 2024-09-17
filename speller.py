from spellchecker import SpellChecker

spell = SpellChecker(language='es')  # use the Spanish Dictionary

def corrige(palabra):
    retorno=[]
    # find those words that may be misspelled
    misspelled = spell.unknown(palabra)
    nospelled = spell.known(palabra)


    for word in nospelled:
        retorno.append(word)

    for word in misspelled:
        # Get the one `most likely` answer
        retorno.append(spell.correction(word))

        # Get a list of `likely` options
        #print(spell.candidates(word))
    return retorno
palabra=['ermano','hola','atividad']

print(corrige(palabra))