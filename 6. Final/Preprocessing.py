# Imports
import numpy as np
import pandas as pd


# utility function
def correct_spellings(text, sym_spell):
    " Function corrects spellings of a given sentence"
    # Need to install and provide symspell
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)

    return suggestions[0]._term
    
    
# Preprocessing    
def preprocess(question, sym_spell, maxlen, answer=None):
    """
    For given question and answer, this function preprocess data.
    Removes any line less 2 or greater maxlen.
    Arguments:
        question: list of questions
        sym_spell: object after loading with proper vocab files
        maxlen: Maximum length to keep
        answer: List of answers, default=None
    """
    # Creating a dataframe from given data
    if answer is not None:
        df = pd.DataFrame({"question":question, "answer":answer})
    else:
        df = pd.DataFrame({"question":question})
    
    # Name of the columns
    columns = df.columns
    
    # For entire data correct spellings and get length
    for col in columns:
        df[col] = df[col].apply(correct_spellings, args=(sym_spell,))
        df[col+"_length"] = df[col].apply(lambda x: len(x.split()))
    
    # Columns which contains "length" keyword
    len_cols = df.columns[df.columns.str.contains("length")]
    for col in len_cols:
        df = df[(df[col]>=2) & (df[col]<=maxlen)]
    
    # Taking only given columns
    df = df[columns]
    
    # Returning
    if answer is not None:
        return df['question'].values, df['answer'].values
    else:
        return df['question'].values