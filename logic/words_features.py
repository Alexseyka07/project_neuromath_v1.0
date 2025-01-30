import filetool
import csv
from colorama import Fore, Back, Style, init

def set_words(txt: str, words: dict) -> dict:
    """
    Add unique words from the provided text to the words dictionary and update the words file.

    Parameters:
        txt (str): The text from which to extract words.
        words (dict): The dictionary of words, where keys are words and values are their indices.
    """
    for word in [word.lower() for word in txt.split(" ")]:
        if (
            word not in words
            and word != ""
            and word != " "
            and word != "\n"
            and word != "\t"
        ):
            words[word] = len(words)
        filetool.write_words(words)
    return words


def add_args(text):
    """
    Replace numbers in a text with alphabetical arguments.

    Args:
        text (str): The input text.

    Returns:
        str: The text with numbers replaced by alphabetical arguments.
    """
    args = "abcdefghijklmnopqrstuvwxyz"
    result = ""
    arg_index = 0

    for word in text.split():
        if word.isdigit():
            result += f"¶{args[arg_index]} "
            arg_index += 1
        else:
            result += word + " "

    return result


def delete_punctuation(txt: str) -> str:

    punctuation = """!()-[]{};:'"\,<>./?@#$%^&*_~·−"""
    txt_without_punct = ""
    for char in txt:
        if char not in punctuation:
            txt_without_punct += char
    return txt_without_punct


def delete_punctuation_expr(txt: str) -> str:

    punctuation = """!-[];:'"\,<>./?@#$%^&*_~·−"""
    txt_without_punct = ""
    for char in txt:
        if char not in punctuation:
            txt_without_punct += char
        else :
            txt_without_punct += " "
    return txt_without_punct


def delete_numbers(txt: str) -> str:
    txt_without_numbers = ""
    for char in txt:
        if not char.isdigit():
            txt_without_numbers += char
    return txt_without_numbers


def set_expr(expr: str, exprs: dict) -> dict:
    for e in expr.split(" "):
        if e not in exprs and e != "" and e != " " and e != "\n" and e != "\t":
            exprs[e] = len(exprs)
    filetool.write_dataset(exprs)
    return exprs


def write_all_expr_json():
    with open("data/dataset.csv", mode="r", encoding="utf-8") as file:
        exprs = {}
        reader = csv.DictReader(file)
        for row in reader:
            expr = delete_punctuation_expr(row["expr"])
            for e in expr.split(" "):
                if e not in exprs and e != "" and e != " " and e != "\n" and e != "\t":
                    exprs[e] = len(exprs)
        filetool.write_exprs(exprs)
        try:
            init()
            print(f"{Fore.GREEN}[INFO] exprs.json updated successfully{Style.RESET_ALL}")
            print(f"{Fore.GREEN}[INFO] {len(exprs)} expressions added{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}[ERROR] {e}{Style.RESET_ALL}")
