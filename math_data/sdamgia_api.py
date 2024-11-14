from sdamgia import SdamGIA
from words_features import add_args, delete_punctuation

sdamgia = SdamGIA()
SUBJECT = "math"
CATALOG = sdamgia.get_catalog(SUBJECT)[5:19]




def print_problem(c, problem, txt, len_words,types):
    """
    Print the problem information: category_id, problem_id, text and length of words.

    Parameters:
        c (int): The category id.
        problem (int): The problem id.
        txt (str): The text of the problem.
        len_words (int): The length of words in the problem.
        types (str): The type of the problem.
    """
    print("\n--------------\n")
    print(f"Номер: {types}")
    print("category_id: ", c, "problem_id: ", problem)
    print(txt)
    print("words: ", len_words)

def get_problem(id: int):
    """
    Retrieve a problem by its ID from the SdamGIA API and clean its text.

    Parameters:
        id (int): The ID of the problem to retrieve.

    Returns:
        dict: The problem data with the cleaned condition text.
    """
    problem = sdamgia.get_problem_by_id(subject=SUBJECT, id=id)
    problem["condition"]["text"] = problem["condition"]["text"].replace("\xad", "")
    return problem


def set_words():
    """
    Process problems from each category in the catalog, clean the text, 
    and build a dictionary of unique words with their corresponding indices.

    This function retrieves problems by category from the SdamGIA API, 
    removes punctuation and unwanted characters from the problem text, 
    and builds a dictionary of unique words. It also prints problem information 
    for each unique problem encountered.

    Returns:
        dict: A dictionary where keys are unique words and values are their indices.
    """
    words = {}
    pre_txt = ""
    categories_id = []
    problems = []
    for types in CATALOG:
        for type in types["categories"]:
            categories_id.append(type["category_id"])
            for c in categories_id:
                problems = sdamgia.get_category_by_id(subject=SUBJECT, categoryid=c)
            for problem in problems:
                pr = sdamgia.get_problem_by_id(subject=SUBJECT, id=problem)
                txt = add_args(delete_punctuation(pr["condition"]["text"].replace("\xad", "")))
                if txt != pre_txt:
                    for word in [word.lower() for word in txt.split(" ")]:
                        if (
                            word not in words
                            and word != ""
                            and word != " "
                            and word != "\n"
                            and word != "\t"
                        ):
                            words[word] = len(words)
                    print_problem(c, problem, txt, len(words),CATALOG.index(types) + 6)
                    pre_txt = txt

                else:
                    print("same problem: ", problem)
    return words

