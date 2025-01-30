import filetool
from words_features import add_args

# конвертация текста в вектор
def to_vector(text: str) -> list:
    text = add_args(text)
    words = filetool.read_words()
    vector = [0.0] * len(words)
    for word in [word.lower() for word in text.split(" ") if word != ""]:
        vector[words[word.lower()]] = 1.0
    return vector


# конвертация выражения в вектор
def expr_to_vector(expr: str) -> list:
    expr_elements = filetool.read_exprs()
    vector = [0.0] * len(expr_elements)
    if expr == "":
        return vector
    for e in expr.split(" "):
        if e != "" and e != " " and e != "\n" and e != "\t":
            vector[expr_elements[e.lower()]] = 1.0
    return vector


# конвертация правил в вектор
def  y_to_vector(text: str) -> list:
    rules = {}
    for r in range(1, 44):
        rules[str(r)] = r
    vector = [0.0] * 44
    text = text[2:].replace("}]", "")
    for word in [word.lower() for word in text.replace(",", "").split(" ") if word != ""]:
        vector[rules[word]] = 1.0
    return vector
