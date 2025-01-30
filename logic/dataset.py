import csv
from words_features import set_words, delete_punctuation
from vector import to_vector, y_to_vector, expr_to_vector
from filetool import read_words, write_dataset
from math_data.sdamgia_api import get_problem, add_args


class Task:
    def __init__(self, id: int, type: int, text: str, rules_json: str, expr: str):
        self.id = id
        self.type = type
        self.text = text
        self.rules_json = rules_json
        self.expr = expr


def write_type_dataset(type: int):
    with open("data/dataset.csv", mode="r", encoding="utf-8") as file:
        tasks = []
        reader = csv.DictReader(file)
        i = 0
        for row in reader:
            if row["type"] == str(type):
                task = Task(
                    id=int(row["\ufeffid"]),
                    type=int(row["type"]),
                    text=row["text"],
                    rules_json=row["rules_json"],
                    expr=row["expr"],
                )
                if task.text == "":
                    task.text = delete_punctuation(add_args(
                        get_problem(id=int(row["\ufeffid"]))["condition"]["text"]
                    ))
                i += 1
                tasks.append(task)

    x_text = []
    x_expr = []
    y = []

    for task in tasks:
        words = read_words()
        set_words(words=words, txt=task.text)
        x_text.append(to_vector(task.text))
        if task.expr == "":
            x_expr.append(expr_to_vector(task.expr))
        y.append(y_to_vector(task.rules_json))

    type_dataset = ([x_text, x_expr], y)

    write_dataset(type=type, dataset=type_dataset)

write_type_dataset(type=10)