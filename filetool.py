import json
import csv


class Rule:
    def __init__(
        self,
        id: int,
        rule: str,
        expr: str,
        example: str,
        description: str,
        description_img: str,
    ):
        self.id = id
        self.rule = rule
        self.expr = expr
        self.example = example
        self.description = description
        self.description_img = description_img


def read_words() -> dict:
    with open("./data/words.json", "r") as f:
        words = f.read()
    return json.loads(words)


def read_exprs() -> dict:
    with open("./data/exprs.json", "r") as f:
        expr = f.read()
    return json.loads(expr)


def get_rules() -> dict:
    rules = {}
    with open(f"./data/rules.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["\ufeffid"] != "":
                rule = Rule(
                    id=int(row["\ufeffid"]),
                    rule=row["rule"],
                    expr=row["expr"],
                    example=row["example"],
                    description=row["description"],
                    description_img=row["description_img"],
                )
                rules[rule.id] = rule
    return rules


def write_words(words: dict):
    with open("./data/words.json", "w") as f:
        f.write(json.dumps(words))


def write_exprs(expr: dict):
    with open("./data/exprs.json", "w") as f:
        f.write(json.dumps(expr))


def get_dataset(type: int) -> dict:
    with open(f"./data/dataset{type}.json", "r") as f:
        dataset = json.loads(f.read())
    return dataset


def write_dataset(type: int, dataset):
    with open(f"./data/dataset{type}.json", "w") as f:
        f.write(json.dumps(dataset))


def write_exprs(exprs: dict):
    with open("./data/exprs.json", "w") as f:
        f.write(json.dumps(exprs))
