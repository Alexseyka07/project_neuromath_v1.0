OPERATIONS = {
    "ro": "ro",
    "(": "(",
    "^": "deg",
    "*": "mult",
    "/": "div",
    "+": "sum",
    "-": "sub",
    "=": "ev",
}

FIRST_OPERATIONS = {
    "root": "ro",
}

LAST_OPERATIONS = {
    "^": "deg",
    "*": "mult",
    "/": "div",
    "+": "sum",
    "-": "sub",
    "=": "ev",
}


def expr(math_list: list) -> str:
    iterations = len(math_list)
    for o in OPERATIONS:
        i = 0
        while i < iterations:
            if math_list[i] == o and o in LAST_OPERATIONS:
                e = f"{OPERATIONS[o]} ( {math_list[i-1]} , {math_list[i+1]} )"
                math_list.insert(i - 1, e)
                for x in range(3):
                    del math_list[i]
                iterations = len(math_list)
                i -= 1
            if math_list[i] == o and o == "ro":
                back_expr = []
                for j in range(i + 2, iterations):
                    if math_list[j] == ")":
                        break
                    back_expr.append(math_list[j])
                math_list.insert(i, f"ro ( {expr(back_expr)} )")
                for x in range(j - i + 1):
                    del math_list[i + 1]
                iterations = len(math_list)
                i -= 1
            if math_list[i] == o and math_list[i] == "(":
                is_more_brackets = False
                for math in range(i + 1, len(math_list)):
                    if math_list[math] == ")":
                        break
                    if math_list[math] == "(":
                        is_more_brackets = True
                        break

                if is_more_brackets:
                    i += 1
                    continue

                back_expr = []
                for j in range(i + 1, iterations):
                    if math_list[j] == ")":
                        break
                    back_expr.append(math_list[j])
                math_list.insert(i, expr(back_expr))

                for x in range(j - i + 1):
                    del math_list[i + 1]
                iterations = len(math_list)
                j = 0
                i = 0

            i += 1
            if math_list[0] == "(" and math_list[1] == "(":
                i = 0

    return str(math_list[0])


def to_expr(ex: str) -> str:
    ex = ex.replace("(", "( ").replace(")", " )")
    for o in OPERATIONS:
        ex = ex.replace(o, f" {o} ")
    ex = ex.split(" ")
    spl = [i for i in ex if (i != " " and i != "")]
    return {"expr_text": expr(spl), "base_text": ex}
