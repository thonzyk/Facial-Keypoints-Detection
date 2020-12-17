from .constants import *

notebook_name = "notebook.txt"
order_file_name = "to_notebook_order.txt"


def create_notebook():
    files_to_transfer = []
    f = open(PROGRAM_DATA_ROOT + order_file_name, "r")
    for line in f:
        if line[-1] == '\n':
            files_to_transfer.append(line[:-1])
        else:
            files_to_transfer.append(line)
    f.close()

    notebook_text = ""

    for file_name in files_to_transfer:
        if file_name != "__main__.py":
            f = open(PROGRAM_FUNC_ROOT + file_name, "r")
        else:
            f = open(PROGRAM_ROOT + file_name, "r")
        for line in f:
            if len(line) > 1 and line[-2] == "*":
                continue
            else:
                notebook_text = notebook_text + line
        f.close()

    with open(PROGRAM_DATA_ROOT + notebook_name, 'w') as out_file:
        out_file.write(notebook_text)
