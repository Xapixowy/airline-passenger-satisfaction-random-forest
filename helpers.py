from datetime import datetime


class Stepper:
    def __init__(self, step):
        self.__step = step

    def __call__(self):
        current_step = self.__step
        self.__step += 1
        return current_step


def print_header(step, title):
    print(f"### {step}. {title} ###")


def print_action(title, is_success=False):
    if is_success:
        print(f"\t✅ {title} zakończone.\n")
    else:
        print(f"\t⏳ {title}...")


def print_file_saved(file_path):
    print(f"\t📄 Plik został zapisany do: {file_path}")


def print_function_execution(title, function, *args, **kwargs):
    print_action(title)
    result = function(*args, **kwargs)
    print_action(title, True)

    return result


def generate_filename_with_timestamp(filename):
    return f"{filename}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
