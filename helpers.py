import time
import functools
from datetime import datetime


class Stepper:
    def __init__(self, step):
        self.__step = step

    def __call__(self):
        current_step = self.__step
        self.__step += 1
        return current_step


def log_execution(title, measure_time=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\t⏳ {title}...")
            if measure_time:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"\t🕒 Czas wykonania: {execution_time:.2f} s")
            else:
                result = func(*args, **kwargs)

            print(f"\t✅ {title} zakończone.\n")
            return result

        return wrapper

    return decorator


def print_header(step, title):
    print(f"### {step}. {title} ###")


def print_file_saved(file_path):
    print(f"\t📄 Plik został zapisany do: {file_path}")


def generate_filename_with_timestamp(filename):
    return f"{filename}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
