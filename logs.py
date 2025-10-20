# logs.py

# Decorator
# def update_log(func):
#     def wrapper(self, *args, **kwargs):
#         logs = self._logs + [(func.__name__, (*args, kwargs))]
#         return func(self, *args, logs=logs, **kwargs)
#     return wrapper

def format_log(log_entry):
    fun, args = log_entry
    args_string = " & ".join(map(lambda x: x.__str__(), args))

    if fun == "filter":
        args_string = args_string.replace("[", "").replace("]", "").strip()

    args_string = args_string.replace("& {}", "")
    return f"==> {fun.upper()}: {args_string}"
