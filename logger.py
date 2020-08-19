import time


class Logger:

    def __init__(self, log_path):
        self.log_path = log_path
        self.log_file = open(log_path, "rw")
        self.stdout_print = True
        self.init_time = time.time()
        self.print_format = "Time passed [minutes]: {:.2f}.     {}"

    def enable_stdout_prints(self):
        self.stdout_print = True

    def disable_stdout_prints(self):
        self.stdout_print = False

    def log_print(self, msg):
        minutes_passed = (time.time() - self.init_time) / 60
        formatted_msg = self.print_format.format(minutes_passed, msg)
        if self.stdout_print:
            print(formatted_msg)
        self.log_file.write(formatted_msg + "\n")

    def flush(self):
        self.log_file.flush()
