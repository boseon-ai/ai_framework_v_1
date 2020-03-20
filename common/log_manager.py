import os

class Log_Manager:
    def __init__(self, log_dir, model_name, show=True):
        self.log_dir = log_dir
        self.model_name = model_name
        self.show = show

        if os.path.exists(log_dir) == False:
            os.makedirs(log_dir)

        self.line_counter = 0
        self.logs = []

        self.add_line('Name: {}'.format(self.model_name))
        self.add_line('Log: {}'.format(self.log_dir))

    def add_line(self, line):
        self.line_counter += 1
        self.logs.append(line)
        if self.show:
            print (line)

    def print_logs(self):
        for i in range(0, self.line_counter):
            line = self.logs[i]
            print (line)

    def write_logs(self):
        with open('report_' + self.model_name, 'w') as f:
            for i in range(0, self.line_counter):
                line = self.logs[i]
                f.write(str(line))
                f.write('\n')
        
