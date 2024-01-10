#!/usr/bin/env python3
from collections import Counter, defaultdict
import os
import time
from collections import defaultdict
import time
import re

def groupByKey(m):
    groupedM = defaultdict(list)
    for k, v in m:
        groupedM[k].append(v)
    return groupedM

class Command:
    def __init__(self, raw):
        tup = raw.split(";")
        self.timestamp_epoch = int(tup[0][2:-2])
        self.timestamp_struct = time.gmtime(self.timestamp_epoch)
        a = re.split(r":\s\d{10}:0;", raw, maxsplit=1)[1]
        self.full_command = a
        self.base_command = tup[1].split()[0]

class HistoryData:
    def __init__(self, filenames):
        if isinstance(filenames, str):
            filenames = [filenames]
        commands = []
        for filename in filenames:
            with open(filename, 'rb') as f:
                it = iter(f)
                for line in it:
                    try:
                        full_line = line.decode()
                        while full_line.strip()[-1] == '\\':
                            full_line += next(it).decode().replace('\\\\ \n', '')
                        commands.append(Command(full_line))
                    except Exception as e:
                        pass
        self.commands = commands

    def get_hourly_breakdowns(self):
        days = self.group_by_day()
        all_freqs = [[] for x in range(24)]
        for day, cmds in sorted(days.items()):
            day_times = [cmd.timestamp_struct.tm_hour for cmd in cmds]
            freq_counter = Counter(day_times)
            freqs = [0 for x in range(24)]
            for hour, num in freq_counter.items():
                freqs[hour] = num
            for hour, num in enumerate(freqs):
                all_freqs[hour].append(num)
        return all_freqs

    def get_weekday_breakdowns(self):
        days = self.group_by_day()
        all_freqs = [[] for x in range(7)]
        for day, cmds in sorted(days.items()):
            all_freqs[cmds[0].timestamp_struct.tm_wday].append(len(cmds))
        return all_freqs

    def get_command_lengths(self):
        lengths = [(len(cmd.base_command), cmd) for cmd in self.commands]
        sortedLengths = sorted(lengths, key=lambda x: x[0], reverse=True)
        for c_len, cmd in sortedLengths[0:5]:
            print("  {}: {}".format(c_len, cmd.base_command))
        return [len(cmd.base_command) for cmd in self.commands]

    def group_by_day(self):
        ts = [(cmd.timestamp_struct, cmd) for cmd in self.commands]
        kv = groupByKey(
            [("{}-{}-{}".format(t.tm_year, t.tm_mon, t.tm_mday), cmd)
             for t, cmd in ts])
        return kv

    def get_base_commands(self):
        return [cmd.base_command for cmd in self.commands]

def main():
    home_dir = os.environ.get("HOME","~")
    history_file = "%s/.zsh_history" % home_dir

    all_hist = HistoryData([history_file])
    with open("./data/history_data.txt", "w") as f:
        for command in all_hist.commands:
            f.write(command.full_command)
    # List of all command history available 

    # Construct JSONL file. Call OpenAI API to get completions for each prompt
    # [{
    #     "prompt": "ssh user1@178.34.25.24 -i ~/.ssh/id_rsa",
    #     "completion": openai.Completion.create(...) # Get completion from OpenAI API using gpt3.5 turbo model
    # } ...]

    # Use completions to fine tune the gpt3.5 turbo model to generate command line commands for the user from the NL prompt
    

if __name__ == '__main__':
    main()