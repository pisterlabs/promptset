import shutil
import csv
import openai
import os
import copy


from helpers import helpers
from subtools.subtool import Subtool

class DatasetSubtool(Subtool):
    def execute(self, args):
        if args.action == 'create':
            self.__create(args)
        elif args.action == 'merge':
            self.__merge(args)
        else:
            print("Action not supported")
            exit(1)


    def __create(self, args):
        if args.sources_cat == 'real' and args.labels_cat == 'real':
            self.__createRealDataset()
        elif args.sources_cat == 'synthetic' and args.labels_cat == 'synthetic':
            self.__createSyntheticDataset()
        elif args.sources_cat == 'real' and args.labels_cat == 'synthetic':
            self.__createSyntheticDataset(file_prefix='collected_logs', dataset_file_prefix='dataset_semisynthetic_v')
        else:
            print("Categories not supported")
            exit(1)

    def __createSyntheticDataset(self, file_prefix='synthetic_logs', dataset_file_prefix='dataset_synthetic_v'):
        print("Creating synthetic dataset")
        oai_client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        # get logs
        synthetic_logs = []
        sources_path = '../sources/'
        files = os.listdir(sources_path)
        files_matching_template = [file for file in files if file_prefix in file]
        files_matching_template.sort()
        if len(files_matching_template) == 0:
            print("No logs found")
            exit(1)
        else:
            with open(sources_path + files_matching_template[-1], 'r') as f:
                csv_reader = csv.reader(f)
                next(csv_reader, None)
                synthetic_logs = [row for row in csv_reader]
                f.close()
        synthetic_summaries = []
        prompt_template = [
            {
                "role": "user",
                "content": "Summarize these logs and generate a single paragraph summary of what is happening in these logs in high technical detail: \n {}"
            }
        ]
        synthetic_logs_num = len(synthetic_logs)
        counter = 0
        check_point = 0
        save_batch_size = 25
        for lidx, rwlog in enumerate(synthetic_logs):
            try:
                log = rwlog[0]
            except:
                synthetic_summaries.append('')
                continue
            if log == '' or lidx == 0 or log == []:
                continue
            prompt = copy.deepcopy(prompt_template)
            prompt[0]["content"] = prompt[0]["content"].format(log)
            response = oai_client.chat.completions.create(
             messages = prompt,
             model="gpt-3.5-turbo",
             max_tokens=200
            ).choices[0].message.content
            synthetic_summaries.append(response)
            print("Generated {} out of {} logs' summaries".format(lidx+1, synthetic_logs_num))
            if counter >= save_batch_size:
                self.__createDataset(synthetic_logs[check_point:lidx], synthetic_summaries, file_template=dataset_file_prefix)
                synthetic_summaries = []
                counter = 0
                check_point = lidx
                print("Saved {} logs with summaries".format(save_batch_size))
            counter += 1
        self.__createDataset(synthetic_logs[check_point:lidx], synthetic_summaries, file_template=dataset_file_prefix)
        oai_client.close()

    def __createRealDataset(self):
        print("Creating real dataset")
        pass

    def __createDataset(self, logs, summaries, file_template='dataset_real_v'):
        current_version = helpers.getNewVersion(file_template, dir='../datasets')
        new_version = helpers.updateMinorVersion(current_version)
        current_file = file_template.replace('v', str(current_version))
        new_file  = file_template.replace('v', str(new_version))
        sources_path = '../datasets/'
        shutil.copyfile(sources_path + current_file + '.csv', sources_path + new_file + '.csv')
        with open(sources_path + new_file + '.csv', 'a', encoding="utf-8") as f:
            csv_writer = csv.writer(f, delimiter=',')
            for lidx, log in enumerate(logs):
                log = self.__logCleanup(str(log))
                summary = self.__logCleanup(summaries[lidx])
                csv_writer.writerow([log, summary])
    
    def __logCleanup(self, log):
        log = log.replace(',', ' ')
        log = log.replace('\n', ' ')
        log = log.replace('\r', ' ')
        log = log.replace('\t', ' ')
        return log

