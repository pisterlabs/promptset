"""
    Warning:

        This will delete all your files and fine-tunings

        If that's what you want, here's the code for it!

"""
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


def delete_files():
    flist = openai.File.list()
    # print(flist)
    # print()

    for f in flist['data']:
        if f['status'] != 'processed':
            print(f'Cannot (yet) delete file {f["id"]} with status {f["status"]} for purpose {f["purpose"]}')
            continue

        print(f'Deleting {f["id"]}')
        openai.File.delete(f['id'])

#   Note that tunings are distinct from fine-tuning jobs, so what we have to
#   do is loop through the tuning jobs and delete the models they generated
def delete_tunings():
    tunes = openai.FineTuningJob.list()
    # print(tunes)
    for t in tunes['data']:
        try:
            openai.Model.delete(t["fine_tuned_model"])
            print(f'Deleted id={t["id"]}, status={t["status"]}, results={t["fine_tuned_model"]}')
        except openai.error.InvalidRequestError as e:
            pass    # Does not exist, so ignore error

print('*** WARNING ***')
print('This will delete all files and all private models on your OpenAI account.')
yesno = input('Are you really sure? [N/y]: ')
if yesno.upper() == 'Y':
    delete_files()
    delete_tunings()

