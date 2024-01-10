import openai

print("")
for finetuning in openai.FineTuningJob.list().data:
	print("___________________________________________________________________\n")
	print(f'Model name: {finetuning.fine_tuned_model}')
	print(f'File name: {openai.File.retrieve(finetuning.training_file).filename}\n')
	
print("___________________________________________________________________")