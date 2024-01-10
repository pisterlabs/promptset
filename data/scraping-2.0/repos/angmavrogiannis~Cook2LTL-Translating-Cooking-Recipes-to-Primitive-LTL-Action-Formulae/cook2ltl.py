from sentence_transformers import SentenceTransformer, util
from string import punctuation
from tqdm import tqdm
import openai
import statistics
import spacy
import pickle
import time
import copy
import csv
import re

class Cook2LTL():
	def __init__(self, example_functions, primitive_imports, similarity_threshold, model_embedding, model_spacy, ner_model):
		self.example_functions = example_functions
		self.primitive_imports = primitive_imports
		self.primitive_actions = [item.replace("<obj>", "").replace("<time>", "").strip() for item in self.primitive_imports]
		self.action_library = {}
		self.model_embedding = model_embedding
		self.similarity_threshold = similarity_threshold
		self.primitive_embs = self.get_primitive_word_embeddings()
		self.model_spacy = model_spacy
		self.ner_model = ner_model
		self.ner_labels = ["VERB", "WHAT", "WHERE", "HOW", "TIME", "TEMP"]
		self.num_api_calls = 0
		self.total_latency = 0
		self.total_cost = 0
		self.num_generated_actions = []
		self.executability = []
		self.llm_output = {}
		self.num_substitutions = 0
		self.prim_substitutions = 0
		self.substitutions = {}

	"""
	Saves action library dictionary as a pickle object.
	"""
	def save_action_library(self):
		with open("action_library.pkl", "wb") as f:
			pickle.dump(self.action_library, f)

	"""
	Saves llm output to file
	"""
	def save_llm_output(self):
		with open("ar_primitive.pkl", "wb") as f:
			pickle.dump(self.llm_output, f)

	"""
	Loads action library dictionary from a saved pickle object.
	"""
	def load_action_library(self):
		with open("action_library.pkl", "rb") as f:
			self.action_library = pickle.load(f)

	"""
	Reads recipes from CSV file
	"""
	def read_recipes(self, csv_file_path):
		recipes = []
		with open(csv_file_path, mode="r") as f:
			csv_reader = csv.reader(f)
			next(csv_reader)
			for recipe in csv_reader:
				recipes.append(recipe[-1])
		return recipes

	"""
	Segments recipe to sentences based on delimiters
	Current delimiters: ; ! .
	"""
	def preprocess(self, recipes):
		# segment into sentences
		in_formula_delimiters = [";", "!"]
		formula_delimiter = "."
		sentence_recipes = [item.split(formula_delimiter) for item in recipes]
		splits = []
		# sentence_recipes = [item for sublist in sentence_recipes for item in sublist]
		for recipe in sentence_recipes:
			split = [re.split(';|!', item) for item in recipe]
			split = [item for sublist in split for item in sublist]
			split = [item.strip() for item in split]
			split = [item.lower() for item in split if item]
			splits.append(split)
		return splits

	"""
	Checks compatibility of actions before matching a new action to an action from the action library.
	"""
	def action_compatibility(self, runtime_action_dict, cached_action_dict):
		runtime_type = {}
		cached_type = {}
		for (k1,v1), (k2,v2) in zip(runtime_action_dict.items(), cached_action_dict.items()):
			if v1:
				runtime_type[k1] = True
			else:
				runtime_type[k1] = False
			if v2:
				cached_type[k2] = True
			else:
				cached_type[k2] = False
		return runtime_type == cached_type

	"""
	Converts dictionary with NER tags to a pythonic function representation string (name + parameters)
	"""
	def dict_to_action_function(self, ner_dict):
		assert ner_dict["VERB"], "No action verb detected"
		action_function = f"def {', '.join(ner_dict['VERB'])}("

		if "WHAT" in ner_dict and ner_dict["WHAT"]:
			action_function += f"{', '.join(ner_dict['WHAT'])}: what"
		if "WHERE" in ner_dict and ner_dict["WHERE"]:
			action_function += f", {', '.join(ner_dict['WHERE'])}: where"
		if "HOW" in ner_dict and ner_dict["HOW"]:
			action_function += f", {', '.join(ner_dict['HOW'])}: how"
		if "TIME" in ner_dict and ner_dict["TIME"]:
			action_function += f", {', '.join(ner_dict['TIME'])}: time"
		if "TEMP" in ner_dict and ner_dict["TEMP"]:
			action_function += f", {', '.join(ner_dict['TEMP'])}: temp"

		action_function += ")"
		return action_function

	"""
	Uses the trained NER model to assign the following tags to the parts of a recipe step:
		("VERB", "WHAT", "WHERE", "HOW", "TIME", "TEMP")
	"""
	def get_ner_tags(self, sentences):
		action_dicts = []
		for recipe in recipes:
			recipe_dicts = []
			for sentence in recipe:
				ner_doc = self.ner_model(sentence)
				action_dict = {label: [] for label in self.ner_labels}
				entities = [(ent.text, ent.label_) for ent in ner_doc.ents]
				for tup in entities:
					action_dict[tup[1]].append(tup[0])
				recipe_dicts.append(action_dict)
			action_dicts.append(recipe_dicts)
		return action_dicts

	"""
	Gets the word embedding of an action based on the overall context of the sentence.
	Note: The first token embedding corresponds to the [CLS] token automatically added by transformers
	For words split into subtokens, we take the average of the derived sub-embeddings.
	"""
	def get_contextual_word_embedding(self, sentence, word):
		sentence_no_punct = sentence.translate(str.maketrans("", "", punctuation))
		word_index = sentence_no_punct.index(word)
		token_embeddings = self.model_embedding.encode(sentence_no_punct, output_value="token_embeddings")
		inputs = self.model_embedding.tokenizer(sentence_no_punct)
		matches = [index for index in inputs.word_ids() if index == word_index]
		assert len(matches) > 0, "The given word does not match the given sentence."
		if len(matches) == 1:
			word_embedding = token_embeddings[matches[0]]
		else:
			subword_embeddings = []
			for match in matches:
				subword_embeddings.append(token_embeddings[match])
			word_embedding = sum(subword_embeddings) / len(subword_embeddings)
		return word_embedding

	"""
	Returns the most similar action from a set of cached actions (primitive set/action library) to a given action
	if the cosine similarity exceeds the similarity threshold.
	"""
	def word_similarity(self, runtime_action_dict, cached_actions):
		verb = runtime_action_dict["VERB"][0]
		what = runtime_action_dict["WHAT"]
		sentence = " ".join([item for sublist in list(runtime_action_dict.values()) for item in sublist])
		contextual_embedding = self.get_contextual_word_embedding(sentence, verb)
		global_embedding = self.model_embedding.encode(verb)
		similar_action = None
		for action in cached_actions:
			contextual_sim = util.cos_sim(contextual_embedding, self.primitive_embs[action])
			global_sim = util.cos_sim(global_embedding, self.primitive_embs[action])
			max_sim = 0
			if contextual_sim > self.similarity_threshold or global_sim > self.similarity_threshold:
				max_sim = {action: max(contextual_sim, global_sim)}
				similar_action = action
			if similar_action is not None:
				print(f"unseen word: {verb} --> primitive word {similar_action}")
		return similar_action

	"""
	Computes global word embeddings of primitive actions.
	"""
	def get_primitive_word_embeddings(self):
		embs = {}
		for action in self.primitive_actions:
			embs[action] = self.model_embedding.encode(action)
		return embs

	"""
	Detects conjunction, disjunction, and negation, and splits formula into separate formulae.
	TO-DOs: 1. handle disjunction outside of chunks
	"""
	def parse_chunks(self, recipe_dicts):
		all_dicts = []
		intermediate_action_dicts = []
		ltl_operators = []
		action_dicts = []
		disj_words = ["or"]
		conj_words = ["and"]
		neg_words = ["not", "don't", "never", "dont"]
		conj_exclusions = ["mac and cheese", "macaroni and cheese", "fish and chips"]
		for recipe_dict_list in recipe_dicts:
			per_recipe_dicts = []
			for individual_dict in recipe_dict_list:
				disjunction = {k: False for k in self.ner_labels}
				conjunction = {k: False for k in self.ner_labels}
				action_dict = {key: ", ".join(val) for key, val in individual_dict.items()}
				neg = False
				disj = False
				conj = False
				for neg_word in neg_words:
					if neg_word in action_dict["VERB"]:
						neg = True
				for label, chunk in action_dict.items():
					doc = self.model_spacy.tokenizer(chunk)
					for token in doc:
						if token.text in disj_words:
							disjunction[label] = True
						if token.text in conj_words:
							conjunction[label] = True
				for label, has_disj in disjunction.items():
					if has_disj:
						disj = True
						disj_parts = [item.strip() for item in action_dict[label].split(" or ")]
						for part in disj_parts:
							derived_action_dict = copy.deepcopy(action_dict)
							derived_action_dict[label] = part
							intermediate_action_dicts.append(derived_action_dict)
							intermediate_action_dicts.append("OR")
				if not disj:
					intermediate_action_dicts = [individual_dict]
				if conjunction["VERB"]:
					if disj and intermediate_action_dicts[-1] == "OR":
						intermediate_action_dicts.pop()
					for derived_dict in intermediate_action_dicts:
						if derived_dict != "OR":
							conj_parts = [item.strip() for item in derived_dict["VERB"][0].split(" and ")]
							and_dicts = []
							for part in conj_parts:
								derived_action_dict = copy.deepcopy(derived_dict)
								derived_action_dict["VERB"] = part
								and_dicts.append(derived_action_dict)
							if disj:
								final_dicts.append(and_dicts)
								final_dicts.append("OR")
							else:
								final_dicts = and_dicts
				if disj:
					final_dicts = intermediate_action_dicts
				if disj and final_dicts[-1] == "OR":
					final_dicts.pop()
				if not conj and not disj:
					final_dicts = individual_dict
				if neg:
					neg_dicts = []
					for j in range(final_dicts):
						neg_dicts.append("NOT")
						neg_dicts.append(final_dicts[j])
					final_dicts = neg_dicts
				per_recipe_dicts.append(final_dicts)
			all_dicts.append(per_recipe_dicts)
		return all_dicts

	"""
	Reduces a high-level action to a set of primitive actions
	"""
	def action_reduction(self, prompt_tuple):
		# prepare prompt with sentence as comment, function definition, and available objects
		prompt, action_function, action_dict = prompt_tuple[0], prompt_tuple[1], prompt_tuple[2]
		start_time = time.time()
		response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], temperature=0.7)
		elapsed_time = time.time() - start_time
		# if response and response["choices"][0]["message"]["content"]:
		if True:
			self.total_latency += elapsed_time
			self.num_api_calls += 1
			self.total_cost += self.compute_cost(response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"])
			# parse output into function
			text = response["choices"][0]["message"]["content"]
			print(f"RUN\n")
			print(text)
			self.llm_output[action_function] = text
			lines = text.split("\n")
			lines = [line.strip("\t") for line in lines]
			functions = [line for line in lines if not line.startswith("#")]
			functions = [function for function in functions if function]
			comments = [line for line in lines if line.startswith("#")]
			verbs = []
			params = []
			for function in functions:
				find_verb = re.search("^[^\(]+", function)
				if find_verb:
					f_verb = find_verb.group(0).strip()
					verbs.append(f_verb)
				f_params = function[function.find("(")+1:function.find(")")]
				f_params = [param.strip() for param in f_params.split(",")]
				params.append(f_params)
			self.num_generated_actions.append(len(verbs))
			self.cache_action(action_dict, verbs, params)
			return verbs, "\n".join(functions)

	"""
	Adds action to action library and to pythonic import for prompting
	"""
	def cache_action(self, action_dict, verbs, params):
		non_primitive_found = False
		if verbs:
			for generated_verb in verbs:
				if generated_verb not in self.primitive_actions and generated_verb not in self.action_library:
					non_primitive_found = True
					break
		if not non_primitive_found:
			new_verb = action_dict["VERB"][0]
			param_type =[]
			for k, param_list in enumerate(params):
				param_type.append([])
				for param in param_list:
					if param in " ".join(action_dict["WHAT"]):
						param_type[k].append("WHAT")
					elif param in " ".join(action_dict["WHERE"]):
						param_type[k].append("WHERE")
					elif param in " ".join(action_dict["HOW"]):
						param_type[k].append("HOW")
					elif param in " ".join(action_dict["TIME"]):
						param_type[k].append("TIME")
					elif param in " ".join(action_dict["TEMP"]):
						param_type[k].append("TEMP")
					else:
						param_type[k].append("OTHER")
			if new_verb in self.action_library:
				self.action_library[new_verb].append({"verbs": verbs, "params": params, "param_type": param_type, "action_dict": action_dict})
			else:
				self.action_library[new_verb] = [{"verbs": verbs, "params": params, "param_type": param_type, "action_dict": action_dict}]
				# new_import = f"{new_verb} {len([l for l in action_dict.values() if l]) * '<obj>'}"
				new_import = f"{new_verb}"
				# if action_dict["TIME"]:
				# 	new_import = new_import[:-5]+"<time>"
				self.primitive_imports.append(new_import)

	"""
	Looks up the definition of a function in the action library and adapts it to a newly seen action.
	Schema of cached_dict: {"verbs": verbs, "params": params, "param_type": param_type, "action_dict": action_dict}
	"""
	def reuse_cached_action(self, verb, runtime_action_dict, cached_dict):
		new_fun = []
		for j, cached_param_list in enumerate(cached_dict["param_type"]):
			new_params = []
			for param in cached_param_list:
				if param == "OTHER":
					continue
				new_params.append(", ".join(runtime_action_dict[param]))
			atomic_function = f"{cached_dict['verbs'][j]}({', '.join(new_params)})"
			new_fun.append(atomic_function)
		adapted_function_body = "\n".join(new_fun)
		# self.substitutions[] = adapted_function_body
		print("RUN\n")
		print(runtime_action_dict)
		print(adapted_function_body)
		return adapted_function_body

	"""
	Creates prompt for querying LLM towards action reduction
	"""
	def create_prompt(self, action_function, action_dict):
		task_description = "Complete the function at the bottom only using actions from the imported actions and objects from the available objects. You can only pick up one object at a time."
		return f"{task_description}\n\nfrom actions import {', '.join(self.primitive_imports)}\n\n{self.example_functions}\n{action_function}:\n\tavailable_objects = [microwave, potato]", action_function, action_dict # [fridge, apple] \n\tavailable_objects = [lettuce]

	"""
	Translates a recipe in natural language to an LTL formula
	"""
	def decode_LTL(self, sent):
		pass
	# 	for i in range(len(actions))	:
	# 		if i != len(actions) - 1 and all(sequence[i] == i for sequence in sequences):
	# 			formula += f"F({actions[i]} ^ "
	# 			action_queue += 1
	# 		else:
	# 			formula += f"F{actions[i]} ^ "
	# 			if action_queue > 0:
	# 				formula = formula[:-2] + action_queue * ")"
	# 				action_queue = 0
	# 	if formula[-2] == "^":
	# 		formula = formula[:-2]


	"""
	Computes the executability of a generated action plan
	Executability: fraction of actions in the plan that are executable
	in the environment, even if they are not relevant for the task
	"""
	def compute_executability(self, plan):
		num_actions = len(plan)
		executable_actions = 0
		for action in plan:
			if action in self.primitive_actions or action in self.action_library:
				executable_actions += 1
		executability = executable_actions / num_actions
		self.executability.append(executability)

	"""
	Success rate (SR) is the fraction of executions that achieved
	all task-relevant goal-conditions
	"""
	def success_rate(self, success_conditions, final_object_states):
		success = True
		for condition in success_conditions:
			if condition not in final_object_states:
				success = False
				break

	"""
	Computes total cost of API calls
	For gpt-3.5-turbo:
	- input: 1.5e-06 * num_input_tokens
	- output: 2e-06 * num_output_tokens
	"""
	def compute_cost(self, prompt_tokens, completion_tokens):
		input_token_cost = 1.5e-06
		output_token_cost = 2e-06
		return prompt_tokens * input_token_cost + completion_tokens * output_token_cost

	"""
	Prints metric after a batch of experiments.
	"""
	def print_metrics(self):
		print(f"Total number of API calls: {self.num_api_calls} calls\n")
		print(f"Total time/latency: {round(self.total_latency, 2)} seconds")
		if self.total_cost >= 0.01:
			print(f"Total cost of API calls: {round(self.total_cost, 2)} $\n")
		else:
			print(f"Total cost of API calls: {self.total_cost} $\n")
		print(f"Average time per API call: {round(self.total_latency / self.num_api_calls, 2)}")
		cost_per_call = self.total_cost / self.num_api_calls
		if cost_per_call >= 0.01:
			print(f"Average cost per API call: {round(cost_per_call, 2)}\n")
		else:
			print(f"Average cost per API call: {cost_per_call}\n")
		print(f"Average executability: {round(statistics.mean(self.executability), 2)}\n")
		print(f"Average length of generated plan (number of generated actions): {round(statistics.mean(self.num_generated_actions), 2)}")
		# print(f"Number of substitutions: {self.num_substitutions}")
		print(f"Number of substitutions: {self.prim_substitutions}")
		# self.save_action_library()



if __name__ == "__main__":
	sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
	spacy_model = spacy.load("en_core_web_md")
	ner_model = spacy.load("ner_500_md")
	openai.organization = ""
	openai.api_key = ""
	csv_file_path = "edited_recipes.csv"
	prompt_path = "ai2thor_prompt.txt"
	prompt_file = open(prompt_path, "r")
	example_functions = prompt_file.read()

	# Beetz most frequent: 1. Add/Combine, 2. Pick/Place, 3. Fill/Pour, 4. Remove, 5. Stir/Beat, 6. Serve, 7. Mix/Blend ()
	# 8. Bake, 9. Cook/Simmer/Boil, 10. Cut/Chop/Slice, 11. Sprinkle, 12. Flip/Turn Over, 13. Regrigerate/Cool/Freeze, 14. Shake, 15. Wait
	# paper: Everything Robots Always Wanted to Know about Housework (But were afraid to ask)
	# primitive_imports = ["stir <obj>", "cut <obj>", "pour <obj><obj>", "turn <obj><obj>", "shake <obj>", "pick up <obj>", "put <obj><obj>", "remove <obj><obj>", "open <obj>", "close <obj>", "turn on <obj>", "turn off <obj>", "taste <obj>", "wait <time>"]
	
	primitive_imports = ["pick up <obj>", "put <obj>", "drop <obj>","open <obj>", "close <obj>", "break <obj>", "cook <obj>", "slice <obj>", "turn on <obj>", "turn off <obj>","use <obj>", "fill <obj>", "empty <obj>", "wait <time>"]
	gen_actions = {"use", "set", "let", "leave", "do", "keep", "allow", "cook", "make", "prepare", "set", "use"}
	similarity_threshold = 0.6

	learned_actions = {}
	available_objects = ["oven", "microwave", "pot", "pan", "bowl", "refrigerator"]
	
	cook = Cook2LTL(example_functions, primitive_imports, similarity_threshold, sbert_model, spacy_model, ner_model)
	recipes = cook.read_recipes(csv_file_path)
	recipes = cook.preprocess(recipes)
	action_dicts = cook.get_ner_tags(recipes)
	# filtering out examples where the NER did not detect a verb
	action_dicts = [[action_dict for action_dict in item if action_dict["VERB"] != [] and action_dict["WHAT"] != []] for item in action_dicts]
	action_dicts = cook.parse_chunks(action_dicts)
	# for recipe_num, recipe in tqdm(enumerate(action_dicts)):
		# plan = []
	# 	for action_dict in recipe:
	# 		if action_dict != "OR":
	# recipes = [["Refrigerate the apple."]]
	# action_dict = cook.get_ner_tags(recipes)
	plan = []
	for i in range(10):
		if type(action_dict) == dict:
			action_function = cook.dict_to_action_function(action_dict)
			# if cook.word_similarity(action_dict, cook.primitive_actions) is not None:
			# 	breakpoint()
			# if action_dict["VERB"][0] not in cook.primitive_actions:
			# 	sim_word = cook.word_similarity(action_dict, cook.primitive_actions)
			verb = action_dict["VERB"][0]
			query_llm = True
			if verb in cook.primitive_actions:
				plan.append(action_function[4:])
				cook.executability.append(1)
				cook.prim_substitutions += 1
				query_llm = False
			elif verb in cook.action_library:
				for implementation in cook.action_library[verb]:
					if cook.action_compatibility(action_dict, implementation["action_dict"]):
						plan.append(cook.reuse_cached_action(verb, action_dict, implementation))
						cook.num_substitutions += 1
						cook.executability.append(1)
						query_llm = False
			if query_llm:
				prompt = cook.create_prompt(action_function, action_dict)
				verbs, gen_subplan = cook.action_reduction(prompt)
				plan.append(gen_subplan)
				cook.compute_executability(verbs)
		elif type(action_dict) == list:
			action_dict.pop(0)
			for subdict in action_dict:
				if subdict != "OR":
					adjusted_subdict = {key: [val] for key, val in subdict.items()}
					action_function = cook.dict_to_action_function(adjusted_subdict)
					verb = adjusted_subdict["VERB"][0]
					query_llm = True
					if verb in cook.primitive_actions:
						plan.append(action_function[4:])
						cook.executability.append(1)
						cook.prim_substitutions += 1
						query_llm = False
					elif verb in cook.action_library:
						for implementation in cook.action_library[verb]:
							if cook.action_compatibility(adjusted_subdict, implementation["action_dict"]):
								plan.append(cook.reuse_cached_action(verb, adjusted_subdict, implementation))
								cook.num_substitutions += 1
								cook.executability.append(1)
								query_llm = False
					if query_llm:
						prompt = cook.create_prompt(action_function, adjusted_subdict)
						verbs, gen_subplan = cook.action_reduction(prompt)
						plan.append(gen_subplan)
						cook.compute_executability(verbs)
	cook.print_metrics()
	cook.save_llm_output()
