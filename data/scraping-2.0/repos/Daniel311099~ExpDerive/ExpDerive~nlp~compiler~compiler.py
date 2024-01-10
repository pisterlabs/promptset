
from typing import Union
import openai
from ExpDerive.nlp.compiler.astree import AST, ALeaf

from .classifiers import ArgClassifier, FuncClassifier
from .locator import ArgLocator, FuncLocator, VarGPTModel, FuncGPTModel


class Compiler():
    def __init__(self, funcLocator: FuncLocator, argLoacator: ArgLocator, argClassifier: ArgClassifier, funcClassifier: FuncClassifier) -> None:
        self.argClassifier = argClassifier
        self.funcClassifier = funcClassifier
        self.funcLocator = funcLocator  
        self.argLoacator = argLoacator

    def buildAst(self, phrase: str) -> Union[AST, ALeaf]:
        # embedding = self.getEmbedding(phrase)
        # check match from vector db if phrase is a known variable
        # return ALeaf(name) if match found or len(phrase.split(" ")) == 1
        print("building ast for", phrase)
        if len(phrase.split(" ")) == 1:
            label = self.argClassifier.classify(phrase, force=True)
            print("phrase is a single word: " + label)
            return ALeaf(label)
        candidate = self.argClassifier.classify(phrase)
        if candidate is not None:
            print("phrase is a variable: " + candidate)
            return ALeaf(candidate)
        # only passes if internal node
        root_func = self.funcLocator.getFunc(phrase)
        if root_func.loc[1] - root_func.loc[0] == 0:
            print("root function is a single word: " + root_func.name)
            label = self.argClassifier.classify(phrase, force=True)
            return ALeaf(label)
        print("root function is: ", root_func)
        root_func = self.funcClassifier.classify(root_func)
        print("root function is: " + root_func.name)
        # func_params = []
        args = self.argLoacator.getArgs(phrase, root_func)
        print("args are: " + str(args))
        if args[0] == phrase:
            label = self.argClassifier.classify(phrase, force=True)
            print("phrase is a variable: " + label)
            return ALeaf(label)

        # arg_labels = [
        #     (arg, self.argClassifier.classify(arg))
        #     for arg in args
        # ]
        ast = AST(root_func)
        for arg in args:
            # if self.argClassifier.is_arg(arg):
            #     continue
            # if arg[1] is None:
            ast.args.append(self.buildAst(arg))
            #     continue
            # ast.args.append(ALeaf(arg))
        return ast
    
    def compile(self, phrase: str) -> str:
        print("compiling", phrase)
        print("building ast")
        ast = self.buildAst(phrase)
        print("generating Latex")
        return ast.build()

    def getEmbedding(self, phrase):
        response = openai.Embedding.create(
            input=[phrase],
            model="text-embedding-ada-002,"
        )
        return response['data'][0]['embedding']

# todo:
# - generate more training data
# - parse each example and generate the correct output including the start and stop tokens
# - write to jsonl files
# - fine tune the model
# - only one function locator, infer type from the output
# - 

# - add methods to populate vector db in db class
# - populate db
# - fix classifier

class CompilerBuilder():
    def __init__(self):
        self.funcLocator = FuncLocator(model=FuncGPTModel(api_key="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"))
        self.argLoacator = ArgLocator(model=VarGPTModel(api_key="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"))
        self.argClassifier = ArgClassifier()
        self.funcClassifier = FuncClassifier()

    def build(self) -> Compiler:
        return Compiler(self.funcLocator, self.argLoacator, self.argClassifier, self.funcClassifier)